"""Drive-day panel experiment: 30-day history, 7-day failure horizon.

One row per eligible drive per prediction date over 2026-01-01 to 2026-03-31,
read directly from the quarterly ZIP archive.

For a prediction date ``t``, features come from the 30 days strictly before
``t`` (``[t-30, t-1]``); day ``t`` itself is excluded. The label is positive
when the drive's failure date falls in ``[t+1, t+7]``. Rows are dropped when
either window would run past the edge of the quarter, when the drive has
already failed on or before ``t``, or when survival across the future window
cannot be verified. Eligible dates are 2026-01-31 to 2026-03-24.

Memory: 53 dates x ~337k drives is ~17.9M rows, or 3.6 GB of float32 features,
so nothing is materialised. Two cheap metadata passes build a drive index,
presence bitmap and failure days. A ``(7 attributes, 30 days, n_drives)``
float32 ring buffer (~294 MB) holds the rolling window. Training keeps every
positive and a sample of negatives; validation and test are scored per
prediction date, retaining only scores and drive ids, so their metrics come
from complete folds.

Negative downsampling applies to training only. It shifts the model's
probability scale, which is harmless because thresholds come from the full
validation fold.

A drive contributes up to 53 rows and a failing drive up to 7 positive rows, so
drive-day metrics overstate the number of independent observations. Event-level
metrics are reported alongside: a failure counts as detected if any alert fires
during its 7-day pre-failure window.

Usage:
    python -m src.panel_experiment --archive ~/Downloads/data_Q1_2026.zip
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_pipeline import (
    DataValidationError,
    _read_csv_frame,
    configure_logging,
    open_csv_source,
)
from src.experiment import SMART_ATTRIBUTES, STATISTICS

LOGGER = logging.getLogger(__name__)

WINDOW_DAYS = 30
HORIZON_DAYS = 7
RANDOM_STATE = 42
TRAIN_FRACTION = 0.6
VALIDATION_FRACTION = 0.2

#: Share of negatives kept for training. All positives are always kept.
NEGATIVE_SAMPLE_RATE = 0.04

#: Selects every daily file in the quarter.
ALL_DAYS_GLOB = "2026-*.csv"

#: Minimum test-fold failure events before selection is attempted.
MIN_EVENTS_FOR_SELECTION = 50

_START = time.perf_counter()


def say(message: str) -> None:
    """Print a progress line immediately.

    Args:
        message: Text to print.
    """
    print(f"[{time.perf_counter() - _START:7.1f}s] {message}", flush=True)


def feature_names() -> List[str]:
    """Return the generated feature names in deterministic order."""
    names = [f"{a}_{s}" for a in SMART_ATTRIBUTES for s in STATISTICS]
    return names + ["observations_30d"]


FEATURE_NAMES = feature_names()
RULE_FEATURES = ("smart_197_raw_max", "smart_5_raw_change", "smart_198_raw_max")


# ---------------------------------------------------------------------------
# Metadata passes
# ---------------------------------------------------------------------------


@dataclass
class Calendar:
    """The archive's daily files, in chronological order.

    Attributes:
        dates: One timestamp per daily file.
        names: Archive member names aligned to ``dates``.
    """

    dates: List[pd.Timestamp]
    names: List[str]

    def __len__(self) -> int:
        """Return the number of days covered."""
        return len(self.dates)

    def label(self, day: int) -> str:
        """Return the ISO date string for a day index."""
        return str(self.dates[day].date())


def scan_calendar(archive: Path, file_glob: str = ALL_DAYS_GLOB) -> Calendar:
    """List the archive's daily files and verify the dates are contiguous.

    Args:
        archive: Archive or directory of daily files.
        file_glob: Pattern selecting the daily files.

    Returns:
        The :class:`Calendar`.

    Raises:
        DataValidationError: If a filename is not a parseable date, or a
            calendar day is missing (a gap corrupts every window spanning it).
    """
    with open_csv_source(archive, file_glob) as source:
        labels = [source.label(name) for name in source.names]
        names = list(source.names)

    dates: List[pd.Timestamp] = []
    for label in labels:
        try:
            dates.append(pd.Timestamp(label.replace(".csv", "")))
        except ValueError as exc:
            raise DataValidationError(
                f"Cannot parse a date from archive member {label!r}"
            ) from exc

    order = np.argsort(np.array(dates))
    dates = [dates[i] for i in order]
    names = [names[i] for i in order]

    expected = pd.date_range(dates[0], dates[-1], freq="D")
    if len(expected) != len(dates) or any(a != b for a, b in zip(expected, dates)):
        missing = sorted(set(expected) - set(dates))
        raise DataValidationError(
            f"The daily files are not contiguous; {len(missing)} day(s) missing, "
            f"first few: {[str(d.date()) for d in missing[:5]]}"
        )

    return Calendar(dates=dates, names=names)


@dataclass
class DriveMetadata:
    """Per-drive facts needed for eligibility and labelling.

    Attributes:
        index: Global drive index; position in it identifies a drive.
        presence: ``(n_days, n_drives)`` bool, whether the drive reported.
        failure_day: Day index of each drive's first ``failure=1`` row, or
            ``-1`` if it never failed inside the quarter.
        rows_per_day: Row count per daily file.
    """

    index: pd.Index
    presence: np.ndarray
    failure_day: np.ndarray
    rows_per_day: List[int]

    @property
    def n_drives(self) -> int:
        """Return the number of distinct drives."""
        return len(self.index)

    @property
    def total_rows(self) -> int:
        """Return the total number of drive-day rows in the archive."""
        return int(sum(self.rows_per_day))


def scan_drives(archive: Path, calendar: Calendar) -> DriveMetadata:
    """Build the global drive index, presence bitmap and failure dates.

    Reads only ``serial_number`` and ``failure``.

    Args:
        archive: Archive or directory of daily files.
        calendar: The archive's calendar.

    Returns:
        The assembled :class:`DriveMetadata`.
    """
    known = pd.Index([], dtype=object)
    per_day_positions: List[np.ndarray] = []
    per_day_failed: List[np.ndarray] = []
    rows_per_day: List[int] = []

    with open_csv_source(archive, ALL_DAYS_GLOB) as source:
        handles = {source.label(name): name for name in source.names}
        say(f"Metadata pass: {source.describe()}")

        for day in range(len(calendar)):
            member = handles[calendar.label(day) + ".csv"]
            frame, _ = _read_csv_frame(
                lambda n=member: source.open(n), calendar.label(day),
                ("serial_number", "failure"),
            )
            frame = frame.dropna(subset=["serial_number"])
            serials = frame["serial_number"].astype(str).to_numpy()
            rows_per_day.append(len(serials))

            positions = known.get_indexer(serials)
            unseen = positions == -1
            if unseen.any():
                fresh = pd.Index(pd.unique(serials[unseen]), dtype=object)
                known = known.append(fresh)
                positions = known.get_indexer(serials)

            per_day_positions.append(positions.astype(np.int32))
            per_day_failed.append(
                positions[(frame["failure"] == 1).to_numpy()].astype(np.int32)
            )

            if (day + 1) % 15 == 0 or day + 1 == len(calendar):
                say(
                    f"  [{day + 1:02d}/{len(calendar)}] {calendar.label(day)}: "
                    f"{len(serials):,} rows | {len(known):,} drives known"
                )
            del frame, serials
            gc.collect()

    n_drives = len(known)
    presence = np.zeros((len(calendar), n_drives), dtype=bool)
    failure_day = np.full(n_drives, -1, dtype=np.int16)

    for day, positions in enumerate(per_day_positions):
        presence[day, positions] = True
    for day, failed in enumerate(per_day_failed):
        if failed.size:
            unset = failure_day[failed] == -1
            failure_day[failed[unset]] = day

    del per_day_positions, per_day_failed
    gc.collect()

    return DriveMetadata(
        index=known,
        presence=presence,
        failure_day=failure_day,
        rows_per_day=rows_per_day,
    )


# ---------------------------------------------------------------------------
# Rolling window over SMART telemetry
# ---------------------------------------------------------------------------


class WindowBuffer:
    """Ring buffer holding the most recent ``window`` days of SMART readings.

    Shaped ``(n_attributes, window, n_drives)`` so one attribute's window is a
    contiguous slice, avoiding a whole-buffer copy per prediction date.
    """

    def __init__(self, n_attributes: int, window: int, n_drives: int) -> None:
        """Allocate the buffer.

        Args:
            n_attributes: Number of SMART attributes tracked.
            window: Window length in days.
            n_drives: Number of drives in the global index.
        """
        self.window = window
        self.buffer = np.full(
            (n_attributes, window, n_drives), np.nan, dtype=np.float32
        )
        self._filled = 0

    def write_day(self, day: int, positions: np.ndarray, values: np.ndarray) -> None:
        """Write one day's readings into its slot, clearing the evicted day.

        Args:
            day: Absolute day index.
            positions: Drive positions present that day.
            values: ``(n_present, n_attributes)`` readings.
        """
        slot = day % self.window
        # Clear the evicted day so stale readings cannot survive in this slot.
        self.buffer[:, slot, :] = np.nan
        # Basic slicing gives a view, so this writes through.
        self.buffer[:, slot, :][:, positions] = values.T
        self._filled = min(self._filled + 1, self.window)

    def chronological_order(self, day: int) -> np.ndarray:
        """Return slot indices for the window ending at ``day``, oldest first.

        Args:
            day: Absolute index of the most recent day in the buffer.

        Returns:
            Slot indices in chronological order.
        """
        return np.array(
            [(day - self.window + 1 + k) % self.window for k in range(self.window)]
        )

    def aggregate(self, day: int) -> Dict[str, np.ndarray]:
        """Summarise the window ending at ``day`` into per-drive statistics.

        Args:
            day: Absolute index of the most recent day in the window.

        Returns:
            Feature name to a ``(n_drives,)`` float32 array.
        """
        order = self.chronological_order(day)
        out: Dict[str, np.ndarray] = {}
        n_drives = self.buffer.shape[2]
        row_indices = np.arange(n_drives)

        with warnings.catch_warnings():
            # All-NaN slices are expected for drives absent all window.
            warnings.simplefilter("ignore", RuntimeWarning)

            for position, attribute in enumerate(SMART_ATTRIBUTES):
                window = np.take(self.buffer[position], order, axis=0)
                valid = ~np.isnan(window)
                count = valid.sum(axis=0)
                any_valid = count > 0

                first_index = np.argmax(valid, axis=0)
                last_index = self.window - 1 - np.argmax(valid[::-1], axis=0)
                first = window[first_index, row_indices]
                last = window[last_index, row_indices]
                first = np.where(any_valid, first, np.nan)
                last = np.where(any_valid, last, np.nan)

                out[f"{attribute}_latest"] = last
                out[f"{attribute}_mean"] = np.nanmean(window, axis=0)
                out[f"{attribute}_min"] = np.nanmin(window, axis=0)
                out[f"{attribute}_max"] = np.nanmax(window, axis=0)
                out[f"{attribute}_std"] = np.nanstd(window, axis=0, ddof=1)
                out[f"{attribute}_change"] = last - first
                out[f"{attribute}_count"] = count.astype(np.float32)

                del window, valid
                gc.collect()

        return out


# ---------------------------------------------------------------------------
# Panel streaming
# ---------------------------------------------------------------------------


@dataclass
class PanelCounts:
    """Row-level accounting for the whole panel."""

    prediction_dates: int = 0
    rows_emitted: int = 0
    positives: int = 0
    events: int = 0
    excluded_already_failed: int = 0
    excluded_unverifiable: int = 0
    excluded_no_history: int = 0
    drives_seen: int = 0


@dataclass
class Block:
    """One prediction date's emitted rows.

    Attributes:
        day: Absolute day index of the prediction date.
        features: ``(m, n_features)`` float32 design matrix.
        labels: ``(m,)`` int8 labels.
        drives: ``(m,)`` int32 positions in the global drive index.
    """

    day: int
    features: np.ndarray
    labels: np.ndarray
    drives: np.ndarray


def eligible_prediction_days(calendar: Calendar) -> List[int]:
    """List day indices with both a full history and a full label window.

    Args:
        calendar: The archive's calendar.

    Returns:
        Eligible prediction-date day indices.
    """
    return [
        day
        for day in range(len(calendar))
        if day >= WINDOW_DAYS and day + HORIZON_DAYS <= len(calendar) - 1
    ]


def stream_blocks(
    archive: Path,
    calendar: Calendar,
    metadata: DriveMetadata,
    wanted_days: Sequence[int],
    counts: Optional[PanelCounts] = None,
    quiet: bool = False,
) -> Iterator[Block]:
    """Yield one :class:`Block` per wanted prediction date.

    Reading starts ``WINDOW_DAYS`` before the earliest wanted date so the ring
    buffer is warm.

    Args:
        archive: Archive or directory of daily files.
        calendar: The archive's calendar.
        metadata: Drive index, presence bitmap and failure days.
        wanted_days: Prediction-date day indices to emit.
        counts: Optional accumulator updated as rows are emitted.
        quiet: Suppress per-date progress lines.

    Yields:
        One block per wanted prediction date, in chronological order.
    """
    wanted = sorted(set(wanted_days))
    if not wanted:
        return

    first_needed = max(0, wanted[0] - WINDOW_DAYS)
    buffer = WindowBuffer(len(SMART_ATTRIBUTES), WINDOW_DAYS, metadata.n_drives)
    wanted_set = set(wanted)
    columns = ("serial_number",) + SMART_ATTRIBUTES

    with open_csv_source(archive, ALL_DAYS_GLOB) as source:
        handles = {source.label(name): name for name in source.names}

        # The window for prediction date t ends on day t-1.
        for day in range(first_needed, max(wanted)):
            member = handles[calendar.label(day) + ".csv"]
            frame, _ = _read_csv_frame(
                lambda n=member: source.open(n), calendar.label(day), columns
            )
            frame = frame.dropna(subset=["serial_number"])
            serials = frame["serial_number"].astype(str).to_numpy()
            positions = metadata.index.get_indexer(serials)
            keep = positions >= 0
            values = frame[list(SMART_ATTRIBUTES)].to_numpy(dtype=np.float32)
            buffer.write_day(day, positions[keep], values[keep])
            del frame, serials, values
            gc.collect()

            # The window covers [day-29, day]: history for prediction day+1.
            prediction_day = day + 1
            if prediction_day not in wanted_set:
                continue

            block = _build_block(buffer, day, prediction_day, metadata, counts, quiet)
            if block is not None:
                yield block
            gc.collect()


def _build_block(
    buffer: WindowBuffer,
    history_end: int,
    prediction_day: int,
    metadata: DriveMetadata,
    counts: Optional[PanelCounts],
    quiet: bool,
) -> Optional[Block]:
    """Assemble one prediction date's eligible rows.

    Args:
        buffer: Warm ring buffer whose window ends at ``history_end``.
        history_end: Last day included in the feature window.
        prediction_day: The prediction date's day index.
        metadata: Drive index, presence bitmap and failure days.
        counts: Optional accumulator to update.
        quiet: Suppress the progress line.

    Returns:
        The block, or ``None`` if no row survived eligibility.
    """
    aggregates = buffer.aggregate(history_end)

    # From the presence bitmap: a drive can report a row with blank attributes.
    history_start = history_end - WINDOW_DAYS + 1
    observations = (
        metadata.presence[history_start : history_end + 1].sum(axis=0).astype(np.float32)
    )
    aggregates["observations_30d"] = observations

    failure_day = metadata.failure_day
    horizon_start = prediction_day + 1
    horizon_end = prediction_day + HORIZON_DAYS

    has_history = observations > 0
    already_failed = (failure_day >= 0) & (failure_day <= prediction_day)
    fails_in_horizon = (failure_day >= horizon_start) & (failure_day <= horizon_end)

    # A negative needs the drive observed on every day of the future window.
    future = metadata.presence[horizon_start : horizon_end + 1]
    observed_throughout = future.all(axis=0)

    eligible = has_history & ~already_failed & (fails_in_horizon | observed_throughout)

    if counts is not None:
        counts.prediction_dates += 1
        counts.excluded_no_history += int((~has_history).sum())
        counts.excluded_already_failed += int((has_history & already_failed).sum())
        counts.excluded_unverifiable += int(
            (has_history & ~already_failed & ~fails_in_horizon & ~observed_throughout).sum()
        )

    if not eligible.any():
        return None

    rows = np.flatnonzero(eligible)
    features = np.empty((rows.size, len(FEATURE_NAMES)), dtype=np.float32)
    for column, name in enumerate(FEATURE_NAMES):
        features[:, column] = aggregates[name][rows]
    labels = fails_in_horizon[rows].astype(np.int8)

    del aggregates
    gc.collect()

    if counts is not None:
        counts.rows_emitted += rows.size
        counts.positives += int(labels.sum())

    if not quiet:
        say(
            f"  {pd.Timestamp('2026-01-01') + pd.Timedelta(days=prediction_day):%Y-%m-%d}"
            f": {rows.size:,} rows | {int(labels.sum()):,} positive"
        )

    return Block(
        day=prediction_day,
        features=features,
        labels=labels,
        drives=rows.astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Return a Wilson score confidence interval for a proportion.

    Args:
        successes: Number of successes.
        trials: Number of trials.
        z: Normal quantile; 1.96 gives a 95% interval.

    Returns:
        ``(lower, upper)``, or ``(0.0, 0.0)`` when ``trials`` is zero.
    """
    if trials <= 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    spread = (
        z * np.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    ) / denominator
    return (float(max(0.0, centre - spread)), float(min(1.0, centre + spread)))


@dataclass
class FoldScores:
    """Scores and identifiers for one evaluated fold.

    Attributes:
        labels: Drive-day labels.
        drives: Drive positions, for event-level aggregation.
        days: Prediction-date day index per row.
        scores: Model probabilities keyed by model name.
        rule: The three-condition rule's boolean alert per row.
    """

    labels: np.ndarray
    drives: np.ndarray
    days: np.ndarray
    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    rule: Optional[np.ndarray] = None

    @property
    def n_rows(self) -> int:
        """Return the number of drive-day rows."""
        return int(self.labels.size)

    @property
    def n_events(self) -> int:
        """Return the number of distinct failure events represented."""
        return int(np.unique(self.drives[self.labels == 1]).size)

    @property
    def n_dates(self) -> int:
        """Return the number of distinct prediction dates."""
        return int(np.unique(self.days).size)


def evaluate(
    fold: FoldScores, alerts: np.ndarray, scores: Optional[np.ndarray], name: str
) -> Dict[str, object]:
    """Compute drive-day and event-level metrics for one alert vector.

    Args:
        fold: The fold being scored.
        alerts: Boolean alert per drive-day row.
        scores: Probabilities for PR-AUC, or ``None`` for the rule baseline.
        name: Model or baseline name.

    Returns:
        Metric name to value.
    """
    from sklearn.metrics import average_precision_score

    labels = fold.labels
    positives = labels == 1
    true_positive = int((alerts & positives).sum())
    false_positive = int((alerts & ~positives).sum())
    false_negative = int((~alerts & positives).sum())
    true_negative = int((~alerts & ~positives).sum())

    healthy = false_positive + true_negative
    flagged = true_positive + false_positive
    actual = true_positive + false_negative

    # A failure is detected if any alert fired in its pre-failure window.
    failing_drives = fold.drives[positives]
    alerted_positive_drives = fold.drives[positives & alerts]
    events_total = int(np.unique(failing_drives).size)
    events_detected = int(np.unique(alerted_positive_drives).size)

    recall_lo, recall_hi = wilson_interval(events_detected, events_total)
    drive_days = labels.size

    return {
        "model": name,
        "pr_auc": (
            float(average_precision_score(labels, scores)) if scores is not None else None
        ),
        "recall_drive_day": float(true_positive / actual) if actual else 0.0,
        "precision_drive_day": float(true_positive / flagged) if flagged else 0.0,
        "false_alert_rate": float(false_positive / healthy) if healthy else 0.0,
        "false_alerts_per_1000_healthy": (
            float(false_positive / healthy * 1000.0) if healthy else 0.0
        ),
        "event_recall": float(events_detected / events_total) if events_total else 0.0,
        "event_recall_ci_low": recall_lo,
        "event_recall_ci_high": recall_hi,
        "events_detected": events_detected,
        "events_total": events_total,
        "alerts_per_1000_drives_per_day": float(flagged / drive_days * 1000.0)
        if drive_days
        else 0.0,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "true_negatives": true_negative,
        "false_negatives": false_negative,
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, bool]:
    """Pick the F1-maximising threshold, refusing degenerate endpoints.

    An unrestricted F1-argmax can land on the last point of the
    precision-recall curve, giving a threshold of 1.0 that never fires.
    Candidates are restricted to thresholds below 1.0 that catch something.

    Args:
        labels: Ground-truth labels.
        scores: Predicted probabilities.

    Returns:
        ``(threshold, f1, was_degenerate)`` where ``was_degenerate`` records
        whether the unrestricted argmax had to be rejected.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.where(total > 0, 2 * precision * recall / np.where(total > 0, total, 1.0), 0.0)

    if f1.size == 0:
        return 0.5, 0.0, True

    unrestricted = int(np.argmax(f1))
    usable = (thresholds < 1.0) & (recall > 0.0) & (precision > 0.0)
    if not usable.any():
        return float(np.median(scores)), 0.0, True

    candidates = np.where(usable, f1, -1.0)
    best = int(np.argmax(candidates))
    return float(thresholds[best]), float(f1[best]), best != unrestricted


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_models(scale_pos_weight: float) -> Dict[str, object]:
    """Construct the three estimator pipelines.

    Preprocessing sits inside each pipeline, fitted on training rows only.

    Args:
        scale_pos_weight: Negatives per positive in the sampled training fold.

    Returns:
        Unfitted pipelines keyed by name.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    return {
        "logistic_regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        min_samples_leaf=50,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="aucpr",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def split_days(days: Sequence[int]) -> Tuple[List[int], List[int], List[int]]:
    """Split prediction dates chronologically into train, validation and test.

    Args:
        days: Eligible prediction-date day indices, ascending.

    Returns:
        ``(train_days, validation_days, test_days)``.
    """
    days = sorted(days)
    n_train = int(len(days) * TRAIN_FRACTION)
    n_validation = int(len(days) * VALIDATION_FRACTION)
    return (
        days[:n_train],
        days[n_train : n_train + n_validation],
        days[n_train + n_validation :],
    )


def collect_training_matrix(
    archive: Path,
    calendar: Calendar,
    metadata: DriveMetadata,
    train_days: Sequence[int],
    counts: PanelCounts,
    sample_rate: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Stream the training dates, keeping all positives and sampled negatives.

    Args:
        archive: Archive or directory of daily files.
        calendar: The archive's calendar.
        metadata: Drive metadata.
        train_days: Training prediction dates.
        counts: Accumulator for full (pre-sampling) row counts.
        sample_rate: Share of negative rows retained.

    Returns:
        ``(X, y, negatives_available, negatives_kept)``.
    """
    generator = np.random.default_rng(RANDOM_STATE)
    feature_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    negatives_available = 0
    negatives_kept = 0

    for block in stream_blocks(
        archive, calendar, metadata, train_days, counts=counts, quiet=True
    ):
        positive = block.labels == 1
        negative = ~positive
        negatives_available += int(negative.sum())

        chosen = positive.copy()
        candidates = np.flatnonzero(negative)
        if candidates.size:
            picked = candidates[
                generator.random(candidates.size) < sample_rate
            ]
            chosen[picked] = True
            negatives_kept += int(picked.size)

        feature_chunks.append(block.features[chosen])
        label_chunks.append(block.labels[chosen])
        del block
        gc.collect()

    X = np.concatenate(feature_chunks, axis=0)
    y = np.concatenate(label_chunks, axis=0)
    del feature_chunks, label_chunks
    gc.collect()
    return X, y, negatives_available, negatives_kept


def score_fold(
    archive: Path,
    calendar: Calendar,
    metadata: DriveMetadata,
    days: Sequence[int],
    fitted: Dict[str, object],
    counts: Optional[PanelCounts],
    label: str,
) -> FoldScores:
    """Score a fold date by date, retaining only predictions.

    Args:
        archive: Archive or directory of daily files.
        calendar: The archive's calendar.
        metadata: Drive metadata.
        days: Prediction dates in this fold.
        fitted: Fitted pipelines keyed by name.
        counts: Optional accumulator for row counts.
        label: Fold name, for progress output.

    Returns:
        The fold's :class:`FoldScores`, covering every eligible row.
    """
    rule_columns = [FEATURE_NAMES.index(name) for name in RULE_FEATURES]
    labels: List[np.ndarray] = []
    drives: List[np.ndarray] = []
    day_ids: List[np.ndarray] = []
    scores: Dict[str, List[np.ndarray]] = {name: [] for name in fitted}
    rules: List[np.ndarray] = []

    for block in stream_blocks(
        archive, calendar, metadata, days, counts=counts, quiet=True
    ):
        labels.append(block.labels)
        drives.append(block.drives)
        day_ids.append(np.full(block.labels.size, block.day, dtype=np.int16))

        rule_alert = np.zeros(block.labels.size, dtype=bool)
        for column in rule_columns:
            values = block.features[:, column]
            rule_alert |= np.nan_to_num(values, nan=0.0) > 0
        rules.append(rule_alert)

        for name, pipeline in fitted.items():
            scores[name].append(
                pipeline.predict_proba(block.features)[:, 1].astype(np.float32)
            )

        del block, rule_alert
        gc.collect()

    fold = FoldScores(
        labels=np.concatenate(labels),
        drives=np.concatenate(drives),
        days=np.concatenate(day_ids),
        scores={name: np.concatenate(chunks) for name, chunks in scores.items()},
        rule=np.concatenate(rules),
    )
    say(
        f"  {label}: {fold.n_rows:,} rows | {int(fold.labels.sum()):,} positive "
        f"| {fold.n_events:,} events | {fold.n_dates} dates"
    )
    return fold


def run(archive: Path, artifacts_dir: Path, sample_rate: float) -> int:
    """Run the panel experiment end to end.

    Args:
        archive: Archive or directory of daily files.
        artifacts_dir: Destination for metrics, metadata and models.
        sample_rate: Negative sampling rate for training.

    Returns:
        Process exit code.
    """
    import joblib
    import sklearn
    import xgboost

    calendar = scan_calendar(archive)
    say(
        f"Calendar: {len(calendar)} contiguous days, "
        f"{calendar.label(0)} .. {calendar.label(len(calendar) - 1)}"
    )

    metadata = scan_drives(archive, calendar)
    total_failures = int((metadata.failure_day >= 0).sum())
    say(
        f"Drives: {metadata.n_drives:,} unique | {metadata.total_rows:,} drive-day rows "
        f"| {total_failures:,} failure events in the quarter"
    )

    days = eligible_prediction_days(calendar)
    train_days, validation_days, test_days = split_days(days)
    say(
        f"Eligible prediction dates: {len(days)} "
        f"({calendar.label(days[0])} .. {calendar.label(days[-1])})"
    )
    say(
        f"  train      : {len(train_days)} dates "
        f"({calendar.label(train_days[0])} .. {calendar.label(train_days[-1])})"
    )
    say(
        f"  validation : {len(validation_days)} dates "
        f"({calendar.label(validation_days[0])} .. {calendar.label(validation_days[-1])})"
    )
    say(
        f"  test       : {len(test_days)} dates "
        f"({calendar.label(test_days[0])} .. {calendar.label(test_days[-1])})"
    )

    print()
    say(f"Building training matrix (all positives + {sample_rate:.0%} of negatives)")
    train_counts = PanelCounts()
    X_train, y_train, negatives_available, negatives_kept = collect_training_matrix(
        archive, calendar, metadata, train_days, train_counts, sample_rate
    )
    positives_train = int(y_train.sum())
    say(
        f"  full training panel   : {train_counts.rows_emitted:,} rows "
        f"| {train_counts.positives:,} positive"
    )
    say(
        f"  sampled for fitting   : {X_train.shape[0]:,} rows "
        f"({positives_train:,} positive + {negatives_kept:,} of "
        f"{negatives_available:,} negatives)"
    )
    if positives_train == 0:
        raise DataValidationError("Training fold has no positive rows.")
    scale_pos_weight = (X_train.shape[0] - positives_train) / positives_train
    say(f"  scale_pos_weight      : {scale_pos_weight:,.2f}")

    print()
    fitted: Dict[str, object] = {}
    fit_seconds: Dict[str, float] = {}
    for name, pipeline in build_models(scale_pos_weight).items():
        say(f"Fitting {name} on {X_train.shape[0]:,} x {X_train.shape[1]} ...")
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds[name] = time.perf_counter() - started
        fitted[name] = pipeline
        say(f"  fitted in {fit_seconds[name]:.1f}s")

    del X_train, y_train
    gc.collect()

    print()
    say("Scoring validation fold (complete, unsampled)")
    validation_counts = PanelCounts()
    validation = score_fold(
        archive, calendar, metadata, validation_days, fitted, validation_counts,
        "validation",
    )

    thresholds: Dict[str, float] = {}
    print()
    say("Threshold selection (validation only, F1-maximising, degeneracy-guarded)")
    for name in fitted:
        threshold, f1, degenerate = choose_threshold(
            validation.labels, validation.scores[name]
        )
        thresholds[name] = threshold
        note = " [rejected degenerate argmax]" if degenerate else ""
        say(f"  {name:<20} threshold={threshold:.6f} validation F1={f1:.4f}{note}")

    validation_rows = [
        evaluate(
            validation,
            validation.scores[name] >= thresholds[name],
            validation.scores[name],
            name,
        )
        for name in fitted
    ]
    validation_rows.append(
        evaluate(validation, validation.rule, None, "rule_baseline")
    )

    del validation
    gc.collect()

    print()
    say("Scoring test fold (complete, unsampled, untouched until now)")
    test_counts = PanelCounts()
    test = score_fold(
        archive, calendar, metadata, test_days, fitted, test_counts, "test"
    )

    test_rows = [
        evaluate(test, test.scores[name] >= thresholds[name], test.scores[name], name)
        for name in fitted
    ]
    test_rows.append(evaluate(test, test.rule, None, "rule_baseline"))

    base_rate = float(test.labels.mean())
    test_events = test.n_events
    test_rows_count = test.n_rows
    test_positives = int(test.labels.sum())
    test_drives = int(np.unique(test.drives).size)
    del test
    gc.collect()

    # ---------------------------------------------------------------- report
    metrics = pd.DataFrame(
        [{**row, "fold": "validation"} for row in validation_rows]
        + [{**row, "fold": "test"} for row in test_rows]
    )
    metrics["threshold"] = metrics["model"].map(thresholds)
    metrics["fit_seconds"] = metrics["model"].map(fit_seconds)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "model_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print()
    say("=" * 108)
    say("PANEL COUNTS")
    say("=" * 108)
    say(f"  drive-day rows in archive        : {metadata.total_rows:,}")
    say(f"  unique drives in archive         : {metadata.n_drives:,}")
    say(f"  failure events in quarter        : {total_failures:,}")
    say(f"  eligible prediction dates        : {len(days)}")
    for tag, counter in (
        ("train", train_counts),
        ("validation", validation_counts),
        ("test", test_counts),
    ):
        say(
            f"  {tag:<11} drive-day examples   : {counter.rows_emitted:,} "
            f"| positive {counter.positives:,}"
        )
    say(f"  test unique drives               : {test_drives:,}")
    say(f"  test unique failure events       : {test_events:,}")
    say(f"  test positive drive-day labels   : {test_positives:,}")
    say(f"  test base rate (no-skill PR-AUC) : {base_rate:.8f}")
    say("  rows excluded across all folds:")
    for tag, counter in (
        ("train", train_counts),
        ("validation", validation_counts),
        ("test", test_counts),
    ):
        say(
            f"    {tag:<10} no history {counter.excluded_no_history:,} | "
            f"already failed {counter.excluded_already_failed:,} | "
            f"unverifiable {counter.excluded_unverifiable:,}"
        )

    for fold_name in ("validation", "test"):
        rows = metrics[metrics["fold"] == fold_name]
        print()
        say("=" * 108)
        say(f"{fold_name.upper()} METRICS")
        say("=" * 108)
        table = rows[
            [
                "model",
                "pr_auc",
                "recall_drive_day",
                "precision_drive_day",
                "false_alert_rate",
                "false_alerts_per_1000_healthy",
                "alerts_per_1000_drives_per_day",
            ]
        ].copy()
        print(table.to_string(index=False))
        print()
        events = rows[
            [
                "model",
                "event_recall",
                "event_recall_ci_low",
                "event_recall_ci_high",
                "events_detected",
                "events_total",
                "true_positives",
                "false_positives",
                "true_negatives",
                "false_negatives",
            ]
        ].copy()
        print(events.to_string(index=False))

    # ------------------------------------------------------ model selection
    model_rows = [row for row in test_rows if row["model"] != "rule_baseline"]
    ranked = sorted(model_rows, key=lambda r: r["event_recall"], reverse=True)
    best, runner_up = ranked[0], ranked[1]
    enough_events = test_events >= MIN_EVENTS_FOR_SELECTION
    separated = best["event_recall_ci_low"] > runner_up["event_recall"]
    conclusive = bool(enough_events and separated)

    print()
    say("=" * 108)
    say("MODEL SELECTION")
    say("=" * 108)
    say(f"  test failure events              : {test_events} "
        f"(minimum for selection: {MIN_EVENTS_FOR_SELECTION})")
    say(f"  best by event recall             : {best['model']} "
        f"{best['event_recall']:.3f} "
        f"CI [{best['event_recall_ci_low']:.3f}, {best['event_recall_ci_high']:.3f}]")
    say(f"  runner-up                        : {runner_up['model']} "
        f"{runner_up['event_recall']:.3f}")
    say(f"  best CI low > runner-up point    : {separated}")

    saved: List[str] = []
    for name, pipeline in fitted.items():
        path = artifacts_dir / f"panel_model_{name}.joblib"
        joblib.dump(
            {
                "model_name": name,
                "pipeline": pipeline,
                "feature_names": FEATURE_NAMES,
                "threshold": thresholds[name],
                "window_days": WINDOW_DAYS,
                "horizon_days": HORIZON_DAYS,
                "trained_on_dates": [
                    calendar.label(train_days[0]),
                    calendar.label(train_days[-1]),
                ],
                "label_definition": "failure recorded in [t+1, t+7]",
                "training_negatives_downsampled_to": sample_rate,
                "selection_conclusive": conclusive,
            },
            path,
            compress=3,
        )
        saved.append(str(path))

    legacy = artifacts_dir / "drive_failure_model.joblib"
    if conclusive:
        joblib.dump(
            {
                "model_name": best["model"],
                "pipeline": fitted[best["model"]],
                "feature_names": FEATURE_NAMES,
                "threshold": thresholds[best["model"]],
                "window_days": WINDOW_DAYS,
                "horizon_days": HORIZON_DAYS,
                "label_definition": "failure recorded in [t+1, t+7]",
                "selection_conclusive": True,
            },
            legacy,
            compress=3,
        )
        say(f"  selection CONCLUSIVE -> saved {legacy}")
    else:
        if legacy.exists():
            legacy.unlink()
            say(f"  removed superseded artifact {legacy}")
        say("  selection INCONCLUSIVE -> all three models saved, no best chosen")

    metadata_payload = {
        "experiment": "drive_day_panel_30d_history_7d_horizon",
        "archive": str(archive),
        "archive_read_in_place": True,
        "calendar": {
            "days": len(calendar),
            "first": calendar.label(0),
            "last": calendar.label(len(calendar) - 1),
            "contiguous": True,
        },
        "windows": {
            "history_days": WINDOW_DAYS,
            "history_window": "[t-30, t-1] (excludes the prediction date)",
            "horizon_days": HORIZON_DAYS,
            "label_window": "[t+1, t+7]",
        },
        "counts": {
            "archive_drive_day_rows": metadata.total_rows,
            "archive_unique_drives": metadata.n_drives,
            "failure_events_in_quarter": total_failures,
            "eligible_prediction_dates": len(days),
            "train_drive_day_examples": train_counts.rows_emitted,
            "train_positive_labels": train_counts.positives,
            "validation_drive_day_examples": validation_counts.rows_emitted,
            "validation_positive_labels": validation_counts.positives,
            "test_drive_day_examples": test_rows_count,
            "test_positive_labels": test_positives,
            "test_unique_drives": test_drives,
            "test_unique_failure_events": test_events,
            "test_base_rate": base_rate,
        },
        "exclusions": {
            fold: {
                "no_history": counter.excluded_no_history,
                "already_failed": counter.excluded_already_failed,
                "unverifiable_future": counter.excluded_unverifiable,
            }
            for fold, counter in (
                ("train", train_counts),
                ("validation", validation_counts),
                ("test", test_counts),
            )
        },
        "splits": {
            "method": "chronological by prediction date",
            "train": [calendar.label(train_days[0]), calendar.label(train_days[-1])],
            "validation": [
                calendar.label(validation_days[0]),
                calendar.label(validation_days[-1]),
            ],
            "test": [calendar.label(test_days[0]), calendar.label(test_days[-1])],
        },
        "training_sampling": {
            "all_positives_kept": True,
            "negative_sample_rate": sample_rate,
            "negatives_available": negatives_available,
            "negatives_kept": negatives_kept,
            "scale_pos_weight": scale_pos_weight,
            "note": "Training fold only; validation and test are complete.",
        },
        "features": {
            "attributes": list(SMART_ATTRIBUTES),
            "statistics": list(STATISTICS),
            "names": FEATURE_NAMES,
            "count": len(FEATURE_NAMES),
        },
        "rule_baseline": " OR ".join(f"{name} > 0" for name in RULE_FEATURES),
        "thresholds": thresholds,
        "threshold_selection": "max F1 on validation fold, degenerate endpoints rejected",
        "model_selection": {
            "criterion": (
                f"test events >= {MIN_EVENTS_FOR_SELECTION} AND best event-recall "
                "Wilson CI lower bound > runner-up point estimate"
            ),
            "test_events": test_events,
            "enough_events": bool(enough_events),
            "separated": bool(separated),
            "conclusive": conclusive,
            "best_model": best["model"] if conclusive else None,
        },
        "artifacts": saved,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    metadata_path = artifacts_dir / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))

    print()
    say(f"Wrote {metrics_path} ({metrics_path.stat().st_size:,} bytes)")
    say(f"Wrote {metadata_path} ({metadata_path.stat().st_size:,} bytes)")
    for path in saved:
        say(f"Wrote {path} ({Path(path).stat().st_size:,} bytes)")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Drive-day panel experiment: 30-day history, 7-day horizon."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path.home() / "Downloads" / "data_Q1_2026.zip",
        help="ZIP archive (read in place) or directory of daily CSV files.",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="Destination for metrics, metadata and models.",
    )
    parser.add_argument(
        "--negative-sample-rate", type=float, default=NEGATIVE_SAMPLE_RATE,
        help="Share of training negatives retained (training fold only).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    configure_logging(args.verbose)
    logging.getLogger("src").setLevel(
        logging.DEBUG if args.verbose else logging.WARNING
    )

    if not 0.0 < args.negative_sample_rate <= 1.0:
        print("ERROR: --negative-sample-rate must be in (0, 1]", file=sys.stderr)
        return 1

    try:
        return run(args.archive, args.artifacts_dir, args.negative_sample_rate)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except DataValidationError as exc:
        print(f"ERROR: data validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
