"""Cross-sectional experiment: January SMART history, February 1-7 outcome.

One row per ``serial_number``, features aggregated over January 1-31, label
taken from the February 1-7 outcome window.

A random stratified split is used here rather than a chronological one. After
aggregation each drive appears once, all features come from January and all
labels from February, so no row can sit on both sides of the split. The split
generalises across drives; it says nothing about generalisation across time,
since one 7-day outcome window is a single week of failure behaviour.

The January window is ~10.5M drive-days, so statistics are accumulated in one
streaming pass: each daily file is read, folded into fixed-size per-drive
accumulators, then released. Peak memory scales with drive count, not row count.

Usage:
    python -m src.experiment --archive ~/Downloads/data_Q1_2026.zip
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.data_pipeline import (
    DataValidationError,
    _read_csv_frame,
    configure_logging,
    open_csv_source,
)

LOGGER = logging.getLogger(__name__)

#: SMART raw attributes aggregated for this experiment.
SMART_ATTRIBUTES: Tuple[str, ...] = (
    "smart_5_raw",  # Reallocated sectors count
    "smart_9_raw",  # Power-on hours
    "smart_187_raw",  # Reported uncorrectable errors
    "smart_188_raw",  # Command timeout
    "smart_197_raw",  # Current pending sector count
    "smart_198_raw",  # Offline uncorrectable sector count
    "smart_199_raw",  # UltraDMA CRC error count
)

#: Statistics produced per attribute, in feature-name order.
STATISTICS: Tuple[str, ...] = (
    "latest",
    "mean",
    "min",
    "max",
    "std",
    "change",
    "count",
)

FEATURE_GLOB_HISTORY = "2026-01-*.csv"
FEATURE_GLOB_OUTCOME = "2026-02-0[1-7].csv"

RANDOM_STATE = 42
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2


def _stamp() -> str:
    """Return a short elapsed-time stamp for progress lines."""
    return f"{time.perf_counter() - _START:7.1f}s"


_START = time.perf_counter()


def say(message: str) -> None:
    """Print a progress line immediately.

    Args:
        message: Text to print.
    """
    print(f"[{_stamp()}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Streaming per-drive aggregation
# ---------------------------------------------------------------------------


class DriveAggregator:
    """Accumulates per-drive SMART statistics one daily file at a time.

    Holds seven accumulators per attribute (count, sum, sum of squares, min,
    max, first, last). Memory scales with drive count, not drive-days.

    Files must be fed in chronological order: ``first``, ``last`` and
    ``change`` depend on arrival order.
    """

    def __init__(self, columns: Sequence[str]) -> None:
        """Create an empty aggregator.

        Args:
            columns: SMART attribute names to track.
        """
        self.columns: List[str] = list(columns)
        width = len(self.columns)
        self.index: pd.Index = pd.Index([], dtype=object)
        self.count = np.zeros((0, width), dtype=np.int32)
        self.total = np.zeros((0, width), dtype=np.float64)
        self.total_sq = np.zeros((0, width), dtype=np.float64)
        self.minimum = np.full((0, width), np.nan, dtype=np.float64)
        self.maximum = np.full((0, width), np.nan, dtype=np.float64)
        self.first = np.full((0, width), np.nan, dtype=np.float64)
        self.last = np.full((0, width), np.nan, dtype=np.float64)
        self.days_observed = np.zeros(0, dtype=np.int32)

    def _grow(self, new_index: pd.Index) -> None:
        """Expand the accumulators to cover ``new_index``.

        Args:
            new_index: Superset of the current drive index.
        """
        if self.index.equals(new_index):
            return

        positions = new_index.get_indexer(self.index)
        size = len(new_index)
        width = len(self.columns)
        had_rows = len(self.index) > 0

        def regrow(old: np.ndarray, fill: float, dtype: type) -> np.ndarray:
            shape = (size, width) if old.ndim == 2 else (size,)
            fresh = np.full(shape, fill, dtype=dtype)
            if had_rows:
                fresh[positions] = old
            return fresh

        self.count = regrow(self.count, 0, np.int32)
        self.total = regrow(self.total, 0.0, np.float64)
        self.total_sq = regrow(self.total_sq, 0.0, np.float64)
        self.minimum = regrow(self.minimum, np.nan, np.float64)
        self.maximum = regrow(self.maximum, np.nan, np.float64)
        self.first = regrow(self.first, np.nan, np.float64)
        self.last = regrow(self.last, np.nan, np.float64)
        self.days_observed = regrow(self.days_observed, 0, np.int32)
        self.index = new_index

    def update(self, frame: pd.DataFrame) -> None:
        """Fold one day's readings into the accumulators.

        Args:
            frame: One day of telemetry indexed by ``serial_number``, with one
                row per drive and a column per tracked attribute.

        Raises:
            DataValidationError: If the frame's index is not unique, which
                would make the in-place accumulator updates double-count.
        """
        if not frame.index.is_unique:
            raise DataValidationError(
                "A daily frame contains duplicate serial_number values; "
                "accumulator updates assume one row per drive per day."
            )

        self._grow(self.index.union(frame.index))
        rows = self.index.get_indexer(frame.index)
        values = frame[self.columns].to_numpy(dtype=np.float64, copy=False)
        present = ~np.isnan(values)
        filled = np.nan_to_num(values, nan=0.0)

        self.count[rows] += present
        self.total[rows] += filled
        self.total_sq[rows] += filled * filled
        # fmin/fmax propagate the non-NaN operand.
        self.minimum[rows] = np.fmin(self.minimum[rows], values)
        self.maximum[rows] = np.fmax(self.maximum[rows], values)

        earliest = self.first[rows]
        unseen = np.isnan(earliest)
        earliest[unseen] = values[unseen]
        self.first[rows] = earliest

        latest = self.last[rows]
        latest[present] = values[present]
        self.last[rows] = latest

        self.days_observed[rows] += 1

    def to_frame(self) -> pd.DataFrame:
        """Derive the final per-drive feature table.

        Returns:
            One row per drive, indexed by ``serial_number``, with
            ``<attribute>_<statistic>`` columns plus ``days_observed``.
        """
        counts = self.count.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            safe_n = np.where(counts > 0, counts, np.nan)
            mean = self.total / safe_n
            # Clamped at zero: cancellation can make this slightly negative.
            safe_dof = np.where(counts > 1, counts - 1.0, np.nan)
            variance = np.maximum((self.total_sq - counts * mean * mean) / safe_dof, 0.0)
            std = np.sqrt(variance)

        change = self.last - self.first

        blocks: Dict[str, np.ndarray] = {}
        for position, attribute in enumerate(self.columns):
            blocks[f"{attribute}_latest"] = self.last[:, position]
            blocks[f"{attribute}_mean"] = mean[:, position]
            blocks[f"{attribute}_min"] = self.minimum[:, position]
            blocks[f"{attribute}_max"] = self.maximum[:, position]
            blocks[f"{attribute}_std"] = std[:, position]
            blocks[f"{attribute}_change"] = change[:, position]
            blocks[f"{attribute}_count"] = self.count[:, position].astype(np.float64)

        blocks["days_observed"] = self.days_observed.astype(np.float64)
        return pd.DataFrame(blocks, index=self.index.rename("serial_number"))


def feature_names(attributes: Sequence[str] = SMART_ATTRIBUTES) -> List[str]:
    """List the generated feature names in deterministic order.

    Args:
        attributes: SMART attributes aggregated.

    Returns:
        Feature names, ending with ``days_observed``.
    """
    names = [f"{a}_{s}" for a in attributes for s in STATISTICS]
    return names + ["days_observed"]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """The modelling table and its counts."""

    X: pd.DataFrame
    y: pd.Series
    drives_in_history: int
    rows_in_history: int
    failed_during_history: int
    failed_during_outcome: int
    positives: int
    observed_in_outcome: int
    absent_from_outcome: int
    outcome_only_drives: int
    history_files: List[str] = field(default_factory=list)
    outcome_files: List[str] = field(default_factory=list)

    @property
    def negatives(self) -> int:
        """Return the number of negative examples."""
        return len(self.y) - self.positives

    @property
    def positive_rate(self) -> float:
        """Return the share of examples that are positive."""
        return self.positives / len(self.y) if len(self.y) else 0.0


def aggregate_history(
    archive: Path, file_glob: str = FEATURE_GLOB_HISTORY
) -> Tuple[pd.DataFrame, Set[str], int, List[str]]:
    """Stream the history window and aggregate it to one row per drive.

    Args:
        archive: Archive or directory holding the daily files.
        file_glob: Pattern selecting the history window.

    Returns:
        ``(features, failed_during_history, rows_read, filenames)``.

    Raises:
        DataValidationError: If no file in the window could be parsed.
    """
    wanted = ("serial_number", "failure") + SMART_ATTRIBUTES
    aggregator = DriveAggregator(SMART_ATTRIBUTES)
    failed_during_history: Set[str] = set()
    rows_read = 0
    filenames: List[str] = []

    with open_csv_source(archive, file_glob) as source:
        say(f"History window: {source.describe()}")
        total = len(source)

        for index, name in enumerate(source.names, start=1):
            label = source.label(name)
            frame, missing = _read_csv_frame(
                lambda n=name: source.open(n), label, wanted
            )
            if missing:
                say(f"  WARNING {label}: column(s) absent, filled with NaN: {missing}")

            rows_read += len(frame)
            filenames.append(label)

            frame = frame.dropna(subset=["serial_number"])
            duplicates = int(frame.duplicated(subset=["serial_number"]).sum())
            if duplicates:
                say(f"  WARNING {label}: {duplicates:,} duplicate serial(s), keeping last")
                frame = frame.drop_duplicates(subset=["serial_number"], keep="last")

            failed_during_history.update(
                frame.loc[frame["failure"] == 1, "serial_number"].astype(str).tolist()
            )

            indexed = frame.set_index(frame["serial_number"].astype(str))
            aggregator.update(indexed[list(SMART_ATTRIBUTES)])

            say(
                f"  [{index:02d}/{total}] {label}: {len(frame):,} rows | "
                f"{len(aggregator.index):,} drives accumulated"
            )

            del frame, indexed
            gc.collect()

    features = aggregator.to_frame()
    del aggregator
    gc.collect()
    return features, failed_during_history, rows_read, filenames


def load_outcomes(
    archive: Path, file_glob: str = FEATURE_GLOB_OUTCOME
) -> Tuple[Set[str], Set[str], List[str]]:
    """Read the outcome window and collect who failed and who was seen.

    Args:
        archive: Archive or directory holding the daily files.
        file_glob: Pattern selecting the outcome window.

    Returns:
        ``(failed, observed, filenames)``.
    """
    failed: Set[str] = set()
    observed: Set[str] = set()
    filenames: List[str] = []

    with open_csv_source(archive, file_glob) as source:
        say(f"Outcome window: {source.describe()}")
        total = len(source)

        for index, name in enumerate(source.names, start=1):
            label = source.label(name)
            frame, _ = _read_csv_frame(
                lambda n=name: source.open(n), label, ("serial_number", "failure")
            )
            frame = frame.dropna(subset=["serial_number"])
            serials = frame["serial_number"].astype(str)
            observed.update(serials.tolist())
            day_failures = set(serials[frame["failure"] == 1].tolist())
            failed.update(day_failures)
            filenames.append(label)

            say(
                f"  [{index}/{total}] {label}: {len(frame):,} rows | "
                f"{len(day_failures):,} failure(s) | {len(failed):,} cumulative"
            )
            del frame, serials
            gc.collect()

    return failed, observed, filenames


def build_dataset(archive: Path) -> Dataset:
    """Assemble the per-drive design matrix and label vector.

    Args:
        archive: Archive or directory holding the daily files.

    Returns:
        The assembled :class:`Dataset`.

    Raises:
        DataValidationError: If the assembled dataset has no positive labels,
            which would make every metric undefined.
    """
    features, failed_in_history, rows_read, history_files = aggregate_history(archive)
    drives_in_history = len(features)
    say(
        f"Aggregated {rows_read:,} drive-day rows into {drives_in_history:,} "
        f"unique drives x {features.shape[1]} features"
    )

    failed_in_outcome, observed_in_outcome, outcome_files = load_outcomes(archive)

    # A drive that already failed is not a valid forecasting example.
    already_failed = [s for s in features.index if s in failed_in_history]
    if already_failed:
        features = features.drop(index=already_failed)
        say(
            f"Excluded {len(already_failed):,} drive(s) that failed during the "
            "history window (not valid forecasting examples)"
        )

    labels = pd.Series(
        [1 if serial in failed_in_outcome else 0 for serial in features.index],
        index=features.index,
        dtype="int8",
        name="failed_within_7d",
    )

    seen_next = features.index.isin(observed_in_outcome)
    absent = int((~seen_next).sum())
    outcome_only = len(observed_in_outcome - set(features.index) - failed_in_history)

    positives = int(labels.sum())
    if positives == 0:
        raise DataValidationError(
            "No positive labels in the outcome window; every rate metric would "
            "be undefined. Check that the outcome files were selected correctly."
        )

    dataset = Dataset(
        X=features,
        y=labels,
        drives_in_history=drives_in_history,
        rows_in_history=rows_read,
        failed_during_history=len(failed_in_history),
        failed_during_outcome=len(failed_in_outcome),
        positives=positives,
        observed_in_outcome=int(seen_next.sum()),
        absent_from_outcome=absent,
        outcome_only_drives=outcome_only,
        history_files=history_files,
        outcome_files=outcome_files,
    )
    return dataset


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_models(scale_pos_weight: float) -> Dict[str, object]:
    """Construct the three estimator pipelines.

    Median imputation sits inside each pipeline so it is fitted on the
    training fold only.

    Args:
        scale_pos_weight: Negatives per positive in the training fold.

    Returns:
        Unfitted pipelines keyed by model name.
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
                        max_iter=2000,
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
                        n_estimators=200,
                        min_samples_leaf=10,
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


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Pick the threshold maximising F1 on the given fold.

    Args:
        y_true: Ground-truth labels.
        scores: Predicted positive-class probabilities.

    Returns:
        ``(threshold, f1_at_threshold)``.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # One more precision/recall value than thresholds.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.where(denominator > 0, 2 * precision * recall / np.where(denominator > 0, denominator, 1), 0.0)
    if len(f1) == 0:
        return 0.5, 0.0
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def score_fold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Compute every reported metric at one threshold.

    Args:
        y_true: Ground-truth labels.
        scores: Predicted positive-class probabilities.
        threshold: Probability at or above which a drive is flagged.

    Returns:
        Metric name to value.
    """
    from sklearn.metrics import average_precision_score, confusion_matrix

    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    healthy = fp + tn
    flagged = tp + fp
    actual_positives = tp + fn

    false_alert_rate = fp / healthy if healthy else 0.0
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "base_rate": float(actual_positives / len(y_true)) if len(y_true) else 0.0,
        "threshold": float(threshold),
        "precision": float(tp / flagged) if flagged else 0.0,
        "recall": float(tp / actual_positives) if actual_positives else 0.0,
        "false_alert_rate": float(false_alert_rate),
        "false_alerts_per_1000_healthy": float(false_alert_rate * 1000.0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(archive: Path, artifacts_dir: Path) -> int:
    """Run the whole experiment and write the artifacts.

    Args:
        archive: Archive or directory holding the daily files.
        artifacts_dir: Destination for metrics, metadata and the model.

    Returns:
        Process exit code.
    """
    import joblib
    import sklearn
    import xgboost
    from sklearn.model_selection import train_test_split

    dataset = build_dataset(archive)
    names = [n for n in feature_names() if n in dataset.X.columns]
    X = dataset.X[names]
    y = dataset.y

    print()
    say("DATASET")
    say(f"  drives in history window        : {dataset.drives_in_history:,}")
    say(f"  drive-day rows aggregated       : {dataset.rows_in_history:,}")
    say(f"  failed during history window    : {dataset.failed_during_history:,} (excluded)")
    say(f"  modelling examples              : {len(y):,}")
    say(f"  features                        : {len(names)}")
    say(f"  failures in outcome window      : {dataset.failed_during_outcome:,}")
    say(f"  positive labels                 : {dataset.positives:,}")
    say(f"  negative labels                 : {dataset.negatives:,}")
    say(f"  positive rate                   : {dataset.positive_rate:.6%}")
    say(f"  observed in outcome window      : {dataset.observed_in_outcome:,}")
    say(f"  absent from outcome window      : {dataset.absent_from_outcome:,} (labelled 0)")
    say(f"  outcome-only drives (no history): {dataset.outcome_only_drives:,} (not modelled)")

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=RANDOM_STATE
    )
    validation_share = VALIDATION_FRACTION / (1.0 - TEST_FRACTION)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_pool,
        y_pool,
        test_size=validation_share,
        stratify=y_pool,
        random_state=RANDOM_STATE,
    )
    del X_pool, y_pool, X
    gc.collect()

    print()
    say("SPLITS (stratified, random_state=42)")
    for tag, target in (
        ("train", y_train),
        ("validation", y_validation),
        ("test", y_test),
    ):
        say(
            f"  {tag:<11}: {len(target):,} rows | {int(target.sum()):,} positive "
            f"| {target.mean():.6%}"
        )

    positives_train = int(y_train.sum())
    scale_pos_weight = (len(y_train) - positives_train) / positives_train
    say(f"  scale_pos_weight (train only)   : {scale_pos_weight:,.2f}")

    models = build_models(scale_pos_weight)
    rows: List[Dict[str, object]] = []
    fitted: Dict[str, object] = {}

    for name, pipeline in models.items():
        print()
        say(f"TRAINING {name}")
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        say(f"  fitted in {fit_seconds:.1f}s")

        validation_scores = pipeline.predict_proba(X_validation)[:, 1]
        threshold, validation_f1 = choose_threshold(
            y_validation.to_numpy(), validation_scores
        )
        validation_metrics = score_fold(
            y_validation.to_numpy(), validation_scores, threshold
        )
        say(
            f"  validation: PR-AUC={validation_metrics['pr_auc']:.4f} "
            f"threshold={threshold:.4f} F1={validation_f1:.4f}"
        )

        test_scores = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = score_fold(y_test.to_numpy(), test_scores, threshold)
        say(
            f"  test      : PR-AUC={test_metrics['pr_auc']:.4f} "
            f"recall={test_metrics['recall']:.4f} "
            f"precision={test_metrics['precision']:.4f}"
        )

        rows.append(
            {
                "model": name,
                "fit_seconds": round(fit_seconds, 2),
                "validation_pr_auc": validation_metrics["pr_auc"],
                "validation_f1": validation_f1,
                "threshold": threshold,
                **{f"test_{k}": v for k, v in test_metrics.items() if k != "threshold"},
            }
        )
        fitted[name] = pipeline
        del validation_scores, test_scores
        gc.collect()

    metrics = pd.DataFrame(rows)
    # Selection uses validation PR-AUC only, never test.
    best_name = str(metrics.loc[metrics["validation_pr_auc"].idxmax(), "model"])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "model_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    metadata = {
        "experiment": "january_history_to_february_7day_failure",
        "archive": str(archive),
        "archive_read_in_place": True,
        "history_window": {
            "glob": FEATURE_GLOB_HISTORY,
            "files": len(dataset.history_files),
            "first": dataset.history_files[0] if dataset.history_files else None,
            "last": dataset.history_files[-1] if dataset.history_files else None,
            "drive_day_rows": dataset.rows_in_history,
        },
        "outcome_window": {
            "glob": FEATURE_GLOB_OUTCOME,
            "files": len(dataset.outcome_files),
            "first": dataset.outcome_files[0] if dataset.outcome_files else None,
            "last": dataset.outcome_files[-1] if dataset.outcome_files else None,
        },
        "counts": {
            "unique_drives_in_history": dataset.drives_in_history,
            "failed_during_history_excluded": dataset.failed_during_history,
            "modelling_examples": int(len(y)),
            "positive_labels": dataset.positives,
            "negative_labels": dataset.negatives,
            "positive_rate": dataset.positive_rate,
            "failures_in_outcome_window": dataset.failed_during_outcome,
            "observed_in_outcome_window": dataset.observed_in_outcome,
            "absent_from_outcome_window_labelled_negative": dataset.absent_from_outcome,
            "outcome_only_drives_not_modelled": dataset.outcome_only_drives,
        },
        "splits": {
            "method": "stratified random over drives",
            "random_state": RANDOM_STATE,
            "train_rows": int(len(y_train)),
            "train_positives": positives_train,
            "validation_rows": int(len(y_validation)),
            "validation_positives": int(y_validation.sum()),
            "test_rows": int(len(y_test)),
            "test_positives": int(y_test.sum()),
            "scale_pos_weight_train": scale_pos_weight,
        },
        "features": {
            "attributes": list(SMART_ATTRIBUTES),
            "statistics": list(STATISTICS),
            "names": names,
            "count": len(names),
        },
        "threshold_selection": "max F1 on validation fold only",
        "model_selection": "max PR-AUC on validation fold only",
        "selected_model": best_name,
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
    metadata_path.write_text(json.dumps(metadata, indent=2))

    model_path = artifacts_dir / "drive_failure_model.joblib"
    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": fitted[best_name],
            "feature_names": names,
            "threshold": float(
                metrics.loc[metrics["model"] == best_name, "threshold"].iloc[0]
            ),
            "trained_on": metadata["history_window"],
            "label_definition": "failure recorded in 2026-02-01..2026-02-07",
        },
        model_path,
        compress=3,
    )

    print()
    say("FINAL TEST-SET METRICS (untouched until now)")
    display = metrics[
        [
            "model",
            "test_pr_auc",
            "test_recall",
            "test_precision",
            "test_false_alert_rate",
            "test_false_alerts_per_1000_healthy",
            "test_true_positives",
            "test_false_positives",
            "test_true_negatives",
            "test_false_negatives",
        ]
    ].copy()
    print(display.to_string(index=False))

    base_rate = float(metrics["test_base_rate"].iloc[0])
    print()
    say(f"Test-set base rate (no-skill PR-AUC): {base_rate:.6f}")
    for row in rows:
        lift = row["test_pr_auc"] / base_rate if base_rate else float("nan")
        say(f"  {row['model']:<20} PR-AUC lift over baseline: {lift:,.1f}x")

    print()
    say(f"Selected by validation PR-AUC: {best_name}")
    say(f"Wrote {metrics_path} ({metrics_path.stat().st_size:,} bytes)")
    say(f"Wrote {metadata_path} ({metadata_path.stat().st_size:,} bytes)")
    say(f"Wrote {model_path} ({model_path.stat().st_size:,} bytes)")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path.home() / "Downloads" / "data_Q1_2026.zip",
        help="ZIP archive (read in place) or directory of daily CSV files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Destination for metrics, metadata and the fitted model.",
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

    try:
        return run(args.archive, args.artifacts_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except DataValidationError as exc:
        print(f"ERROR: data validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
