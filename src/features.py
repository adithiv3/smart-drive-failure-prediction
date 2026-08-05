"""Trailing-window feature engineering for Backblaze SMART telemetry.

Summarises each drive's preceding 30 days into five statistics per SMART
attribute: ``_mean_30d``, ``_max_30d``, ``_std_30d``, ``_last_30d`` and
``_change_30d`` (last minus earliest reading in the window).

Windows are trailing and half-open — ``(date - 30 days, date]`` — so no
feature can see the future. The window is defined in time rather than rows,
because drives skip days and a fixed 30-row window would reach back months for
an intermittent reporter.

:func:`build_feature_matrix` rejects any column in
:data:`src.data_pipeline.LEAKAGE_COLUMNS`.

Usage:
    python -m src.features --input-dir data/raw \\
        --output data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_pipeline import (
    LABEL_COLUMN,
    LEAKAGE_COLUMNS,
    SMART_RAW_COLUMNS,
    DataValidationError,
    PipelineConfig,
    configure_logging,
    load_labelled_dataset,
    run_pipeline,
    write_labelled_dataset,
)

LOGGER = logging.getLogger(__name__)

#: Suffixes appended to each SMART column, in generation order.
FEATURE_SUFFIXES: Tuple[str, ...] = ("mean", "max", "std", "last", "change")

#: Non-SMART engineered features that do not depend on the window length.
#: The window-dependent ``observations_<N>d`` count is named by
#: :meth:`FeatureConfig.feature_names`.
STATIC_FEATURE_COLUMNS: Tuple[str, ...] = (
    "capacity_gb",
    "power_on_days",
    "history_days",
)

#: Row identifiers carried alongside the design matrix. Never features.
METADATA_COLUMNS: Tuple[str, ...] = ("date", "serial_number", "model")

#: Per-value cost of a float64 column, for the memory warning only.
_BYTES_PER_VALUE: int = 8

#: Warn and suggest a Polars/DuckDB backend above this estimated footprint.
_MEMORY_WARN_BYTES: int = 8 * 1024**3


@dataclass(frozen=True)
class FeatureConfig:
    """Settings for temporal feature construction.

    Attributes:
        window_days: Length of the trailing window in calendar days.
        smart_columns: SMART raw columns to summarise.
        min_periods: Minimum non-null readings before ``std`` is computed.
            ``mean`` and ``max`` are emitted from a single reading.
        min_history_days: Drop observations whose drive has fewer than this
            many days of prior history.
        drop_all_nan_features: Remove generated features that are NaN for every
            row (happens when a SMART attribute is absent from the whole date
            range for the drive models present).
    """

    window_days: int = 30
    smart_columns: Tuple[str, ...] = SMART_RAW_COLUMNS
    min_periods: int = 2
    min_history_days: int = 7
    drop_all_nan_features: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            DataValidationError: If any setting is out of range.
        """
        if self.window_days < 2:
            raise DataValidationError(
                f"window_days must be >= 2, got {self.window_days!r}"
            )
        if self.min_periods < 1:
            raise DataValidationError(
                f"min_periods must be >= 1, got {self.min_periods!r}"
            )
        if self.min_history_days < 0:
            raise DataValidationError(
                f"min_history_days must be >= 0, got {self.min_history_days!r}"
            )
        if self.min_history_days >= self.window_days:
            raise DataValidationError(
                f"min_history_days ({self.min_history_days}) must be less than "
                f"window_days ({self.window_days}); otherwise every row with a "
                "partially filled window is discarded."
            )
        if not self.smart_columns:
            raise DataValidationError("smart_columns must not be empty")

    @property
    def window(self) -> str:
        """Return the pandas offset alias for the trailing window, e.g. ``30D``."""
        return f"{self.window_days}D"

    def feature_names(self, smart_columns: Optional[Sequence[str]] = None) -> List[str]:
        """List every feature name this configuration generates.

        Args:
            smart_columns: Override the configured SMART columns, e.g. to
                reflect columns that actually survived loading.

        Returns:
            Feature names in deterministic generation order.
        """
        columns = list(smart_columns if smart_columns is not None else self.smart_columns)
        rolling = [
            f"{column}_{suffix}_{self.window_days}d"
            for column in columns
            for suffix in FEATURE_SUFFIXES
        ]
        return (
            rolling
            + [f"observations_{self.window_days}d"]
            + list(STATIC_FEATURE_COLUMNS)
        )


@dataclass
class FeatureMatrix:
    """A leakage-checked design matrix with its target and row metadata.

    Attributes:
        X: Numeric features, one row per observation.
        y: Binary target aligned to ``X`` by position and index.
        metadata: ``date``/``serial_number``/``model`` for each row.
        feature_names: Column order of ``X``.
        dropped_features: Features removed because they were entirely NaN.
    """

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    feature_names: List[str]
    dropped_features: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Assert the invariants downstream code depends on.

        Raises:
            DataValidationError: If shapes disagree, indexes are misaligned, or
                a leakage column made it into ``X``.
        """
        if not (len(self.X) == len(self.y) == len(self.metadata)):
            raise DataValidationError(
                f"Misaligned lengths: X={len(self.X)}, y={len(self.y)}, "
                f"metadata={len(self.metadata)}"
            )
        if not self.X.index.equals(self.y.index):
            raise DataValidationError("X and y indexes are not identical")
        leaked = sorted(set(self.X.columns) & LEAKAGE_COLUMNS)
        if leaked:
            raise DataValidationError(
                f"Outcome-derived column(s) {leaked} reached the design matrix. "
                "These encode the answer and must not be features."
            )

    @property
    def positive_count(self) -> int:
        """Return the number of positive labels."""
        return int(self.y.sum())

    @property
    def imbalance_ratio(self) -> float:
        """Return negatives per positive, for ``scale_pos_weight``."""
        positives = self.positive_count
        return (len(self.y) - positives) / positives if positives else float("inf")


@dataclass
class FeatureSummary:
    """Counts describing a generated feature table.

    ``failing_drives`` counts drives with a known failure date, not
    ``failure=1`` rows: the pipeline drops those, so counting them would always
    report zero.
    """

    records: int
    unique_drives: int
    failing_drives: int
    positive_labels: int
    n_features: int
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    rows_dropped_short_history: int = 0
    dropped_features: List[str] = field(default_factory=list)
    nan_rate_by_feature: Dict[str, float] = field(default_factory=dict)

    @property
    def positive_rate(self) -> float:
        """Return the share of rows carrying a positive label (0.0 if empty)."""
        return self.positive_labels / self.records if self.records else 0.0

    def as_lines(self, top_nan: int = 10) -> List[str]:
        """Render the summary as printable report lines.

        Args:
            top_nan: How many of the highest-missingness features to list.
        """
        span = (
            f"{self.date_min.date()} .. {self.date_max.date()}"
            if self.date_min is not None and self.date_max is not None
            else "n/a (empty dataset)"
        )
        negatives = self.records - self.positive_labels
        ratio = f"{negatives / self.positive_labels:,.1f}:1" if self.positive_labels else "n/a"
        lines = [
            "=" * 62,
            "FEATURE TABLE SUMMARY",
            "=" * 62,
            f"  date range                 : {span}",
            f"  records processed          : {self.records:,}",
            f"  unique drives              : {self.unique_drives:,}",
            f"  drives that fail (in-range): {self.failing_drives:,}",
            f"  positive labels            : {self.positive_labels:,}",
            f"  positive label rate        : {self.positive_rate:.6f}"
            f"  ({self.positive_rate * 100:.4f}%)",
            f"  negatives : positives      : {ratio}",
            f"  features generated         : {self.n_features:,}",
            "-" * 62,
            f"  rows dropped, < min history: {self.rows_dropped_short_history:,}",
            f"  features dropped, all NaN  : {len(self.dropped_features):,}",
        ]
        if self.dropped_features:
            lines.append(f"    {', '.join(self.dropped_features[:8])}"
                         + (" ..." if len(self.dropped_features) > 8 else ""))
        if self.nan_rate_by_feature:
            worst = sorted(
                self.nan_rate_by_feature.items(), key=lambda kv: kv[1], reverse=True
            )[:top_nan]
            if worst and worst[0][1] > 0:
                lines.append("-" * 62)
                lines.append("  highest missingness (feature: NaN share):")
                for name, rate in worst:
                    if rate > 0:
                        lines.append(f"    {name:<34} {rate:6.2%}")
        lines.append("=" * 62)
        return lines

    def print_report(self, stream=sys.stdout) -> None:
        """Print the summary to ``stream`` (stdout by default)."""
        print("\n".join(self.as_lines()), file=stream)


# ---------------------------------------------------------------------------
# Rolling-window construction
# ---------------------------------------------------------------------------


def _validate_input_frame(
    frame: pd.DataFrame, smart_columns: Sequence[str]
) -> List[str]:
    """Check the labelled frame and report which SMART columns are usable.

    Args:
        frame: Labelled telemetry from ``src.data_pipeline``.
        smart_columns: SMART columns requested by the configuration.

    Returns:
        The requested SMART columns that are present in ``frame``.

    Raises:
        DataValidationError: If required columns are absent, the frame is
            empty, dates are unparsed, or no requested SMART column exists.
    """
    if frame.empty:
        raise DataValidationError("Cannot build features from an empty frame")

    required = ("date", "serial_number", LABEL_COLUMN)
    absent = [column for column in required if column not in frame.columns]
    if absent:
        raise DataValidationError(
            f"Input frame is missing required column(s) {absent}. Build it with "
            "src.data_pipeline.run_pipeline() first."
        )
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        raise DataValidationError(
            f"`date` must be datetime64, got {frame['date'].dtype}"
        )

    available = [column for column in smart_columns if column in frame.columns]
    if not available:
        raise DataValidationError(
            f"None of the requested SMART columns {list(smart_columns)} are "
            "present in the input frame."
        )
    missing = [column for column in smart_columns if column not in frame.columns]
    if missing:
        LOGGER.warning(
            "%d requested SMART column(s) absent from the loaded data and "
            "skipped: %s",
            len(missing),
            missing,
        )

    duplicated = int(frame.duplicated(subset=["serial_number", "date"]).sum())
    if duplicated:
        raise DataValidationError(
            f"{duplicated:,} duplicate (serial_number, date) row(s) found. "
            "Trailing time windows assume one observation per drive per day; "
            "run the data pipeline's cleaning step first."
        )
    return available


def _warn_if_large(n_rows: int, n_features: int) -> None:
    """Log a warning when the dense feature table is likely to strain memory.

    Args:
        n_rows: Number of observations.
        n_features: Number of generated features.
    """
    estimate = n_rows * n_features * _BYTES_PER_VALUE
    if estimate > _MEMORY_WARN_BYTES:
        LOGGER.warning(
            "The feature table is projected at ~%.1f GiB (%d rows x %d float64 "
            "features). If this run thrashes or is killed, narrow --max-files / "
            "the date range, or move this step to Polars or DuckDB (both are "
            "already in requirements.txt).",
            estimate / 1024**3,
            n_rows,
            n_features,
        )


def _rolling_statistics(
    frame: pd.DataFrame, smart_columns: Sequence[str], config: FeatureConfig
) -> pd.DataFrame:
    """Compute trailing-window mean, max, std, last and observation count.

    The window is ``(date - window_days, date]`` per drive: trailing, and
    closed on the observation being described.

    Args:
        frame: Labelled telemetry sorted by ``(serial_number, date)``.
        smart_columns: SMART columns to summarise.
        config: Feature configuration.

    Returns:
        A frame indexed like ``frame`` holding the rolling statistics and the
        in-window row count ``observations_<window_days>d``.
    """
    columns = list(smart_columns)
    working = frame[["serial_number", "date"] + columns].copy()
    # A constant column counts rows in the window regardless of missingness.
    working["_row"] = 1.0

    indexed = working.set_index("date")
    rolling = indexed.groupby("serial_number", observed=True, sort=False)[
        columns + ["_row"]
    ].rolling(config.window, min_periods=1)

    means = rolling.mean()
    maxima = rolling.max()
    counts = rolling.count()
    # min_periods is applied by hand so mean/max survive a single reading while
    # std still requires `min_periods` of them.
    stds = rolling.std()

    # Realign onto the original row order; pair uniqueness is pre-validated.
    target = pd.MultiIndex.from_arrays(
        [frame["serial_number"], frame["date"]], names=["serial_number", "date"]
    )
    means = means.reindex(target)
    maxima = maxima.reindex(target)
    counts = counts.reindex(target)
    stds = stds.reindex(target)

    out = pd.DataFrame(index=frame.index)
    suffix = config.window_days

    for column in columns:
        observed = counts[column].to_numpy()
        enough_for_std = observed >= config.min_periods

        out[f"{column}_mean_{suffix}d"] = means[column].to_numpy()
        out[f"{column}_max_{suffix}d"] = maxima[column].to_numpy()
        out[f"{column}_std_{suffix}d"] = np.where(
            enough_for_std, stds[column].to_numpy(), np.nan
        )
        # Forward-fill within the drive, then blank where the window is empty
        # so a pre-window value cannot leak in as current.
        last = frame.groupby("serial_number", observed=True, sort=False)[column].ffill()
        out[f"{column}_last_{suffix}d"] = np.where(observed > 0, last.to_numpy(), np.nan)

    out[f"observations_{suffix}d"] = counts["_row"].to_numpy()
    return out


def _window_first_values(
    frame: pd.DataFrame, smart_columns: Sequence[str], config: FeatureConfig
) -> pd.DataFrame:
    """Find the earliest reading inside each row's trailing window.

    ``pandas`` rolling has no vectorised "first value in window" and an
    ``apply`` would be O(rows x window). An as-of join matches the first
    observation strictly after ``date - window_days``, the left edge of the
    half-open window used by :func:`_rolling_statistics`.

    Args:
        frame: Labelled telemetry sorted by ``(serial_number, date)``.
        smart_columns: SMART columns to look up.
        config: Feature configuration.

    Returns:
        A frame indexed like ``frame`` with one column per SMART attribute
        holding its earliest in-window reading.
    """
    columns = list(smart_columns)
    offset = pd.Timedelta(days=config.window_days)

    # Slice the join key out of `frame` rather than rebuilding it from a numpy
    # array: merge_asof requires the `by` key to carry an identical dtype on
    # both sides, and `.to_numpy()` on a StringDtype column silently downgrades
    # it to object.
    left = frame[["serial_number"]].copy()
    left["_row_id"] = np.arange(len(frame))
    left["window_start"] = frame["date"] - offset
    left = left.sort_values("window_start", kind="mergesort")

    right = (
        frame[["serial_number", "date"] + columns]
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )

    matched = pd.merge_asof(
        left,
        right,
        left_on="window_start",
        right_on="date",
        by="serial_number",
        direction="forward",
        allow_exact_matches=False,  # window is (date - window_days, date]
    )

    matched = matched.sort_values("_row_id", kind="mergesort")
    first = pd.DataFrame(index=frame.index)
    for column in columns:
        first[column] = matched[column].to_numpy()
    return first


def add_static_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive per-row features that need no window.

    ``capacity_gb`` rescales capacity, ``power_on_days`` converts SMART 9 into
    drive age, and ``history_days`` records available history.

    Args:
        frame: Labelled telemetry sorted by ``(serial_number, date)``.

    Returns:
        A frame indexed like ``frame`` with the static features. Columns whose
        source is absent are omitted.
    """
    static = pd.DataFrame(index=frame.index)

    if "capacity_bytes" in frame.columns:
        static["capacity_gb"] = frame["capacity_bytes"].astype("float64") / 1e9
    else:
        LOGGER.warning("capacity_bytes absent; skipping capacity_gb")

    if "smart_9_raw" in frame.columns:
        # SMART 9 is power-on hours for every vendor in the Backblaze fleet.
        static["power_on_days"] = frame["smart_9_raw"].astype("float64") / 24.0
    else:
        LOGGER.warning("smart_9_raw absent; skipping power_on_days")

    first_seen = frame.groupby("serial_number", observed=True, sort=False)[
        "date"
    ].transform("min")
    static["history_days"] = (frame["date"] - first_seen).dt.days.astype("float64")
    return static


def build_features(
    frame: pd.DataFrame, config: Optional[FeatureConfig] = None
) -> Tuple[pd.DataFrame, FeatureSummary]:
    """Attach trailing-window and static features to labelled telemetry.

    Args:
        frame: Labelled telemetry produced by ``src.data_pipeline``.
        config: Feature configuration; defaults to :class:`FeatureConfig`.

    Returns:
        A tuple of ``(featured_frame, summary)``. The frame keeps the metadata,
        label and audit columns of the input and gains one column per generated
        feature.

    Raises:
        DataValidationError: If the input frame fails validation or every row
            is filtered out by ``min_history_days``.
    """
    config = config or FeatureConfig()
    available = _validate_input_frame(frame, config.smart_columns)

    frame = frame.sort_values(["serial_number", "date"], kind="mergesort").reset_index(
        drop=True
    )
    _warn_if_large(len(frame), len(config.feature_names(available)))

    LOGGER.info(
        "Computing %d-day trailing features for %d observation(s) across %d "
        "drive(s) and %d SMART attribute(s)",
        config.window_days,
        len(frame),
        frame["serial_number"].nunique(),
        len(available),
    )

    rolling = _rolling_statistics(frame, available, config)
    first_in_window = _window_first_values(frame, available, config)
    static = add_static_features(frame)

    changes = pd.DataFrame(index=frame.index)
    for column in available:
        last = rolling[f"{column}_last_{config.window_days}d"]
        changes[f"{column}_change_{config.window_days}d"] = (
            last - first_in_window[column]
        )

    featured = pd.concat([frame, rolling, changes, static], axis=1)

    # Deterministic order keeps downstream artefacts comparable across runs.
    generated = [
        name
        for name in config.feature_names(available)
        if name in featured.columns
    ]

    rows_before = len(featured)
    if config.min_history_days > 0:
        keep = featured["history_days"] >= config.min_history_days
        featured = featured.loc[keep].reset_index(drop=True)
        dropped = rows_before - len(featured)
        LOGGER.info(
            "Dropped %d row(s) with fewer than %d day(s) of prior history",
            dropped,
            config.min_history_days,
        )
    else:
        dropped = 0

    if featured.empty:
        raise DataValidationError(
            f"No rows survived the min_history_days={config.min_history_days} "
            "filter. The loaded date range is shorter than the history each "
            "drive needs; load more consecutive daily files."
        )

    nan_rates = {
        name: float(featured[name].isna().mean()) for name in generated
    }
    dropped_features: List[str] = []
    if config.drop_all_nan_features:
        dropped_features = [name for name, rate in nan_rates.items() if rate >= 1.0]
        if dropped_features:
            LOGGER.info(
                "Dropping %d feature(s) that are NaN for every row: %s",
                len(dropped_features),
                dropped_features[:8],
            )
            featured = featured.drop(columns=dropped_features)
            generated = [n for n in generated if n not in dropped_features]

    summary = FeatureSummary(
        records=len(featured),
        unique_drives=int(featured["serial_number"].nunique()),
        failing_drives=(
            int(featured.loc[featured["failure_date"].notna(), "serial_number"].nunique())
            if "failure_date" in featured.columns
            else 0
        ),
        positive_labels=int(featured[LABEL_COLUMN].sum()),
        n_features=len(generated),
        date_min=featured["date"].min(),
        date_max=featured["date"].max(),
        rows_dropped_short_history=dropped,
        dropped_features=dropped_features,
        nan_rate_by_feature={n: nan_rates[n] for n in generated},
    )
    return featured, summary


def build_feature_matrix(
    featured: pd.DataFrame, config: Optional[FeatureConfig] = None
) -> FeatureMatrix:
    """Split a featured frame into a leakage-checked ``(X, y, metadata)`` triple.

    Args:
        featured: Output of :func:`build_features`.
        config: Feature configuration used to generate ``featured``; defaults
            to :class:`FeatureConfig`.

    Returns:
        A :class:`FeatureMatrix`. Its ``__post_init__`` raises if any
        outcome-derived column reached ``X``.

    Raises:
        DataValidationError: If the label column is absent or no generated
            feature is present.
    """
    config = config or FeatureConfig()
    if LABEL_COLUMN not in featured.columns:
        raise DataValidationError(
            f"{LABEL_COLUMN!r} is absent; pass a frame produced by build_features()"
        )

    feature_names = [
        name for name in config.feature_names() if name in featured.columns
    ]
    if not feature_names:
        raise DataValidationError(
            "No generated feature columns found. Did you pass the raw labelled "
            "frame instead of the output of build_features()?"
        )

    X = featured[feature_names].astype("float64")
    y = featured[LABEL_COLUMN].astype("int8")
    metadata = featured[
        [column for column in METADATA_COLUMNS if column in featured.columns]
    ].copy()

    matrix = FeatureMatrix(
        X=X, y=y, metadata=metadata, feature_names=feature_names
    )
    LOGGER.info(
        "Design matrix: %d row(s) x %d feature(s); %d positive(s); "
        "negatives:positives = %.1f:1",
        len(X),
        len(feature_names),
        matrix.positive_count,
        matrix.imbalance_ratio,
    )
    return matrix


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build trailing-window SMART features from Backblaze telemetry. "
            "Reads either raw daily CSVs (--input-dir) or a labelled table "
            "already written by src.data_pipeline (--labelled)."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory of daily Backblaze CSV files (default: data/raw).",
    )
    source.add_argument(
        "--labelled",
        type=Path,
        default=None,
        help="Path to a labelled .parquet/.csv from src.data_pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for the feature table (.parquet or .csv).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Trailing window length in calendar days (default: 30).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="Label horizon in calendar days (default: 7). Ignored with --labelled.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        default=7,
        help="Drop rows with less prior history than this (default: 7).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Read only the first N daily files (useful for a smoke test).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Build features from the command line and print the resulting counts.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` on a handled data error,
        ``2`` on a missing input path.
    """
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    configure_logging(args.verbose)

    try:
        feature_config = FeatureConfig(
            window_days=args.window_days, min_history_days=args.min_history_days
        )

        if args.labelled is not None:
            LOGGER.info("Loading labelled dataset from %s", args.labelled)
            labelled = load_labelled_dataset(args.labelled)
        else:
            pipeline_config = PipelineConfig(
                input_dir=args.input_dir,
                horizon_days=args.horizon_days,
                max_files=args.max_files,
            )
            labelled, data_summary = run_pipeline(pipeline_config)
            data_summary.print_report()

        featured, summary = build_features(labelled, feature_config)
        matrix = build_feature_matrix(featured, feature_config)

        if args.output is not None:
            write_labelled_dataset(featured, args.output)
    except (FileNotFoundError, NotADirectoryError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except DataValidationError as exc:
        LOGGER.error("Feature construction failed: %s", exc)
        return 1

    summary.print_report()
    print(
        f"\nDesign matrix ready: X={matrix.X.shape}, positives={matrix.positive_count:,}, "
        f"negatives:positives={matrix.imbalance_ratio:,.1f}:1"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
