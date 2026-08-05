"""Ingestion and labelling pipeline for Backblaze daily SMART telemetry.

Reads a directory or ZIP of daily CSVs and produces a chronologically sorted,
labelled observation table:

    date | serial_number | model | capacity_bytes | <smart_*_raw ...> | label

A row is labelled positive when the drive's failure date falls in
``[date + 1, date + horizon_days]``.

Rows on or after a drive's failure date are dropped, so the label window is
strictly in the future. Negatives whose drive stops appearing within
``horizon_days`` are flagged ``label_censored`` and dropped by default; their
outcome is unknown rather than healthy. Outcome-derived columns are listed in
:data:`LEAKAGE_COLUMNS` and rejected by ``src/features.py``.

Usage:
    python -m src.data_pipeline --input-dir data/raw \\
        --output data/processed/labelled.parquet
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Columns that identify an observation. Every daily file must provide these.
REQUIRED_COLUMNS: Tuple[str, ...] = ("date", "serial_number", "failure")

#: Identity/metadata columns kept from each daily file.
IDENTITY_COLUMNS: Tuple[str, ...] = (
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
)

#: SMART raw attributes retained for modelling. Raw rather than
#: vendor-normalised, since normalisation is not comparable across makers.
SMART_RAW_COLUMNS: Tuple[str, ...] = (
    "smart_5_raw",  # Reallocated sectors count
    "smart_9_raw",  # Power-on hours
    "smart_187_raw",  # Reported uncorrectable errors
    "smart_188_raw",  # Command timeout
    "smart_190_raw",  # Airflow / temperature difference
    "smart_193_raw",  # Load-unload cycle count
    "smart_194_raw",  # Temperature (Celsius)
    "smart_197_raw",  # Current pending sector count
    "smart_198_raw",  # Offline uncorrectable sector count
    "smart_199_raw",  # UltraDMA CRC error count
    "smart_241_raw",  # Total LBAs written
    "smart_242_raw",  # Total LBAs read
)

#: Every column the loader tries to read from a daily file.
KEPT_COLUMNS: Tuple[str, ...] = IDENTITY_COLUMNS + SMART_RAW_COLUMNS

#: Name of the generated supervised target.
LABEL_COLUMN: str = "label_fail_7d"

#: Outcome-derived columns. ``src/features.py`` asserts against this set.
LEAKAGE_COLUMNS: frozenset = frozenset(
    {
        "failure",
        "failure_date",
        "days_to_failure",
        "observed_through",
        "label_censored",
        LABEL_COLUMN,
    }
)

#: SMART counters stay float64: LBAs written/read exceed ~1e12, beyond float32
#: exact range, and all columns need a NaN slot.
_COLUMN_DTYPES: Dict[str, str] = dict(
    {"serial_number": "string", "model": "string"},
    **{column: "float64" for column in ("capacity_bytes", "failure") + SMART_RAW_COLUMNS},
)

#: Backblaze writes ``-1`` for an unreported capacity.
_CAPACITY_SENTINEL: float = -1.0


class DataValidationError(ValueError):
    """Raised when input data violates an assumption the pipeline relies on."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Settings for a single pipeline run.

    Attributes:
        input_dir: Either a directory containing Backblaze daily CSV files, or
            a ``.zip`` archive containing them. Directories are searched
            recursively, since the official archives unzip into per-quarter
            subdirectories; archives are read in place and never extracted.
        horizon_days: Label horizon. ``label == 1`` when the drive's failure
            date falls in ``[date + 1, date + horizon_days]``.
        smart_columns: SMART raw columns to retain.
        drop_post_failure_rows: Drop observations on or after a drive's failure
            date.
        drop_censored_rows: Drop negatives whose outcome cannot be confirmed
            because the drive is not observed for a further ``horizon_days``.
        file_glob: Filename pattern for daily files. For an archive it is
            matched against each member's basename, so ``"2026-01-*.csv"``
            selects a single month.
        max_files: Optional cap on files read, for smoke tests. ``None`` reads
            everything found.
    """

    input_dir: Path
    horizon_days: int = 7
    smart_columns: Tuple[str, ...] = SMART_RAW_COLUMNS
    drop_post_failure_rows: bool = True
    drop_censored_rows: bool = True
    file_glob: str = "*.csv"
    max_files: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            DataValidationError: If the horizon is not positive, no SMART
                columns were requested, or ``max_files`` is not positive.
        """
        if self.horizon_days < 1:
            raise DataValidationError(
                f"horizon_days must be >= 1, got {self.horizon_days!r}"
            )
        if not self.smart_columns:
            raise DataValidationError("smart_columns must not be empty")
        if self.max_files is not None and self.max_files < 1:
            raise DataValidationError(
                f"max_files must be >= 1 when set, got {self.max_files!r}"
            )

    @property
    def kept_columns(self) -> Tuple[str, ...]:
        """Return identity columns plus the configured SMART columns."""
        return IDENTITY_COLUMNS + tuple(self.smart_columns)


@dataclass
class DatasetSummary:
    """Counts describing a labelled dataset."""

    files_read: int
    records: int
    unique_drives: int
    failure_events: int
    positive_labels: int
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    rows_dropped_duplicate: int = 0
    rows_dropped_post_failure: int = 0
    rows_dropped_censored: int = 0
    missing_columns: Dict[str, int] = field(default_factory=dict)

    @property
    def positive_rate(self) -> float:
        """Return the share of rows carrying a positive label (0.0 if empty)."""
        return self.positive_labels / self.records if self.records else 0.0

    def as_lines(self) -> List[str]:
        """Render the summary as printable report lines."""
        span = (
            f"{self.date_min.date()} .. {self.date_max.date()}"
            if self.date_min is not None and self.date_max is not None
            else "n/a (empty dataset)"
        )
        lines = [
            "=" * 62,
            "DATA PIPELINE SUMMARY",
            "=" * 62,
            f"  daily files read           : {self.files_read:,}",
            f"  date range                 : {span}",
            f"  records processed          : {self.records:,}",
            f"  unique drives              : {self.unique_drives:,}",
            f"  failure events (failure=1) : {self.failure_events:,}",
            f"  positive labels            : {self.positive_labels:,}",
            f"  positive label rate        : {self.positive_rate:.6f}"
            f"  ({self.positive_rate * 100:.4f}%)",
            "-" * 62,
            f"  rows dropped, duplicate    : {self.rows_dropped_duplicate:,}",
            f"  rows dropped, post-failure : {self.rows_dropped_post_failure:,}",
            f"  rows dropped, censored     : {self.rows_dropped_censored:,}",
        ]
        if self.missing_columns:
            lines.append("-" * 62)
            lines.append("  columns absent from some files (filled with NaN):")
            for column, count in sorted(self.missing_columns.items()):
                lines.append(f"    {column:<24} missing in {count:,} file(s)")
        lines.append("=" * 62)
        return lines

    def print_report(self, stream=sys.stdout) -> None:
        """Print the summary to ``stream`` (stdout by default)."""
        print("\n".join(self.as_lines()), file=stream)


# ---------------------------------------------------------------------------
# File discovery and loading
# ---------------------------------------------------------------------------


def discover_csv_files(
    input_dir: Path, file_glob: str = "*.csv", max_files: Optional[int] = None
) -> List[Path]:
    """Find Backblaze daily CSV files under ``input_dir``.

    The search is recursive and results are sorted by path, which for
    Backblaze's ``YYYY-MM-DD.csv`` naming is also chronological order.

    Args:
        input_dir: Directory to search.
        file_glob: Filename pattern, e.g. ``"*.csv"``.
        max_files: If set, keep only the first this-many files.

    Returns:
        Sorted list of file paths.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist or contains no
            matching files.
        NotADirectoryError: If ``input_dir`` exists but is not a directory.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}. Download the daily "
            "files from https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data "
            "and unzip them there (raw CSVs are intentionally git-ignored)."
        )
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    paths = sorted(p for p in input_dir.rglob(file_glob) if p.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No files matching {file_glob!r} found under {input_dir}."
        )

    if max_files is not None:
        paths = paths[:max_files]

    LOGGER.info("Discovered %d daily file(s) under %s", len(paths), input_dir)
    return paths


# ---------------------------------------------------------------------------
# CSV sources: a directory of files, or a ZIP archive read in place
# ---------------------------------------------------------------------------


class CsvSource:
    """A collection of daily CSVs that can be opened one at a time.

    Attributes:
        names: Member names sorted, which for ``YYYY-MM-DD.csv`` is also
            chronological order.
    """

    names: List[str]

    def open(self, name: str) -> IO[bytes]:
        """Return a fresh binary handle for ``name``.

        Must return a new handle each call: readers open a member twice, once
        for the header and once for the data.

        Args:
            name: One of :attr:`names`.

        Returns:
            An open binary handle the caller must close.
        """
        raise NotImplementedError

    def label(self, name: str) -> str:
        """Return a short display name for ``name``, for logs and errors."""
        return name.rsplit("/", 1)[-1]

    def describe(self) -> str:
        """Return a one-line description of where the data is coming from."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources."""

    def __enter__(self) -> "CsvSource":
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the source on context exit."""
        self.close()

    def __len__(self) -> int:
        """Return the number of members."""
        return len(self.names)


class DirectorySource(CsvSource):
    """Daily CSVs held as individual files on disk."""

    def __init__(
        self,
        input_dir: Path,
        file_glob: str = "*.csv",
        max_files: Optional[int] = None,
    ) -> None:
        """Discover matching files under ``input_dir``.

        Args:
            input_dir: Directory to search, recursively.
            file_glob: Filename pattern.
            max_files: If set, keep only the first this-many files.

        Raises:
            FileNotFoundError: If the directory or matching files are absent.
            NotADirectoryError: If the path is not a directory.
        """
        self._input_dir = Path(input_dir)
        self._paths = discover_csv_files(self._input_dir, file_glob, max_files)
        self.names = [str(path) for path in self._paths]

    def open(self, name: str) -> IO[bytes]:
        """Open one CSV file for reading."""
        return open(name, "rb")

    def describe(self) -> str:
        """Return a one-line description of the directory."""
        return f"{len(self.names)} file(s) in directory {self._input_dir}"


class ZipArchiveSource(CsvSource):
    """Daily CSVs streamed out of a ZIP archive without extracting it."""

    def __init__(
        self,
        archive_path: Path,
        file_glob: str = "*.csv",
        max_files: Optional[int] = None,
    ) -> None:
        """Open the archive and select its matching members.

        Args:
            archive_path: Path to a ``.zip`` file.
            file_glob: Pattern matched against each member's basename, so
                ``"2026-01-*.csv"`` selects one month regardless of nesting.
            max_files: If set, keep only the first this-many members.

        Raises:
            FileNotFoundError: If the archive does not exist or holds no
                matching members.
            DataValidationError: If the file is not a readable ZIP archive.
        """
        self._archive_path = Path(archive_path)
        if not self._archive_path.exists():
            raise FileNotFoundError(f"Archive does not exist: {self._archive_path}")

        try:
            self._archive = zipfile.ZipFile(self._archive_path)
        except zipfile.BadZipFile as exc:
            raise DataValidationError(
                f"{self._archive_path} is not a readable ZIP archive: {exc}"
            ) from exc

        members = sorted(
            info.filename
            for info in self._archive.infolist()
            if not info.is_dir()
            and fnmatch.fnmatch(info.filename.rsplit("/", 1)[-1], file_glob)
        )
        if not members:
            self._archive.close()
            raise FileNotFoundError(
                f"No members matching {file_glob!r} found in {self._archive_path}. "
                f"The archive holds {len(self._archive.namelist())} entr(ies); "
                "the pattern is matched against each member's basename."
            )

        if max_files is not None:
            members = members[:max_files]

        self.names = members
        LOGGER.info(
            "Reading %d member(s) in place from %s", len(members), self._archive_path
        )

    def open(self, name: str) -> IO[bytes]:
        """Open one archive member as a decompressing stream."""
        return self._archive.open(name)

    def describe(self) -> str:
        """Return a one-line description of the archive."""
        return f"{len(self.names)} member(s) in archive {self._archive_path}"

    def close(self) -> None:
        """Close the underlying archive."""
        self._archive.close()


def open_csv_source(
    path: Path, file_glob: str = "*.csv", max_files: Optional[int] = None
) -> CsvSource:
    """Open ``path`` as a source of daily CSVs.

    A ``.zip`` suffix selects :class:`ZipArchiveSource`; anything else is
    treated as a directory.

    Args:
        path: A directory of CSVs, or a ``.zip`` archive containing them.
        file_glob: Filename pattern. For archives this is matched against each
            member's basename.
        max_files: If set, keep only the first this-many files.

    Returns:
        A :class:`CsvSource`. Use it as a context manager so archives close.

    Raises:
        FileNotFoundError: If the path or its matching members are absent.
        DataValidationError: If a ``.zip`` path is not a readable archive.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        return ZipArchiveSource(path, file_glob, max_files)
    return DirectorySource(path, file_glob, max_files)


def _read_csv_frame(
    opener: Callable[[], IO[bytes]],
    label: str,
    kept_columns: Sequence[str] = KEPT_COLUMNS,
) -> Tuple[pd.DataFrame, List[str]]:
    """Read one daily CSV from a handle factory, normalising its column set.

    Args:
        opener: Callable returning a fresh binary handle. Called twice — once
            for the header, once for the data.
        label: Display name used in log and error messages.
        kept_columns: Columns to retain, in output order.

    Returns:
        A tuple of ``(frame, missing_columns)``.

    Raises:
        DataValidationError: If the file is unreadable, empty, or missing any
            of :data:`REQUIRED_COLUMNS`.
    """
    kept_columns = list(kept_columns)
    try:
        with opener() as handle:
            header = pd.read_csv(handle, nrows=0)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"{label} is empty and cannot be parsed") from exc
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Failed to read header of {label}: {exc}") from exc

    available = set(header.columns)
    absent_required = [c for c in REQUIRED_COLUMNS if c not in available]
    if absent_required:
        raise DataValidationError(
            f"{label} is missing required column(s) {absent_required}. "
            "This does not look like a Backblaze daily SMART file."
        )

    present = [c for c in kept_columns if c in available]
    missing = [c for c in kept_columns if c not in available]

    try:
        with opener() as handle:
            frame = pd.read_csv(
                handle,
                usecols=present,
                dtype={c: t for c, t in _COLUMN_DTYPES.items() if c in present},
                parse_dates=["date"] if "date" in present else None,
            )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise DataValidationError(f"Failed to parse {label}: {exc}") from exc

    for column in missing:
        frame[column] = pd.NA if column in ("serial_number", "model") else float("nan")

    if missing:
        LOGGER.debug("%s: %d column(s) absent, filled with NaN", label, len(missing))

    return frame.reindex(columns=kept_columns), missing


def read_daily_file(
    path: Path, kept_columns: Sequence[str] = KEPT_COLUMNS
) -> Tuple[pd.DataFrame, List[str]]:
    """Read one daily Backblaze CSV, normalising its column set.

    The column set is not stable across Backblaze files. Absent requested
    columns are added as ``NaN`` so every daily frame shares one schema.

    Args:
        path: Path to a single daily CSV.
        kept_columns: Columns to retain, in output order.

    Returns:
        A tuple of ``(frame, missing_columns)`` where ``frame`` has exactly
        ``kept_columns`` and ``missing_columns`` lists the requested columns
        that the file did not contain.

    Raises:
        DataValidationError: If the file is unreadable, empty, or missing any
            of :data:`REQUIRED_COLUMNS`.
    """
    path = Path(path)
    return _read_csv_frame(lambda: open(path, "rb"), str(path), kept_columns)


def load_raw_telemetry(config: PipelineConfig) -> Tuple[pd.DataFrame, DatasetSummary]:
    """Load and concatenate every daily file described by ``config``.

    Args:
        config: Pipeline configuration.

    Returns:
        A tuple of ``(frame, summary)``. The frame is deduplicated, cleaned and
        sorted by ``(serial_number, date)``. Labels are not yet assigned, so
        ``positive_labels`` is ``0``.

    Raises:
        FileNotFoundError: If no daily files are found.
        DataValidationError: If every discovered file fails to parse, or the
            combined data violates a schema assumption.
    """
    frames: List[pd.DataFrame] = []
    missing_counts: Dict[str, int] = {}
    failed: List[str] = []

    with open_csv_source(
        config.input_dir, config.file_glob, config.max_files
    ) as source:
        LOGGER.info("Source: %s", source.describe())
        total = len(source)

        for index, name in enumerate(source.names, start=1):
            label = source.label(name)
            try:
                frame, missing = _read_csv_frame(
                    lambda n=name: source.open(n), label, config.kept_columns
                )
            except DataValidationError as exc:
                # Skip malformed days; raises below only if all files fail.
                LOGGER.warning("Skipping %s: %s", label, exc)
                failed.append(label)
                continue

            for column in missing:
                missing_counts[column] = missing_counts.get(column, 0) + 1
            frames.append(frame)

            if index % 100 == 0 or index == total:
                LOGGER.info("Read %d/%d file(s)", index, total)

    if not frames:
        raise DataValidationError(
            f"None of the {total} discovered file(s) could be parsed. "
            f"First failures: {failed[:5]}"
        )
    if failed:
        LOGGER.warning("%d file(s) skipped as unparseable: %s", len(failed), failed[:10])

    raw = pd.concat(frames, ignore_index=True, copy=False)
    del frames
    LOGGER.info("Concatenated %d raw row(s)", len(raw))

    clean, duplicates_dropped = _clean_telemetry(raw)

    summary = DatasetSummary(
        files_read=total - len(failed),
        records=len(clean),
        unique_drives=int(clean["serial_number"].nunique()),
        failure_events=int((clean["failure"] == 1).sum()),
        positive_labels=0,
        date_min=clean["date"].min() if not clean.empty else None,
        date_max=clean["date"].max() if not clean.empty else None,
        rows_dropped_duplicate=duplicates_dropped,
        missing_columns=missing_counts,
    )
    return clean, summary


def _clean_telemetry(frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Validate, deduplicate and sort the concatenated telemetry.

    Args:
        frame: Concatenated raw frame with the canonical column set.

    Returns:
        A tuple of ``(clean_frame, duplicate_rows_dropped)``.

    Raises:
        DataValidationError: If dates cannot be parsed, serial numbers are
            missing, or ``failure`` holds values outside ``{0, 1}``.
    """
    frame = frame.copy()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    unparsed_dates = int(frame["date"].isna().sum())
    if unparsed_dates:
        raise DataValidationError(
            f"{unparsed_dates:,} row(s) have an unparseable date. Refusing to "
            "guess: chronological splitting and 30-day windows both depend on "
            "the date being exact."
        )

    missing_serials = int(frame["serial_number"].isna().sum())
    if missing_serials:
        raise DataValidationError(
            f"{missing_serials:,} row(s) have no serial_number, so they cannot "
            "be attributed to a drive."
        )

    failure_values = set(frame["failure"].dropna().unique().tolist())
    unexpected = failure_values - {0.0, 1.0}
    if unexpected:
        raise DataValidationError(
            f"`failure` must be 0 or 1; found unexpected value(s) {sorted(unexpected)}"
        )
    if frame["failure"].isna().any():
        n_missing = int(frame["failure"].isna().sum())
        LOGGER.warning(
            "%d row(s) have a missing `failure` flag; treating them as 0 "
            "(not-failed-on-that-day), which is how Backblaze records a live drive.",
            n_missing,
        )
        frame["failure"] = frame["failure"].fillna(0.0)
    frame["failure"] = frame["failure"].astype("int8")

    if "capacity_bytes" in frame.columns:
        sentinels = int((frame["capacity_bytes"] == _CAPACITY_SENTINEL).sum())
        if sentinels:
            LOGGER.info(
                "capacity_bytes == %d (Backblaze's 'unknown' sentinel) in %d row(s); "
                "converted to NaN",
                int(_CAPACITY_SENTINEL),
                sentinels,
            )
            frame.loc[frame["capacity_bytes"] == _CAPACITY_SENTINEL, "capacity_bytes"] = (
                float("nan")
            )

    before = len(frame)
    # Sort failure last so keep="last" retains the failure-flagged duplicate.
    frame = frame.sort_values(
        ["serial_number", "date", "failure"], kind="mergesort"
    ).drop_duplicates(subset=["serial_number", "date"], keep="last")
    duplicates_dropped = before - len(frame)
    if duplicates_dropped:
        LOGGER.warning(
            "Dropped %d duplicate (serial_number, date) row(s), keeping the "
            "failure-flagged row where one existed",
            duplicates_dropped,
        )

    frame = frame.sort_values(["serial_number", "date"], kind="mergesort").reset_index(
        drop=True
    )
    return frame, duplicates_dropped


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def add_failure_labels(
    frame: pd.DataFrame,
    horizon_days: int = 7,
    drop_post_failure_rows: bool = True,
    drop_censored_rows: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Attach the forward-looking failure label to each observation.

    The failure date is the first date on which ``failure == 1``. A row is
    positive when ``1 <= (failure_date - date).days <= horizon_days``, so the
    label uses only days after the observation.

    Rows whose outcome cannot be established are flagged ``label_censored``
    rather than assumed healthy.

    Args:
        frame: Cleaned telemetry sorted by ``(serial_number, date)``.
        horizon_days: Length of the forward label window, in calendar days.
        drop_post_failure_rows: Drop rows on or after the failure date.
        drop_censored_rows: Drop rows whose negative label is unverifiable.

    Returns:
        A tuple of ``(labelled_frame, drop_counts)`` where ``drop_counts`` maps
        ``"post_failure"`` and ``"censored"`` to the number of rows removed.

    Raises:
        DataValidationError: If ``horizon_days < 1`` or required columns are
            absent.
    """
    if horizon_days < 1:
        raise DataValidationError(f"horizon_days must be >= 1, got {horizon_days!r}")
    for column in ("date", "serial_number", "failure"):
        if column not in frame.columns:
            raise DataValidationError(f"Cannot label: column {column!r} is absent")

    labelled = frame.copy()

    failure_dates = (
        labelled.loc[labelled["failure"] == 1]
        .groupby("serial_number", observed=True)["date"]
        .min()
        .rename("failure_date")
    )
    multi_failure = (
        labelled.loc[labelled["failure"] == 1]
        .groupby("serial_number", observed=True)["date"]
        .size()
    )
    repeat_offenders = int((multi_failure > 1).sum())
    if repeat_offenders:
        LOGGER.warning(
            "%d drive(s) carry more than one failure=1 row; using the earliest "
            "failure date for each",
            repeat_offenders,
        )

    labelled = labelled.merge(failure_dates, on="serial_number", how="left")
    labelled["days_to_failure"] = (
        labelled["failure_date"] - labelled["date"]
    ).dt.days.astype("Float64")

    # Last day this drive is observed at all.
    labelled["observed_through"] = labelled.groupby(
        "serial_number", observed=True
    )["date"].transform("max")

    days = labelled["days_to_failure"]
    labelled[LABEL_COLUMN] = (
        (days >= 1) & (days <= horizon_days)
    ).fillna(False).astype("int8")

    # A negative is only verifiable if the drive is observed horizon_days later.
    days_observed_after = (labelled["observed_through"] - labelled["date"]).dt.days
    labelled["label_censored"] = (
        labelled["failure_date"].isna() & (days_observed_after < horizon_days)
    )

    drop_counts = {"post_failure": 0, "censored": 0}

    if drop_post_failure_rows:
        post_failure = days.notna() & (days <= 0)
        drop_counts["post_failure"] = int(post_failure.sum())
        labelled = labelled.loc[~post_failure.fillna(False)]
        LOGGER.info(
            "Dropped %d row(s) on/after a failure date (the event itself is not "
            "a forecasting point)",
            drop_counts["post_failure"],
        )

    if drop_censored_rows:
        censored = labelled["label_censored"]
        drop_counts["censored"] = int(censored.sum())
        labelled = labelled.loc[~censored]
        LOGGER.info(
            "Dropped %d row(s) with an unverifiable negative label (drive not "
            "observed for a further %d day(s))",
            drop_counts["censored"],
            horizon_days,
        )

    labelled = labelled.sort_values(
        ["serial_number", "date"], kind="mergesort"
    ).reset_index(drop=True)

    positives = int(labelled[LABEL_COLUMN].sum())
    if positives == 0:
        LOGGER.warning(
            "No positive labels were produced. With a %d-day horizon this "
            "usually means the loaded date range contains no failures, or too "
            "few consecutive days for a failure to be preceded by observations.",
            horizon_days,
        )

    return labelled, drop_counts


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    config: PipelineConfig, output_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, DatasetSummary]:
    """Load, clean, label and optionally persist the telemetry.

    Args:
        config: Pipeline configuration.
        output_path: Optional ``.parquet`` (or ``.csv``) destination for the
            labelled table. Parent directories are created as needed.

    Returns:
        A tuple of ``(labelled_frame, summary)``.

    Raises:
        FileNotFoundError: If the input directory or daily files are missing.
        DataValidationError: If the data violates a pipeline assumption.
    """
    frame, summary = load_raw_telemetry(config)

    labelled, drop_counts = add_failure_labels(
        frame,
        horizon_days=config.horizon_days,
        drop_post_failure_rows=config.drop_post_failure_rows,
        drop_censored_rows=config.drop_censored_rows,
    )

    summary.records = len(labelled)
    summary.unique_drives = int(labelled["serial_number"].nunique())
    summary.failure_events = int(frame["failure"].sum())
    summary.positive_labels = int(labelled[LABEL_COLUMN].sum())
    summary.date_min = labelled["date"].min() if not labelled.empty else None
    summary.date_max = labelled["date"].max() if not labelled.empty else None
    summary.rows_dropped_post_failure = drop_counts["post_failure"]
    summary.rows_dropped_censored = drop_counts["censored"]

    if output_path is not None:
        write_labelled_dataset(labelled, output_path)

    return labelled, summary


def write_labelled_dataset(frame: pd.DataFrame, output_path: Path) -> Path:
    """Persist the labelled table to Parquet (preferred) or CSV.

    Args:
        frame: Labelled dataset.
        output_path: Destination path; ``.parquet`` and ``.csv`` are supported.

    Returns:
        The path written.

    Raises:
        ValueError: If the file extension is not supported.
        OSError: If the file cannot be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".parquet":
        try:
            frame.to_parquet(output_path, index=False)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Writing Parquet requires pyarrow. Install it with "
                "`pip install -r requirements.txt`, or pass a .csv path."
            ) from exc
    elif suffix == ".csv":
        frame.to_csv(output_path, index=False)
    else:
        raise ValueError(
            f"Unsupported output extension {suffix!r}; use '.parquet' or '.csv'"
        )

    LOGGER.info("Wrote %d labelled row(s) to %s", len(frame), output_path)
    return output_path


def load_labelled_dataset(path: Path) -> pd.DataFrame:
    """Read a labelled dataset previously written by :func:`run_pipeline`.

    Args:
        path: Path to a ``.parquet`` or ``.csv`` file.

    Returns:
        The labelled dataframe, with ``date`` parsed and rows sorted by
        ``(serial_number, date)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the extension is unsupported.
        DataValidationError: If the label column is absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Labelled dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, parse_dates=["date"])
    else:
        raise ValueError(f"Unsupported input extension {suffix!r}")

    if LABEL_COLUMN not in frame.columns:
        raise DataValidationError(
            f"{path} has no {LABEL_COLUMN!r} column; it was not produced by "
            "this pipeline."
        )

    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.sort_values(["serial_number", "date"], kind="mergesort").reset_index(
        drop=True
    )


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging for command-line use.

    Args:
        verbose: Emit ``DEBUG`` records instead of ``INFO``.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Load Backblaze daily SMART CSVs and attach a forward-looking "
            "failure label."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing daily Backblaze CSV files (searched recursively).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for the labelled table (.parquet or .csv).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="Forward label window in calendar days (default: 7).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Read only the first N daily files (useful for a smoke test).",
    )
    parser.add_argument(
        "--keep-censored",
        action="store_true",
        help=(
            "Keep negatives whose outcome is unverifiable because the drive "
            "leaves the dataset inside the horizon."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Run the pipeline from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` on a handled data error,
        ``2`` on a missing input path.
    """
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    configure_logging(args.verbose)

    try:
        config = PipelineConfig(
            input_dir=args.input_dir,
            horizon_days=args.horizon_days,
            max_files=args.max_files,
            drop_censored_rows=not args.keep_censored,
        )
        _, summary = run_pipeline(config, output_path=args.output)
    except (FileNotFoundError, NotADirectoryError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except DataValidationError as exc:
        LOGGER.error("Data validation failed: %s", exc)
        return 1

    summary.print_report()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
