"""Tests for the ingestion and labelling pipeline.

Expected counts come from the fixture in ``conftest.py`` and are annotated with
the arithmetic that produces them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline import (
    LABEL_COLUMN,
    DataValidationError,
    PipelineConfig,
    add_failure_labels,
    discover_csv_files,
    load_labelled_dataset,
    load_raw_telemetry,
    open_csv_source,
    read_daily_file,
    run_pipeline,
    write_labelled_dataset,
)
from tests.conftest import FILES_WITHOUT_SMART_187, FIXTURE_SMART_COLUMNS

KEPT = (
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
) + FIXTURE_SMART_COLUMNS


class TestDiscovery:
    """File discovery and its failure modes."""

    def test_finds_every_daily_file(self, raw_dir: Path) -> None:
        assert len(discover_csv_files(raw_dir)) == 20

    def test_results_are_chronological(self, raw_dir: Path) -> None:
        names = [p.name for p in discover_csv_files(raw_dir)]
        assert names == sorted(names)
        assert names[0] == "2024-01-01.csv"

    def test_max_files_truncates(self, raw_dir: Path) -> None:
        assert len(discover_csv_files(raw_dir, max_files=3)) == 3

    def test_missing_directory_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            discover_csv_files(Path("/nonexistent/backblaze"))

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No files matching"):
            discover_csv_files(tmp_path)

    def test_file_instead_of_directory_raises(self, raw_dir: Path) -> None:
        with pytest.raises(NotADirectoryError):
            discover_csv_files(raw_dir / "2024-01-01.csv")


class TestSchemaNormalisation:
    """Columns absent from a daily file must become NaN, not vanish."""

    def test_absent_column_is_reported(self, raw_dir: Path) -> None:
        _, missing = read_daily_file(raw_dir / "2024-01-01.csv", KEPT)
        assert missing == ["smart_187_raw"]

    def test_absent_column_is_filled_with_nan(self, raw_dir: Path) -> None:
        frame, _ = read_daily_file(raw_dir / "2024-01-01.csv", KEPT)
        assert "smart_187_raw" in frame.columns
        assert frame["smart_187_raw"].isna().all()

    def test_column_order_is_stable_across_files(self, raw_dir: Path) -> None:
        # Files with and without the drifting column must still concatenate.
        early, _ = read_daily_file(raw_dir / "2024-01-01.csv", KEPT)
        late, _ = read_daily_file(raw_dir / "2024-01-20.csv", KEPT)
        assert list(early.columns) == list(late.columns) == list(KEPT)

    def test_missing_columns_are_counted_per_file(
        self, pipeline_config: PipelineConfig
    ) -> None:
        _, summary = load_raw_telemetry(pipeline_config)
        assert summary.missing_columns["smart_187_raw"] == FILES_WITHOUT_SMART_187

    def test_file_without_required_column_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "2024-02-01.csv"
        pd.DataFrame({"date": ["2024-02-01"], "model": ["X"]}).to_csv(bad, index=False)
        with pytest.raises(DataValidationError, match="missing required column"):
            read_daily_file(bad, KEPT)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "2024-02-01.csv"
        empty.write_text("")
        with pytest.raises(DataValidationError, match="empty"):
            read_daily_file(empty, KEPT)


class TestCleaning:
    """Validation and normalisation of the concatenated frame."""

    def test_capacity_sentinel_becomes_nan(self, labelled: pd.DataFrame) -> None:
        # SN_C reports capacity_bytes == -1, Backblaze's "unknown" marker.
        sn_c = labelled.loc[labelled["serial_number"] == "SN_C", "capacity_bytes"]
        assert sn_c.isna().all()

    def test_rows_are_sorted_by_drive_then_date(self, labelled: pd.DataFrame) -> None:
        expected = labelled.sort_values(["serial_number", "date"], kind="mergesort")
        pd.testing.assert_frame_equal(labelled, expected.reset_index(drop=True))

    def test_one_row_per_drive_day(self, labelled: pd.DataFrame) -> None:
        assert not labelled.duplicated(subset=["serial_number", "date"]).any()

    def test_duplicate_rows_keep_the_failure(self, tmp_path: Path) -> None:
        # Same drive-day twice; deduplication must keep the failure.
        day = tmp_path / "2024-03-01.csv"
        pd.DataFrame(
            {
                "date": ["2024-03-01", "2024-03-01"],
                "serial_number": ["SN_X", "SN_X"],
                "model": ["M", "M"],
                "capacity_bytes": [4e12, 4e12],
                "failure": [0, 1],
                "smart_5_raw": [1.0, 1.0],
            }
        ).to_csv(day, index=False)

        frame, summary = load_raw_telemetry(
            PipelineConfig(input_dir=tmp_path, smart_columns=("smart_5_raw",))
        )
        assert summary.rows_dropped_duplicate == 1
        assert len(frame) == 1
        assert int(frame["failure"].iloc[0]) == 1

    def test_invalid_failure_flag_raises(self, tmp_path: Path) -> None:
        day = tmp_path / "2024-03-01.csv"
        pd.DataFrame(
            {
                "date": ["2024-03-01"],
                "serial_number": ["SN_X"],
                "failure": [7],
                "smart_5_raw": [0.0],
            }
        ).to_csv(day, index=False)
        with pytest.raises(DataValidationError, match="must be 0 or 1"):
            load_raw_telemetry(
                PipelineConfig(input_dir=tmp_path, smart_columns=("smart_5_raw",))
            )

    def test_unparseable_date_raises(self, tmp_path: Path) -> None:
        day = tmp_path / "2024-03-01.csv"
        pd.DataFrame(
            {
                "date": ["not-a-date"],
                "serial_number": ["SN_X"],
                "failure": [0],
                "smart_5_raw": [0.0],
            }
        ).to_csv(day, index=False)
        with pytest.raises(DataValidationError, match="unparseable date"):
            load_raw_telemetry(
                PipelineConfig(input_dir=tmp_path, smart_columns=("smart_5_raw",))
            )


class TestLabelling:
    """The forward label, and the two ways it can be silently wrong."""

    def test_summary_counts(self, pipeline_config: PipelineConfig) -> None:
        _, summary = run_pipeline(pipeline_config)
        # SN_A: 15 observations, the failure day dropped        -> 14
        # SN_B: 20 observations, last 7 censored                -> 13
        # SN_C: 10 observations, last 7 censored                ->  3
        assert summary.records == 14 + 13 + 3
        assert summary.unique_drives == 3
        assert summary.failure_events == 1
        assert summary.files_read == 20

    def test_positive_window_is_exactly_the_horizon(
        self, labelled: pd.DataFrame
    ) -> None:
        # SN_A fails on 01-15, so 01-08..01-14 are the seven positive days.
        sn_a = labelled[labelled["serial_number"] == "SN_A"].set_index("date")
        assert int(sn_a.loc["2024-01-07", LABEL_COLUMN]) == 0  # 8 days out
        assert int(sn_a.loc["2024-01-08", LABEL_COLUMN]) == 1  # 7 days out
        assert int(sn_a.loc["2024-01-14", LABEL_COLUMN]) == 1  # 1 day out
        assert int(labelled[LABEL_COLUMN].sum()) == 7

    def test_failure_day_is_not_a_prediction_point(
        self, labelled: pd.DataFrame
    ) -> None:
        sn_a = labelled[labelled["serial_number"] == "SN_A"]
        assert pd.Timestamp("2024-01-15") not in set(sn_a["date"])

    def test_no_row_survives_on_or_after_its_failure_date(
        self, labelled: pd.DataFrame
    ) -> None:
        assert (labelled["days_to_failure"].dropna() > 0).all()

    def test_drives_without_a_failure_are_never_positive(
        self, labelled: pd.DataFrame
    ) -> None:
        never_failed = labelled["failure_date"].isna()
        assert labelled.loc[never_failed, LABEL_COLUMN].eq(0).all()

    def test_unverifiable_negatives_are_dropped(
        self, pipeline_config: PipelineConfig
    ) -> None:
        # SN_B is observed through 01-20 and SN_C through 01-10; the final
        # seven days of each have no confirmable outcome. 7 + 7 = 14.
        _, summary = run_pipeline(pipeline_config)
        assert summary.rows_dropped_censored == 14

    def test_censored_rows_can_be_kept(self, raw_dir: Path) -> None:
        config = PipelineConfig(
            input_dir=raw_dir,
            smart_columns=FIXTURE_SMART_COLUMNS,
            drop_censored_rows=False,
        )
        frame, summary = run_pipeline(config)
        assert summary.rows_dropped_censored == 0
        assert int(frame["label_censored"].sum()) == 14

    @pytest.mark.parametrize(
        "horizon, expected_positives",
        [
            (1, 1),  # only 01-14
            (3, 3),  # 01-12 .. 01-14
            (7, 7),  # 01-08 .. 01-14
            (14, 14),  # 01-01 .. 01-14, SN_A's whole observed life
        ],
    )
    def test_horizon_is_honoured(
        self, pipeline_config: PipelineConfig, horizon: int, expected_positives: int
    ) -> None:
        raw, _ = load_raw_telemetry(pipeline_config)
        frame, _ = add_failure_labels(raw, horizon_days=horizon)
        assert int(frame[LABEL_COLUMN].sum()) == expected_positives

    def test_label_never_depends_on_the_present_row(
        self, pipeline_config: PipelineConfig
    ) -> None:
        # Every positive must sit strictly in the future of its observation.
        raw, _ = load_raw_telemetry(pipeline_config)
        frame, _ = add_failure_labels(raw, horizon_days=7)
        positives = frame[frame[LABEL_COLUMN] == 1]
        gap = (positives["failure_date"] - positives["date"]).dt.days
        assert gap.between(1, 7).all()

    def test_zero_horizon_raises(self, labelled: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="horizon_days must be >= 1"):
            add_failure_labels(labelled, horizon_days=0)


class TestConfigValidation:
    """Configuration errors surface at construction, not mid-run."""

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"horizon_days": 0}, "horizon_days"),
            ({"horizon_days": -3}, "horizon_days"),
            ({"smart_columns": ()}, "smart_columns"),
            ({"max_files": 0}, "max_files"),
        ],
    )
    def test_invalid_config_raises(self, raw_dir: Path, kwargs, match: str) -> None:
        with pytest.raises(DataValidationError, match=match):
            PipelineConfig(input_dir=raw_dir, **kwargs)


class TestPersistence:
    """Round-tripping the labelled table."""

    def test_parquet_round_trip(self, labelled: pd.DataFrame, tmp_path: Path) -> None:
        path = write_labelled_dataset(labelled, tmp_path / "labelled.parquet")
        restored = load_labelled_dataset(path)
        assert len(restored) == len(labelled)
        assert int(restored[LABEL_COLUMN].sum()) == int(labelled[LABEL_COLUMN].sum())

    def test_csv_round_trip(self, labelled: pd.DataFrame, tmp_path: Path) -> None:
        path = write_labelled_dataset(labelled, tmp_path / "labelled.csv")
        restored = load_labelled_dataset(path)
        assert len(restored) == len(labelled)

    def test_unsupported_extension_raises(
        self, labelled: pd.DataFrame, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported output extension"):
            write_labelled_dataset(labelled, tmp_path / "labelled.xlsx")

    def test_loading_a_frame_without_labels_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "unlabelled.csv"
        pd.DataFrame({"date": ["2024-01-01"], "serial_number": ["SN_X"]}).to_csv(
            path, index=False
        )
        with pytest.raises(DataValidationError, match="not produced by this pipeline"):
            load_labelled_dataset(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_labelled_dataset(tmp_path / "absent.parquet")


class TestZipArchiveSource:
    """Reading daily CSVs straight out of a ZIP, without extracting it."""

    @staticmethod
    def _make_archive(raw_dir: Path, destination: Path, prefix: str = "data/") -> Path:
        """Zip the fixture's daily files under a nested folder."""
        import zipfile

        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(raw_dir.glob("*.csv")):
                archive.write(path, arcname=f"{prefix}{path.name}")
        return destination

    def test_archive_and_directory_agree_exactly(
        self, raw_dir: Path, tmp_path: Path, pipeline_config: PipelineConfig
    ) -> None:
        # Same bytes in, same frame out.
        archive = self._make_archive(raw_dir, tmp_path / "q.zip")
        from_dir, summary_dir = load_raw_telemetry(pipeline_config)
        from_zip, summary_zip = load_raw_telemetry(
            PipelineConfig(input_dir=archive, smart_columns=FIXTURE_SMART_COLUMNS)
        )
        pd.testing.assert_frame_equal(from_dir, from_zip)
        assert summary_zip.records == summary_dir.records
        assert summary_zip.files_read == summary_dir.files_read

    def test_nothing_is_extracted(self, raw_dir: Path, tmp_path: Path) -> None:
        archive = self._make_archive(raw_dir, tmp_path / "q.zip")
        before = {p for p in tmp_path.rglob("*") if p.is_file()}
        load_raw_telemetry(
            PipelineConfig(input_dir=archive, smart_columns=FIXTURE_SMART_COLUMNS)
        )
        assert {p for p in tmp_path.rglob("*") if p.is_file()} == before

    def test_glob_matches_member_basename(self, raw_dir: Path, tmp_path: Path) -> None:
        # Nested folders must not defeat the pattern.
        archive = self._make_archive(raw_dir, tmp_path / "q.zip", prefix="data_Q1/x/")
        with open_csv_source(archive, "2024-01-0[1-3].csv") as source:
            assert [source.label(n) for n in source.names] == [
                "2024-01-01.csv",
                "2024-01-02.csv",
                "2024-01-03.csv",
            ]

    def test_members_are_chronological(self, raw_dir: Path, tmp_path: Path) -> None:
        archive = self._make_archive(raw_dir, tmp_path / "q.zip")
        with open_csv_source(archive) as source:
            labels = [source.label(n) for n in source.names]
        assert labels == sorted(labels)

    def test_max_files_truncates(self, raw_dir: Path, tmp_path: Path) -> None:
        archive = self._make_archive(raw_dir, tmp_path / "q.zip")
        with open_csv_source(archive, "*.csv", max_files=4) as source:
            assert len(source) == 4

    def test_open_returns_a_fresh_handle_each_call(
        self, raw_dir: Path, tmp_path: Path
    ) -> None:
        # The reader opens each member twice: header, then data.
        archive = self._make_archive(raw_dir, tmp_path / "q.zip")
        with open_csv_source(archive) as source:
            name = source.names[0]
            with source.open(name) as first:
                first.read()
            with source.open(name) as second:
                assert len(second.read()) > 0

    def test_missing_archive_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Archive does not exist"):
            open_csv_source(tmp_path / "absent.zip")

    def test_no_matching_members_raises(self, raw_dir: Path, tmp_path: Path) -> None:
        archive = self._make_archive(raw_dir, tmp_path / "q.zip")
        with pytest.raises(FileNotFoundError, match="No members matching"):
            open_csv_source(archive, "2099-*.csv")

    def test_corrupt_archive_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"this is not a zip file")
        with pytest.raises(DataValidationError, match="not a readable ZIP"):
            open_csv_source(bad)
