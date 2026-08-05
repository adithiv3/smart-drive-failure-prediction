"""Tests for the drive-day panel experiment.

Covers the properties that would otherwise inflate every metric: a prediction
row cannot see its own day, already-failed drives are excluded, and
unverifiable survivals are dropped rather than counted as healthy.

The fixture is a 12-day synthetic archive with three drives; the window
constants are shrunk to ``window=3, horizon=2`` so every expected row can be
enumerated by hand.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

import src.panel_experiment as panel
from src.data_pipeline import DataValidationError, open_csv_source
from src.experiment import SMART_ATTRIBUTES

WINDOW = 3
HORIZON = 2
N_DAYS = 12

#: Day index on which P1 records failure=1.
P1_FAILURE_DAY = 7
#: Last day G1 is present in the fleet.
G1_LAST_DAY = 5
#: Day from which H1's smart_5_raw jumps; used to prove day t is excluded.
SPIKE_DAY = 6
SPIKE_VALUE = 100.0


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a 12-day synthetic archive with three drives.

    ``P1`` fails on day 7. ``H1`` never fails and is always present, with a
    ``smart_5_raw`` spike from day 6 that no earlier prediction date may see.
    ``G1`` leaves the fleet after day 5 without failing.
    """
    directory = tmp_path_factory.mktemp("panel")
    destination = directory / "q.zip"
    dates = pd.date_range("2026-01-01", periods=N_DAYS, freq="D")

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        for day, date in enumerate(dates):
            rows: List[Dict[str, object]] = []

            def add(serial: str, failure: int, smart_5: float) -> None:
                row: Dict[str, object] = {
                    "date": date,
                    "serial_number": serial,
                    "failure": failure,
                }
                for attribute in SMART_ATTRIBUTES:
                    row[attribute] = 0.0
                row["smart_5_raw"] = smart_5
                rows.append(row)

            if day <= P1_FAILURE_DAY:
                add("P1", 1 if day == P1_FAILURE_DAY else 0, float(day))
            add("H1", 0, SPIKE_VALUE if day >= SPIKE_DAY else 0.0)
            if day <= G1_LAST_DAY:
                add("G1", 0, 0.0)

            path = directory / f"{date.date()}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            bundle.write(path, arcname=f"data_Q1_2026/{date.date()}.csv")
            path.unlink()

    return destination


@pytest.fixture(scope="module")
def shrunk(archive: Path) -> Tuple[panel.Calendar, panel.DriveMetadata]:
    """Return the calendar and metadata for the synthetic archive."""
    calendar = panel.scan_calendar(archive)
    metadata = panel.scan_drives(archive, calendar)
    return calendar, metadata


@pytest.fixture(autouse=True)
def _shrink_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the window and horizon so expectations stay hand-checkable."""
    monkeypatch.setattr(panel, "WINDOW_DAYS", WINDOW)
    monkeypatch.setattr(panel, "HORIZON_DAYS", HORIZON)


def collect(archive: Path, calendar, metadata, days) -> Dict[int, panel.Block]:
    """Materialise blocks for the given prediction dates, keyed by day."""
    return {
        block.day: block
        for block in panel.stream_blocks(
            archive, calendar, metadata, days, quiet=True
        )
    }


class TestCalendar:
    """Calendar parsing and contiguity."""

    def test_days_are_discovered(self, shrunk) -> None:
        calendar, _ = shrunk
        assert len(calendar) == N_DAYS
        assert calendar.label(0) == "2026-01-01"
        assert calendar.label(N_DAYS - 1) == "2026-01-12"

    def test_a_gap_is_fatal(self, tmp_path: Path) -> None:
        # A missing day corrupts every window spanning it.
        bundle = tmp_path / "gap.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            for label in ("2026-01-01", "2026-01-02", "2026-01-04"):
                path = tmp_path / f"{label}.csv"
                pd.DataFrame(
                    {"date": [label], "serial_number": ["A"], "failure": [0]}
                ).to_csv(path, index=False)
                archive.write(path, arcname=f"{label}.csv")
                path.unlink()
        with pytest.raises(DataValidationError, match="not contiguous"):
            panel.scan_calendar(bundle)


class TestDriveMetadata:
    """Presence bitmap and failure days."""

    def test_drives_are_indexed(self, shrunk) -> None:
        _, metadata = shrunk
        assert metadata.n_drives == 3
        assert set(metadata.index) == {"P1", "H1", "G1"}

    def test_failure_day_is_recorded(self, shrunk) -> None:
        _, metadata = shrunk
        position = metadata.index.get_loc("P1")
        assert int(metadata.failure_day[position]) == P1_FAILURE_DAY

    def test_non_failing_drives_are_marked_minus_one(self, shrunk) -> None:
        _, metadata = shrunk
        for serial in ("H1", "G1"):
            assert int(metadata.failure_day[metadata.index.get_loc(serial)]) == -1

    def test_presence_tracks_fleet_departure(self, shrunk) -> None:
        _, metadata = shrunk
        position = metadata.index.get_loc("G1")
        assert metadata.presence[: G1_LAST_DAY + 1, position].all()
        assert not metadata.presence[G1_LAST_DAY + 1 :, position].any()


class TestEligibility:
    """Which prediction dates and rows survive."""

    def test_eligible_dates_need_both_windows(self, shrunk) -> None:
        calendar, _ = shrunk
        # day >= 3 (full history) and day + 2 <= 11 (full label window).
        assert panel.eligible_prediction_days(calendar) == [3, 4, 5, 6, 7, 8, 9]

    def test_row_counts_per_date(self, archive: Path, shrunk) -> None:
        calendar, metadata = shrunk
        blocks = collect(archive, calendar, metadata, [3, 4, 5, 6, 7, 8, 9])
        serials = {
            day: sorted(metadata.index[block.drives]) for day, block in blocks.items()
        }
        # t=3: all three drives verifiable (G1 present on days 4 and 5).
        assert serials[3] == ["G1", "H1", "P1"]
        # t=4: G1's future window reaches day 6, where it is absent -> unknown.
        assert serials[4] == ["H1", "P1"]
        assert serials[5] == ["H1", "P1"]
        assert serials[6] == ["H1", "P1"]
        # t>=7: P1 has already failed on day 7.
        assert serials[7] == ["H1"]
        assert serials[8] == ["H1"]
        assert serials[9] == ["H1"]

    def test_total_rows_and_positives(self, archive: Path, shrunk) -> None:
        calendar, metadata = shrunk
        counts = panel.PanelCounts()
        for _ in panel.stream_blocks(
            archive, calendar, metadata, [3, 4, 5, 6, 7, 8, 9], counts=counts, quiet=True
        ):
            pass
        # P1 4 rows + H1 7 rows + G1 1 row.
        assert counts.rows_emitted == 12
        # P1 is positive at t=5 (horizon 6,7) and t=6 (horizon 7,8).
        assert counts.positives == 2
        # P1 has already failed at t=7, 8 and 9.
        assert counts.excluded_already_failed == 3
        # G1 leaves after day 5, so its future window is incomplete at
        # t=4 (days 5,6), 5 (6,7), 6 (7,8), 7 (8,9) and 8 (9,10) -> five rows
        # whose survival cannot be verified.
        assert counts.excluded_unverifiable == 5
        # At t=9 G1's history window (days 6,7,8) is empty, so it is dropped
        # for want of history before the future window is even considered.
        assert counts.excluded_no_history == 1

    def test_labels_mark_only_the_horizon(self, archive: Path, shrunk) -> None:
        calendar, metadata = shrunk
        blocks = collect(archive, calendar, metadata, [3, 4, 5, 6])
        for day, expected in ((3, 0), (4, 0), (5, 1), (6, 1)):
            position = list(metadata.index[blocks[day].drives]).index("P1")
            assert int(blocks[day].labels[position]) == expected, f"day {day}"


class TestNoLeakage:
    """A prediction row must not see its own day, nor any later one."""

    def test_window_excludes_the_prediction_date(self, archive: Path, shrunk) -> None:
        calendar, metadata = shrunk
        blocks = collect(archive, calendar, metadata, [6, 7, 8, 9])
        column = panel.FEATURE_NAMES.index("smart_5_raw_max")

        # H1 jumps to 100 on day 6; date 6 uses days 3-5 and must still see 0.
        row = list(metadata.index[blocks[6].drives]).index("H1")
        assert blocks[6].features[row, column] == pytest.approx(0.0)

        # Date 7 uses days 4-6, so the spike is now legitimately visible.
        row = list(metadata.index[blocks[7].drives]).index("H1")
        assert blocks[7].features[row, column] == pytest.approx(SPIKE_VALUE)

    def test_features_match_the_prior_window_exactly(
        self, archive: Path, shrunk
    ) -> None:
        calendar, metadata = shrunk
        blocks = collect(archive, calendar, metadata, [5])
        row = list(metadata.index[blocks[5].drives]).index("P1")
        features = blocks[5].features[row]

        # P1's smart_5_raw equals its day index, so days 2,3,4 hold 2,3,4.
        window = np.array([2.0, 3.0, 4.0])
        for statistic, expected in (
            ("latest", 4.0),
            ("mean", window.mean()),
            ("min", 2.0),
            ("max", 4.0),
            ("std", window.std(ddof=1)),
            ("change", 2.0),
            ("count", 3.0),
        ):
            index = panel.FEATURE_NAMES.index(f"smart_5_raw_{statistic}")
            assert features[index] == pytest.approx(expected), statistic

    def test_observation_count_reflects_days_present(
        self, archive: Path, shrunk
    ) -> None:
        calendar, metadata = shrunk
        blocks = collect(archive, calendar, metadata, [3])
        column = panel.FEATURE_NAMES.index("observations_30d")
        for serial in ("P1", "H1", "G1"):
            row = list(metadata.index[blocks[3].drives]).index(serial)
            assert blocks[3].features[row, column] == pytest.approx(3.0)

    def test_evicted_days_do_not_linger(self, archive: Path, shrunk) -> None:
        # G1's window at date 9 (days 6-8) is empty, so it must be excluded.
        calendar, metadata = shrunk
        blocks = collect(archive, calendar, metadata, [9])
        assert "G1" not in set(metadata.index[blocks[9].drives])


class TestSplitting:
    """Chronological splitting of prediction dates."""

    def test_split_is_ordered_and_disjoint(self) -> None:
        days = list(range(53))
        train, validation, test = panel.split_days(days)
        assert train and validation and test
        assert max(train) < min(validation) < max(validation) < min(test)
        assert sorted(train + validation + test) == days

    def test_no_date_appears_in_two_folds(self) -> None:
        train, validation, test = panel.split_days(list(range(53)))
        assert not (set(train) & set(validation))
        assert not (set(validation) & set(test))
        assert not (set(train) & set(test))


class TestThresholdSelection:
    """The degeneracy guard that the previous run needed."""

    def test_rejects_a_threshold_of_one(self) -> None:
        # Scores saturating at 1.0 must not yield a threshold of 1.0.
        labels = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 1.0, 1.0])
        threshold, f1, degenerate = panel.choose_threshold(labels, scores)
        assert threshold < 1.0
        assert f1 > 0.0
        del degenerate

    def test_picks_a_separating_threshold(self) -> None:
        labels = np.array([0, 0, 0, 0, 1, 1])
        scores = np.array([0.01, 0.02, 0.03, 0.04, 0.9, 0.95])
        threshold, f1, _ = panel.choose_threshold(labels, scores)
        assert 0.04 < threshold <= 0.9
        assert f1 == pytest.approx(1.0)


class TestEventMetrics:
    """Drive-level event aggregation over repeated rows."""

    def test_one_alert_in_the_window_detects_the_event(self) -> None:
        # Drive 7 has three positive rows; any one alert catches the failure.
        fold = panel.FoldScores(
            labels=np.array([1, 1, 1, 0, 0], dtype=np.int8),
            drives=np.array([7, 7, 7, 8, 9], dtype=np.int32),
            days=np.array([1, 2, 3, 1, 1], dtype=np.int16),
        )
        alerts = np.array([False, True, False, False, False])
        result = panel.evaluate(fold, alerts, None, "t")
        assert result["events_total"] == 1
        assert result["events_detected"] == 1
        assert result["event_recall"] == pytest.approx(1.0)
        # Drive-day recall sees only 1 of 3 positive rows.
        assert result["recall_drive_day"] == pytest.approx(1 / 3)

    def test_missing_every_row_misses_the_event(self) -> None:
        fold = panel.FoldScores(
            labels=np.array([1, 1, 0], dtype=np.int8),
            drives=np.array([7, 7, 8], dtype=np.int32),
            days=np.array([1, 2, 1], dtype=np.int16),
        )
        result = panel.evaluate(fold, np.array([False, False, True]), None, "t")
        assert result["events_detected"] == 0
        assert result["event_recall"] == pytest.approx(0.0)
        assert result["false_positives"] == 1

    def test_counts_and_rates_are_consistent(self) -> None:
        fold = panel.FoldScores(
            labels=np.array([1, 0, 0, 0], dtype=np.int8),
            drives=np.array([1, 2, 3, 4], dtype=np.int32),
            days=np.array([1, 1, 1, 1], dtype=np.int16),
        )
        result = panel.evaluate(fold, np.array([True, True, False, False]), None, "t")
        assert (result["true_positives"], result["false_positives"]) == (1, 1)
        assert (result["true_negatives"], result["false_negatives"]) == (2, 0)
        assert result["false_alert_rate"] == pytest.approx(1 / 3)
        assert result["false_alerts_per_1000_healthy"] == pytest.approx(1000 / 3)
        assert result["alerts_per_1000_drives_per_day"] == pytest.approx(500.0)


class TestWilsonInterval:
    """The interval used to judge whether selection is supportable."""

    def test_interval_brackets_the_point_estimate(self) -> None:
        low, high = panel.wilson_interval(30, 100)
        assert low < 0.30 < high

    def test_wider_when_there_are_fewer_events(self) -> None:
        narrow = panel.wilson_interval(150, 500)
        wide = panel.wilson_interval(3, 10)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_zero_trials_is_not_a_crash(self) -> None:
        assert panel.wilson_interval(0, 0) == (0.0, 0.0)


class TestArchiveIsNotExtracted:
    """Streaming contract: the archive stays sealed."""

    def test_no_files_appear_beside_the_archive(self, archive: Path, shrunk) -> None:
        calendar, metadata = shrunk
        before = {p for p in archive.parent.rglob("*") if p.is_file()}
        collect(archive, calendar, metadata, [3, 4, 5])
        assert {p for p in archive.parent.rglob("*") if p.is_file()} == before

    def test_members_stream_without_a_temp_copy(self, archive: Path) -> None:
        with open_csv_source(archive) as source:
            assert len(source) == N_DAYS
            with source.open(source.names[0]) as handle:
                assert handle.read(1) != b""
