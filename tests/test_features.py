"""Tests for trailing-window feature construction.

Expected values are computed with numpy over the fixture's known readings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import LABEL_COLUMN, LEAKAGE_COLUMNS, DataValidationError
from src.features import (
    FeatureConfig,
    FeatureMatrix,
    build_feature_matrix,
    build_features,
)
from tests.conftest import FIXTURE_SMART_COLUMNS

#: SN_A's smart_5_raw readings for 01-01..01-14: ten zeros, then 1, 2, 3, 4.
SN_A_SMART5_TO_JAN14 = np.array([0.0] * 10 + [1.0, 2.0, 3.0, 4.0])


@pytest.fixture(scope="module")
def sn_a_jan14(featured: pd.DataFrame) -> pd.Series:
    """Return SN_A's feature row for 2024-01-14, the last day it is scored."""
    return featured[featured["serial_number"] == "SN_A"].set_index("date").loc[
        "2024-01-14"
    ]


class TestRollingStatistics:
    """Window statistics against independently computed values."""

    def test_mean(self, sn_a_jan14: pd.Series) -> None:
        assert sn_a_jan14["smart_5_raw_mean_30d"] == pytest.approx(
            SN_A_SMART5_TO_JAN14.mean()
        )

    def test_max(self, sn_a_jan14: pd.Series) -> None:
        assert sn_a_jan14["smart_5_raw_max_30d"] == pytest.approx(
            SN_A_SMART5_TO_JAN14.max()
        )

    def test_std_is_the_sample_standard_deviation(self, sn_a_jan14: pd.Series) -> None:
        # ddof=1, matching pandas' rolling default.
        assert sn_a_jan14["smart_5_raw_std_30d"] == pytest.approx(
            SN_A_SMART5_TO_JAN14.std(ddof=1)
        )

    def test_last_is_the_reading_on_the_day_itself(self, sn_a_jan14: pd.Series) -> None:
        assert sn_a_jan14["smart_5_raw_last_30d"] == pytest.approx(4.0)

    def test_change_is_last_minus_first_in_window(self, sn_a_jan14: pd.Series) -> None:
        # 4.0 on 01-14 minus 0.0 on 01-01.
        assert sn_a_jan14["smart_5_raw_change_30d"] == pytest.approx(4.0)

    def test_observation_count(self, sn_a_jan14: pd.Series) -> None:
        assert sn_a_jan14["observations_30d"] == pytest.approx(14.0)

    def test_history_days(self, sn_a_jan14: pd.Series) -> None:
        # First seen 01-01, scored 01-14.
        assert sn_a_jan14["history_days"] == pytest.approx(13.0)


class TestStaticFeatures:
    """Derived per-row features."""

    def test_power_on_days_converts_from_hours(self, sn_a_jan14: pd.Series) -> None:
        # smart_9_raw on day index 13 is 1000 + 24*13 hours.
        assert sn_a_jan14["power_on_days"] == pytest.approx((1000.0 + 24 * 13) / 24)

    def test_capacity_is_rescaled_to_gigabytes(self, sn_a_jan14: pd.Series) -> None:
        assert sn_a_jan14["capacity_gb"] == pytest.approx(4000.0)

    def test_unknown_capacity_stays_missing(self, labelled: pd.DataFrame) -> None:
        # SN_C's capacity sentinel became NaN and must stay missing.
        config = FeatureConfig(
            window_days=30, smart_columns=FIXTURE_SMART_COLUMNS, min_history_days=0
        )
        frame, _ = build_features(labelled, config)
        sn_c = frame[frame["serial_number"] == "SN_C"]
        assert sn_c["capacity_gb"].isna().all()


class TestWindowBoundary:
    """The window is trailing, half-open and measured in time, not rows."""

    def test_window_is_half_open_on_the_left(self, labelled: pd.DataFrame) -> None:
        # A 5-day window at 01-13 covers (01-08, 01-13] = 01-09..01-13, whose
        # smart_5_raw readings are 0, 0, 1, 2, 3. If the left edge were
        # inclusive, 01-08 would join and the mean would drop.
        config = FeatureConfig(
            window_days=5, smart_columns=FIXTURE_SMART_COLUMNS, min_history_days=4
        )
        frame, _ = build_features(labelled, config)
        row = frame[frame["serial_number"] == "SN_A"].set_index("date").loc["2024-01-13"]

        expected = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
        assert row["observations_5d"] == pytest.approx(5.0)
        assert row["smart_5_raw_mean_5d"] == pytest.approx(expected.mean())
        assert row["smart_5_raw_max_5d"] == pytest.approx(3.0)
        assert row["smart_5_raw_change_5d"] == pytest.approx(3.0)

    def test_window_counts_days_not_rows(self, labelled: pd.DataFrame) -> None:
        # A row-based window would reach further back and find five; a
        # time-based one finds two.
        gapped = labelled[labelled["serial_number"] == "SN_B"]
        gapped = gapped[
            ~gapped["date"].isin(
                pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"])
            )
        ]
        config = FeatureConfig(
            window_days=5, smart_columns=FIXTURE_SMART_COLUMNS, min_history_days=4
        )
        frame, _ = build_features(gapped, config)
        row = frame.set_index("date").loc["2024-01-09"]
        assert row["observations_5d"] == pytest.approx(2.0)

    def test_features_never_see_the_future(self, labelled: pd.DataFrame) -> None:
        # Truncating after a date must not change features computed before it.
        config = FeatureConfig(
            window_days=30, smart_columns=FIXTURE_SMART_COLUMNS, min_history_days=7
        )
        full, _ = build_features(labelled, config)
        truncated, _ = build_features(
            labelled[labelled["date"] <= pd.Timestamp("2024-01-12")], config
        )

        key = ["serial_number", "date"]
        columns = [c for c in truncated.columns if c.endswith("_30d")]
        merged = truncated[key + columns].merge(
            full[key + columns], on=key, suffixes=("_trunc", "_full")
        )
        assert len(merged) > 0
        for column in columns:
            pd.testing.assert_series_equal(
                merged[f"{column}_trunc"],
                merged[f"{column}_full"],
                check_names=False,
            )


class TestFiltering:
    """Rows and features removed, and why."""

    def test_short_history_rows_are_dropped(self, featured: pd.DataFrame) -> None:
        # min_history_days=7 keeps SN_A 01-08..01-14 (7 rows) and SN_B
        # 01-08..01-13 (6). SN_C never accumulates 7 days before it is
        # censored away, so it contributes nothing.
        assert len(featured) == 13
        assert set(featured["serial_number"]) == {"SN_A", "SN_B"}
        assert (featured["history_days"] >= 7).all()

    def test_positive_labels_survive_feature_construction(
        self, featured: pd.DataFrame
    ) -> None:
        assert int(featured[LABEL_COLUMN].sum()) == 7

    def test_all_nan_features_are_dropped(self, labelled: pd.DataFrame) -> None:
        config = FeatureConfig(
            window_days=30, smart_columns=FIXTURE_SMART_COLUMNS, min_history_days=7
        )
        _, summary = build_features(labelled, config)
        # smart_187_raw is absent from the first five files, so the earliest
        # in-window reading is missing and _change_30d is NaN throughout.
        assert "smart_187_raw_change_30d" in summary.dropped_features

    def test_failing_drives_are_counted_not_failure_rows(
        self, labelled: pd.DataFrame
    ) -> None:
        # failure=1 rows are dropped, so counting them would report zero.
        config = FeatureConfig(
            window_days=30, smart_columns=FIXTURE_SMART_COLUMNS, min_history_days=7
        )
        _, summary = build_features(labelled, config)
        assert summary.failing_drives == 1


class TestLeakageGuard:
    """Outcome-derived columns must not reach the design matrix."""

    def test_matrix_excludes_every_leakage_column(self, featured: pd.DataFrame) -> None:
        matrix = build_feature_matrix(featured)
        assert not set(matrix.X.columns) & LEAKAGE_COLUMNS

    def test_injecting_a_leakage_column_raises(self, featured: pd.DataFrame) -> None:
        matrix = build_feature_matrix(featured)
        with pytest.raises(DataValidationError, match="Outcome-derived column"):
            FeatureMatrix(
                X=matrix.X.assign(failure=1),
                y=matrix.y,
                metadata=matrix.metadata,
                feature_names=list(matrix.X.columns) + ["failure"],
            )

    def test_misaligned_lengths_raise(self, featured: pd.DataFrame) -> None:
        matrix = build_feature_matrix(featured)
        with pytest.raises(DataValidationError, match="Misaligned lengths"):
            FeatureMatrix(
                X=matrix.X.iloc[:-1],
                y=matrix.y,
                metadata=matrix.metadata,
                feature_names=matrix.feature_names,
            )

    def test_matrix_is_entirely_numeric(self, featured: pd.DataFrame) -> None:
        matrix = build_feature_matrix(featured)
        assert all(np.issubdtype(dtype, np.number) for dtype in matrix.X.dtypes)

    def test_imbalance_ratio(self, featured: pd.DataFrame) -> None:
        # 13 rows, 7 positive -> 6 negatives per 7 positives.
        matrix = build_feature_matrix(featured)
        assert matrix.positive_count == 7
        assert matrix.imbalance_ratio == pytest.approx(6 / 7)


class TestConfigValidation:
    """Configuration errors surface at construction."""

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"window_days": 1}, "window_days"),
            ({"min_periods": 0}, "min_periods"),
            ({"min_history_days": -1}, "min_history_days"),
            ({"min_history_days": 30, "window_days": 30}, "must be less than"),
            ({"smart_columns": ()}, "smart_columns"),
        ],
    )
    def test_invalid_config_raises(self, kwargs, match: str) -> None:
        with pytest.raises(DataValidationError, match=match):
            FeatureConfig(**kwargs)

    def test_feature_names_track_the_window_length(self) -> None:
        names = FeatureConfig(window_days=14).feature_names(["smart_5_raw"])
        assert "smart_5_raw_mean_14d" in names
        assert "observations_14d" in names
        assert not any(name.endswith("_30d") for name in names)


class TestInputValidation:
    """Bad input frames are rejected with an actionable message."""

    def test_empty_frame_raises(self, labelled: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="empty frame"):
            build_features(labelled.iloc[0:0])

    def test_missing_label_column_raises(self, labelled: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="missing required column"):
            build_features(labelled.drop(columns=[LABEL_COLUMN]))

    def test_duplicate_drive_days_raise(self, labelled: pd.DataFrame) -> None:
        doubled = pd.concat([labelled, labelled.iloc[:1]], ignore_index=True)
        with pytest.raises(DataValidationError, match="duplicate"):
            build_features(doubled)

    def test_unknown_smart_columns_are_skipped_not_fatal(
        self, labelled: pd.DataFrame
    ) -> None:
        config = FeatureConfig(
            window_days=30,
            smart_columns=FIXTURE_SMART_COLUMNS + ("smart_999_raw",),
            min_history_days=7,
        )
        frame, _ = build_features(labelled, config)
        assert not any(c.startswith("smart_999_raw") for c in frame.columns)

    def test_all_smart_columns_absent_raises(self, labelled: pd.DataFrame) -> None:
        config = FeatureConfig(window_days=30, smart_columns=("smart_999_raw",))
        with pytest.raises(DataValidationError, match="None of the requested"):
            build_features(labelled, config)
