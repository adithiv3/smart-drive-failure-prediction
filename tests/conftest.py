"""Shared fixtures.

The fixture fleet is synthetic, not Backblaze data, so the labelling and window
arithmetic can be checked against hand-computed numbers.

The fleet spans 2024-01-01 to 2024-01-20 and contains three drives chosen to
exercise the three cases that matter:

======  =====================================  ============================
drive   lifetime                               what it tests
======  =====================================  ============================
SN_A    01-01 .. 01-15, ``failure=1`` on 01-15  the positive label window
SN_B    01-01 .. 01-20, never fails             right-censoring at the end
                                                of the dataset
SN_C    01-01 .. 01-10, leaves without failing  right-censoring on fleet
                                                departure
======  =====================================  ============================

``smart_187_raw`` is omitted from the first five files, reproducing the schema
drift the real archives contain.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pytest

from src.data_pipeline import PipelineConfig, run_pipeline
from src.features import FeatureConfig, build_features

#: SMART columns present in the fixture. A deliberate subset: the pipeline must
#: cope with being asked for columns the data does not have.
FIXTURE_SMART_COLUMNS: Tuple[str, ...] = (
    "smart_5_raw",
    "smart_9_raw",
    "smart_187_raw",
    "smart_197_raw",
    "smart_194_raw",
)

FIRST_DAY = pd.Timestamp("2024-01-01")
LAST_DAY = pd.Timestamp("2024-01-20")

#: Day index (0-based) on which SN_A records ``failure=1``.
SN_A_FAILURE_INDEX = 14
#: Day index after which SN_C stops reporting.
SN_C_LAST_INDEX = 9
#: Files missing ``smart_187_raw``.
FILES_WITHOUT_SMART_187 = 5


@pytest.fixture(scope="session", autouse=True)
def _quiet_logging() -> None:
    """Silence pipeline INFO chatter so test output stays readable."""
    logging.getLogger("src").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the synthetic daily CSV files and return their directory.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.

    Returns:
        Directory containing one CSV per day.
    """
    directory = tmp_path_factory.mktemp("raw")
    dates = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")

    for index, day in enumerate(dates):
        rows: List[Dict[str, object]] = []

        if index <= SN_A_FAILURE_INDEX:
            # smart_5_raw holds at 0 for ten days, then climbs 1..5 — a
            # reallocation ramp ending in failure.
            rows.append(
                {
                    "date": day,
                    "serial_number": "SN_A",
                    "model": "ST4000DM000",
                    "capacity_bytes": 4e12,
                    "failure": 1 if index == SN_A_FAILURE_INDEX else 0,
                    "smart_5_raw": 0.0 if index < 10 else float(index - 9),
                    "smart_9_raw": 1000.0 + 24 * index,
                    "smart_187_raw": 0.0,
                    "smart_197_raw": 0.0,
                    "smart_194_raw": 28.0,
                }
            )

        rows.append(
            {
                "date": day,
                "serial_number": "SN_B",
                "model": "ST4000DM000",
                "capacity_bytes": 4e12,
                "failure": 0,
                "smart_5_raw": 0.0,
                "smart_9_raw": 5000.0 + 24 * index,
                "smart_187_raw": 0.0,
                "smart_197_raw": 0.0,
                "smart_194_raw": 30.0,
            }
        )

        if index <= SN_C_LAST_INDEX:
            rows.append(
                {
                    "date": day,
                    "serial_number": "SN_C",
                    "model": "HGST HMS5C4040",
                    "capacity_bytes": -1,  # Backblaze's "unknown" sentinel
                    "failure": 0,
                    "smart_5_raw": 0.0,
                    "smart_9_raw": 200.0 + 24 * index,
                    "smart_187_raw": 0.0,
                    "smart_197_raw": 0.0,
                    "smart_194_raw": 25.0,
                }
            )

        frame = pd.DataFrame(rows)
        if index < FILES_WITHOUT_SMART_187:
            frame = frame.drop(columns=["smart_187_raw"])
        frame.to_csv(directory / f"{day.date()}.csv", index=False)

    return directory


@pytest.fixture(scope="session")
def pipeline_config(raw_dir: Path) -> PipelineConfig:
    """Return a pipeline configuration pointed at the fixture fleet."""
    return PipelineConfig(
        input_dir=raw_dir, horizon_days=7, smart_columns=FIXTURE_SMART_COLUMNS
    )


@pytest.fixture(scope="session")
def labelled(pipeline_config: PipelineConfig) -> pd.DataFrame:
    """Return the labelled fixture dataset."""
    frame, _ = run_pipeline(pipeline_config)
    return frame


@pytest.fixture(scope="session")
def feature_config() -> FeatureConfig:
    """Return a feature configuration matching the fixture fleet."""
    return FeatureConfig(
        window_days=30, smart_columns=FIXTURE_SMART_COLUMNS, min_history_days=7
    )


@pytest.fixture(scope="session")
def featured(labelled: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    """Return the fixture dataset with trailing-window features attached."""
    frame, _ = build_features(labelled, feature_config)
    return frame
