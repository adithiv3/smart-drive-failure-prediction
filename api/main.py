"""FastAPI inference endpoint for 7-day drive-failure prediction.

Partial scaffold: the application, configuration and request/response schemas
are functional and the service boots. ``/health`` and ``/model`` report what is
actually loaded. The scoring routes return 501 until ``src.train`` can produce
a model artifact; they do not return placeholder probabilities, which a caller
could not distinguish from real ones.

``/predict`` takes a drive's recent daily SMART observations rather than
precomputed features, and derives features server-side with the same
``src.features`` code used in training. Otherwise every client would
reimplement the 30-day window arithmetic, and any divergence would cause
training/serving skew that is invisible in the response.

Run with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.data_pipeline import SMART_RAW_COLUMNS
from src.features import FeatureConfig

LOGGER = logging.getLogger(__name__)

API_VERSION = "0.1.0"

#: Minimum daily observations required; fewer gives too little history.
MIN_OBSERVATIONS = 7


def _artifacts_dir() -> Path:
    """Return the artifacts directory from the environment."""
    return Path(os.getenv("SDFP_ARTIFACTS_DIR", "artifacts"))


def _model_path() -> Path:
    """Return the configured model artifact path."""
    return _artifacts_dir() / os.getenv("SDFP_MODEL_FILE", "model.joblib")


def _alert_threshold() -> float:
    """Return the configured alert threshold."""
    return float(os.getenv("SDFP_ALERT_THRESHOLD", "0.5"))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SmartObservation(BaseModel):
    """One drive-day of SMART telemetry.

    Attributes:
        date: Observation date.
        smart: Raw SMART attribute values keyed by column name, e.g.
            ``{"smart_5_raw": 0, "smart_197_raw": 8}``. Attributes the drive
            does not report may be omitted and are treated as missing.
    """

    date: date
    smart: Dict[str, Optional[float]] = Field(default_factory=dict)

    @field_validator("smart")
    @classmethod
    def reject_unknown_attributes(
        cls, value: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        """Reject SMART keys the model was never trained on.

        Args:
            value: Submitted SMART mapping.

        Returns:
            The mapping, unchanged, if every key is known.

        Raises:
            ValueError: If any key is outside :data:`SMART_RAW_COLUMNS`.
        """
        unknown = sorted(set(value) - set(SMART_RAW_COLUMNS))
        if unknown:
            raise ValueError(
                f"Unknown SMART attribute(s) {unknown}. Supported: "
                f"{list(SMART_RAW_COLUMNS)}"
            )
        return value


class PredictionRequest(BaseModel):
    """A scoring request for one drive.

    Attributes:
        serial_number: Drive identifier, echoed back in the response.
        model: Drive model string, e.g. ``"ST4000DM000"``.
        capacity_bytes: Drive capacity; ``None`` if unknown.
        observations: Daily readings, most recent last. At least
            :data:`MIN_OBSERVATIONS` are required, and the last one is the day
            being scored.
        threshold: Optional per-request override of the alert threshold.
    """

    serial_number: str = Field(min_length=1, max_length=64)
    model: Optional[str] = Field(default=None, max_length=64)
    capacity_bytes: Optional[float] = Field(default=None, gt=0)
    observations: List[SmartObservation] = Field(min_length=MIN_OBSERVATIONS)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("observations")
    @classmethod
    def reject_duplicate_dates(
        cls, value: List[SmartObservation]
    ) -> List[SmartObservation]:
        """Reject repeated observation dates.

        Trailing windows assume one reading per drive per day.

        Args:
            value: Submitted observations.

        Returns:
            The observations, unchanged, if all dates are distinct.

        Raises:
            ValueError: If any date appears more than once.
        """
        dates = [observation.date for observation in value]
        if len(set(dates)) != len(dates):
            raise ValueError("observations contain duplicate dates")
        return value


class PredictionResponse(BaseModel):
    """A scored drive-day.

    Attributes:
        serial_number: Echoed drive identifier.
        as_of_date: Date scored, the latest submitted observation.
        failure_probability: Probability the drive fails within the horizon.
        alert: Whether the probability met the threshold.
        threshold: Threshold applied.
        horizon_days: Forward window the probability refers to.
        model_name: Estimator that produced the score.
        observations_used: Submitted readings falling inside the window.
        features_missing: Features that were NaN for this request.
    """

    serial_number: str
    as_of_date: date
    failure_probability: float = Field(ge=0.0, le=1.0)
    alert: bool
    threshold: float
    horizon_days: int
    model_name: str
    observations_used: int
    features_missing: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service liveness and readiness.

    Attributes:
        status: ``"ok"`` when the service is up.
        api_version: Version of this service.
        model_loaded: Whether a model artifact is loaded and ready to score.
        model_path: Configured artifact path.
        detail: Human-readable note, e.g. why no model is loaded.
    """

    status: str
    api_version: str
    model_loaded: bool
    model_path: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SMART Drive Failure Prediction",
    version=API_VERSION,
    description=(
        "Scores a hard drive's probability of failing within the next 7 days "
        "from its recent daily SMART telemetry."
    ),
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Report service liveness and whether a model is loaded.

    Returns:
        The current health state.
    """
    path = _model_path()
    exists = path.is_file()
    return HealthResponse(
        status="ok",
        api_version=API_VERSION,
        model_loaded=False,
        model_path=str(path),
        detail=(
            "Model artifact found but the loader is not implemented yet; "
            "scoring routes return 501."
            if exists
            else f"No model artifact at {path}. Train one with `python -m src.train`."
        ),
    )


@app.get("/model", tags=["ops"])
def model_info() -> Dict[str, object]:
    """Describe the loaded model: name, training period, feature order.

    Returns:
        Model metadata.

    Raises:
        HTTPException: 501 until model persistence lands in ``src.train``.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Model metadata is unavailable until src.train.save_model is implemented.",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one drive from its recent daily SMART observations.

    Request validation is active; only the scoring step is missing.

    Args:
        request: The drive's identity and recent telemetry.

    Returns:
        The scored drive-day.

    Raises:
        HTTPException: 501 until a trained model artifact can be loaded.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Scoring is not implemented yet. Implement src.train first.",
    )


@app.post("/predict/batch", tags=["inference"])
def predict_batch(requests: List[PredictionRequest]) -> List[PredictionResponse]:
    """Score several drives in one call.

    Args:
        requests: One entry per drive.

    Returns:
        One response per request, in the same order.

    Raises:
        HTTPException: 501 until a trained model artifact can be loaded.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Scoring is not implemented yet. Implement src.train first.",
    )


def _feature_config() -> FeatureConfig:
    """Build the feature configuration from the environment.

    Returns:
        The configured :class:`~src.features.FeatureConfig`.
    """
    return FeatureConfig(
        window_days=int(os.getenv("SDFP_WINDOW_DAYS", "30")),
        min_history_days=int(os.getenv("SDFP_MIN_HISTORY_DAYS", "7")),
    )
