"""Model training for 7-day drive-failure prediction.

Scaffold: interfaces are defined, function bodies raise
:class:`NotImplementedError`.

Trains logistic regression, random forest and XGBoost on the design matrix
from ``src.features``. The split is chronological — drive-days from one drive
are autocorrelated, so a random split leaks. Class imbalance is handled by
weighting the training fold; the test fold keeps its natural failure rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data_pipeline import DataValidationError
from src.features import FeatureMatrix

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from sklearn.base import BaseEstimator

LOGGER = logging.getLogger(__name__)

#: Estimators this module trains, in reporting order.
MODEL_NAMES: Tuple[str, ...] = ("logistic_regression", "random_forest", "xgboost")


@dataclass(frozen=True)
class TrainConfig:
    """Settings for a training run.

    Attributes:
        test_start_date: First date of the held-out period. Every observation
            on or after it goes to test. Takes precedence over
            ``test_fraction`` when set.
        test_fraction: Used when ``test_start_date`` is ``None``.
        models: Which estimators to train.
        random_state: Seed for the estimators that use one.
        n_jobs: Worker count for the estimators that support it (``-1`` = all
            cores).
        artifacts_dir: Destination for fitted models and reports.
        calibrate: Wrap each estimator in probability calibration. Tree
            scores rank well but are poorly calibrated, and thresholds are
            chosen on a probability scale.
    """

    test_start_date: Optional[pd.Timestamp] = None
    test_fraction: float = 0.2
    models: Tuple[str, ...] = MODEL_NAMES
    random_state: int = 42
    n_jobs: int = -1
    artifacts_dir: Path = Path("artifacts")
    calibrate: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            DataValidationError: If the test fraction is outside ``(0, 1)`` or
                an unknown model name is requested.
        """
        if not 0.0 < self.test_fraction < 1.0:
            raise DataValidationError(
                f"test_fraction must be in (0, 1), got {self.test_fraction!r}"
            )
        unknown = [name for name in self.models if name not in MODEL_NAMES]
        if unknown:
            raise DataValidationError(
                f"Unknown model name(s) {unknown}; choose from {list(MODEL_NAMES)}"
            )
        if not self.models:
            raise DataValidationError("At least one model must be selected")


@dataclass
class ChronologicalSplit:
    """A train/test split made on time, with the counts needed to audit it.

    Attributes:
        X_train: Training features.
        y_train: Training labels.
        X_test: Held-out features.
        y_test: Held-out labels.
        train_dates: Observation dates for the training rows.
        test_dates: Observation dates for the test rows.
        cut_date: First date belonging to the test set.
        drives_in_both: Drives appearing on both sides of the cut. Non-zero is
            expected; their rows never overlap in time.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_dates: pd.Series
    test_dates: pd.Series
    cut_date: pd.Timestamp
    drives_in_both: int = 0

    @property
    def train_positive_rate(self) -> float:
        """Return the positive-label share of the training fold."""
        return float(self.y_train.mean()) if len(self.y_train) else 0.0

    @property
    def test_positive_rate(self) -> float:
        """Return the positive-label share of the test fold."""
        return float(self.y_test.mean()) if len(self.y_test) else 0.0

    @property
    def scale_pos_weight(self) -> float:
        """Return negatives per positive in the training fold only."""
        positives = int(self.y_train.sum())
        return (len(self.y_train) - positives) / positives if positives else float("inf")


@dataclass
class TrainedModel:
    """A fitted estimator with its provenance.

    Attributes:
        name: One of :data:`MODEL_NAMES`.
        estimator: The fitted estimator or pipeline.
        feature_names: Column order the estimator was fitted on. Inference must
            present features in this order.
        train_rows: Number of training observations.
        train_positives: Number of positive labels in training.
        cut_date: Split boundary.
        fit_seconds: Wall-clock fit time.
        params: Hyperparameters actually used.
    """

    name: str
    estimator: "BaseEstimator"
    feature_names: List[str]
    train_rows: int
    train_positives: int
    cut_date: pd.Timestamp
    fit_seconds: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def resolve_cut_date(
    dates: pd.Series,
    test_start_date: Optional[pd.Timestamp] = None,
    test_fraction: float = 0.2,
) -> pd.Timestamp:
    """Determine the first date belonging to the test set.

    Args:
        dates: Observation dates across the whole dataset.
        test_start_date: Explicit boundary; returned as-is when supplied.
        test_fraction: Share of the date range to hold out.

    Returns:
        The first date of the test period.

    Raises:
        DataValidationError: If ``dates`` is empty, or the resolved boundary
            leaves one side of the split with no rows.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def chronological_split(
    matrix: FeatureMatrix, config: Optional[TrainConfig] = None
) -> ChronologicalSplit:
    """Split a design matrix on time.

    Observations before ``cut_date`` train; everything on or after is held out.

    Args:
        matrix: Design matrix from ``src.features.build_feature_matrix``.
        config: Training configuration; defaults to :class:`TrainConfig`.

    Returns:
        The split.

    Raises:
        DataValidationError: If the matrix lacks a ``date`` column in its
            metadata, or either fold ends up empty or without a positive label.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def build_logistic_regression(config: TrainConfig) -> "BaseEstimator":
    """Build the logistic-regression baseline.

    A ``Pipeline`` of median imputation, standard scaling and
    ``LogisticRegression(class_weight="balanced")``. Preprocessing sits inside
    the pipeline so it is fitted on the training fold only.

    Args:
        config: Training configuration.

    Returns:
        An unfitted estimator.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def build_random_forest(config: TrainConfig) -> "BaseEstimator":
    """Build the random-forest classifier.

    Scaling is unnecessary for trees; imputation stays because scikit-learn's
    forest rejects NaN.

    Args:
        config: Training configuration.

    Returns:
        An unfitted estimator with ``class_weight="balanced_subsample"``.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def build_xgboost(config: TrainConfig, scale_pos_weight: float) -> "BaseEstimator":
    """Build the gradient-boosted classifier.

    No imputation: XGBoost learns a default direction for missing values.

    Args:
        config: Training configuration.
        scale_pos_weight: Negatives per positive, from the training fold only.

    Returns:
        An unfitted estimator.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def build_estimator(
    name: str, config: TrainConfig, scale_pos_weight: float
) -> "BaseEstimator":
    """Dispatch to the builder for ``name``.

    Args:
        name: One of :data:`MODEL_NAMES`.
        config: Training configuration.
        scale_pos_weight: Negatives per positive in the training fold.

    Returns:
        An unfitted estimator.

    Raises:
        DataValidationError: If ``name`` is not a known model.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


# ---------------------------------------------------------------------------
# Fitting and persistence
# ---------------------------------------------------------------------------


def train_model(
    name: str, split: ChronologicalSplit, config: Optional[TrainConfig] = None
) -> TrainedModel:
    """Fit one estimator on the training fold.

    Args:
        name: One of :data:`MODEL_NAMES`.
        split: Chronological split.
        config: Training configuration; defaults to :class:`TrainConfig`.

    Returns:
        The fitted model with its provenance.

    Raises:
        DataValidationError: If the training fold has no positive labels, which
            would make the fit meaningless.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def train_all(
    matrix: FeatureMatrix, config: Optional[TrainConfig] = None
) -> Dict[str, TrainedModel]:
    """Fit every configured estimator on one shared split.

    Args:
        matrix: Design matrix from ``src.features``.
        config: Training configuration; defaults to :class:`TrainConfig`.

    Returns:
        Fitted models keyed by name.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def save_model(model: TrainedModel, artifacts_dir: Path) -> Path:
    """Persist a fitted model and its metadata.

    The feature order and cut date are written alongside the estimator: a model
    served without its exact feature order silently scores garbage.

    Args:
        model: Fitted model.
        artifacts_dir: Destination directory, created if absent.

    Returns:
        Path to the written ``.joblib`` file.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def load_model(path: Path) -> TrainedModel:
    """Load a model previously written by :func:`save_model`.

    Args:
        path: Path to a ``.joblib`` artifact.

    Returns:
        The reconstructed model with its metadata.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def main(argv: Optional[List[str]] = None) -> int:
    """Train from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
