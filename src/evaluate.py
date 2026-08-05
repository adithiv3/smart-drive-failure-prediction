"""Evaluation, threshold selection and SHAP explanations.

Scaffold: interfaces and metric definitions are fixed, function bodies raise
:class:`NotImplementedError`.

Accuracy is not reported: with a positive class well under 1%, predicting
"healthy" everywhere scores above 99%. Reported instead are PR-AUC, recall,
precision, the confusion matrix, false-alert rate (``FP / (FP + TN)``) and
false alerts per 1,000 truly healthy drive-days.

The per-1,000 denominator is truly-healthy drive-days, not rows the model
called healthy; the latter moves with the threshold and cannot compare
thresholds. Reports include the no-skill PR baseline, which for average
precision is the positive rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_pipeline import DataValidationError

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from src.train import ChronologicalSplit, TrainedModel

# Re-exported so callers can catch it without importing the pipeline.
__all__ = [
    "DEFAULT_THRESHOLD",
    "ClassificationReport",
    "DataValidationError",
    "ThresholdSweep",
    "compare_models",
    "evaluate_model",
    "evaluate_predictions",
    "explain_model",
    "save_reports",
    "sweep_thresholds",
]

LOGGER = logging.getLogger(__name__)

#: Placeholder default; pick the real value from the precision-recall curve.
DEFAULT_THRESHOLD: float = 0.5


@dataclass
class ClassificationReport:
    """Performance of one model at one threshold.

    Attributes:
        model_name: Model these numbers describe.
        threshold: Probability at or above which a row was flagged.
        n_samples: Rows scored.
        n_positives: True positive-class rows in the test fold.
        true_negatives: TN count.
        false_positives: FP count.
        false_negatives: FN count.
        true_positives: TP count.
        pr_auc: Average precision (area under the precision-recall curve).
        roc_auc: Reported for completeness.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1: Harmonic mean of precision and recall.
        brier_score: Calibration of the raw probabilities.
        evaluated_period: ``(first_date, last_date)`` of the scored rows.
    """

    model_name: str
    threshold: float
    n_samples: int
    n_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    evaluated_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    @property
    def base_rate(self) -> float:
        """Return the test-fold positive rate, the no-skill PR-AUC baseline."""
        return self.n_positives / self.n_samples if self.n_samples else 0.0

    @property
    def pr_auc_lift(self) -> float:
        """Return PR-AUC as a multiple of the no-skill baseline."""
        base = self.base_rate
        return self.pr_auc / base if base else float("nan")

    @property
    def false_alert_rate(self) -> float:
        """Return FP / (FP + TN): the share of healthy drive-days flagged."""
        healthy = self.false_positives + self.true_negatives
        return self.false_positives / healthy if healthy else 0.0

    @property
    def false_alerts_per_1000_healthy(self) -> float:
        """Return false alerts per 1,000 truly healthy drive-days."""
        return self.false_alert_rate * 1000.0

    @property
    def confusion_matrix(self) -> "np.ndarray":
        """Return the 2x2 matrix as ``[[TN, FP], [FN, TP]]``."""
        return np.array(
            [
                [self.true_negatives, self.false_positives],
                [self.false_negatives, self.true_positives],
            ]
        )

    def as_lines(self) -> List[str]:
        """Render the report as printable lines."""
        raise NotImplementedError("Not yet implemented — see module docstring.")

    def as_dict(self) -> Dict[str, object]:
        """Return a flat mapping suitable for a comparison table or JSON."""
        raise NotImplementedError("Not yet implemented — see module docstring.")


@dataclass
class ThresholdSweep:
    """Model behaviour across candidate thresholds.

    Attributes:
        model_name: Model swept.
        table: One row per threshold, with precision, recall, false-alert rate
            and false alerts per 1,000 healthy drive-days.
    """

    model_name: str
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def threshold_for_alert_budget(self, max_false_alerts_per_1000: float) -> float:
        """Return the lowest threshold meeting a false-alert budget.

        Args:
            max_false_alerts_per_1000: Tolerated false alerts per 1,000 healthy
                drive-days.

        Returns:
            The chosen threshold.

        Raises:
            DataValidationError: If no swept threshold meets the budget.
        """
        raise NotImplementedError("Not yet implemented — see module docstring.")

    def threshold_for_recall(self, min_recall: float) -> float:
        """Return the highest threshold still achieving ``min_recall``.

        Args:
            min_recall: Required recall.

        Returns:
            The chosen threshold.

        Raises:
            DataValidationError: If no swept threshold reaches ``min_recall``.
        """
        raise NotImplementedError("Not yet implemented — see module docstring.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: Sequence[int],
    y_score: Sequence[float],
    model_name: str,
    threshold: float = DEFAULT_THRESHOLD,
    dates: Optional[pd.Series] = None,
) -> ClassificationReport:
    """Score held-out predictions.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted probabilities of the positive class.
        model_name: Name recorded in the report.
        threshold: Probability at or above which a row is flagged.
        dates: Optional observation dates, recorded as the evaluated period.

    Returns:
        A :class:`ClassificationReport`.

    Raises:
        DataValidationError: If the inputs differ in length, the threshold or
            scores fall outside ``[0, 1]``, or ``y_true`` has no positive label.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def evaluate_model(
    model: "TrainedModel",
    split: "ChronologicalSplit",
    threshold: float = DEFAULT_THRESHOLD,
) -> ClassificationReport:
    """Score a fitted model on the held-out fold.

    Args:
        model: Fitted model.
        split: The split it was trained on; only the test fold is scored.
        threshold: Operating threshold.

    Returns:
        A :class:`ClassificationReport`.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def sweep_thresholds(
    y_true: Sequence[int],
    y_score: Sequence[float],
    model_name: str,
    thresholds: Optional[Sequence[float]] = None,
) -> ThresholdSweep:
    """Recompute the operating metrics across candidate thresholds.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted probabilities.
        model_name: Name recorded in the sweep.
        thresholds: Candidates; defaults to a grid spanning ``[0, 1]``.

    Returns:
        A :class:`ThresholdSweep`.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def compare_models(reports: Sequence[ClassificationReport]) -> pd.DataFrame:
    """Assemble a side-by-side comparison table.

    Args:
        reports: Reports produced at the same threshold on the same test fold.

    Returns:
        One row per model, sorted by PR-AUC descending.

    Raises:
        DataValidationError: If the reports came from different thresholds or
            different-sized test folds.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------


def explain_model(
    model: "TrainedModel",
    X: pd.DataFrame,
    max_samples: int = 5000,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute SHAP values and rank features by mean absolute contribution.

    ``TreeExplainer`` for the tree models, ``LinearExplainer`` for the logistic
    pipeline. A sample is used because SHAP on the full fold is expensive.

    Args:
        model: Fitted model.
        X: Rows to explain, in the model's feature order.
        max_samples: Rows sampled when ``X`` is larger.
        output_dir: If given, beeswarm and bar plots are written here.

    Returns:
        Features ranked by mean absolute SHAP value.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def save_reports(
    reports: Sequence[ClassificationReport],
    output_dir: Path,
    sweeps: Optional[Sequence[ThresholdSweep]] = None,
) -> List[Path]:
    """Write reports, sweeps and the comparison table to disk.

    Args:
        reports: Reports to persist.
        output_dir: Destination, created if absent.
        sweeps: Optional threshold sweeps to persist alongside.

    Returns:
        Paths written.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


def main(argv: Optional[List[str]] = None) -> int:
    """Evaluate saved models from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    raise NotImplementedError("Not yet implemented — see module docstring.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
