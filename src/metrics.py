"""Binary prediction metrics and patient-level bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    """Fixed-width ECE; empty bins contribute zero."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(result)


def calibration_intercept_slope(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    """Fit y ~ logit(p) for descriptive calibration intercept and slope."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def evaluate_binary(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "auprc": (
            float(average_precision_score(y, p))
            if len(np.unique(y)) == 2
            else float("nan")
        ),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10": expected_calibration_error(y, p, bins=10),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "sensitivity": _safe_ratio(int(tp), int(tp + fn)),
        "specificity": _safe_ratio(int(tn), int(tn + fp)),
        "ppv": _safe_ratio(int(tp), int(tp + fp)),
        "npv": _safe_ratio(int(tn), int(tn + fn)),
        "threshold": float(threshold),
    }


BOOTSTRAP_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": lambda y, p: float(roc_auc_score(y, p)),
    "auprc": lambda y, p: float(average_precision_score(y, p)),
    "brier": lambda y, p: float(brier_score_loss(y, p)),
    "log_loss": lambda y, p: float(log_loss(y, p, labels=[0, 1])),
    "ece_10": lambda y, p: expected_calibration_error(y, p, bins=10),
}


def bootstrap_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Patient-level percentile intervals with invalid-resample accounting."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    generator = np.random.default_rng(seed)
    values = {name: [] for name in BOOTSTRAP_METRICS}
    invalid = 0
    for _ in range(replicates):
        indices = generator.integers(0, len(y), size=len(y))
        sampled_y = y[indices]
        sampled_p = p[indices]
        if len(np.unique(sampled_y)) < 2:
            invalid += 1
            continue
        for name, function in BOOTSTRAP_METRICS.items():
            values[name].append(function(sampled_y, sampled_p))

    alpha = (1 - confidence_level) / 2
    intervals: dict[str, object] = {}
    for name, samples in values.items():
        array = np.asarray(samples, dtype=float)
        intervals[name] = {
            "low": float(np.quantile(array, alpha)),
            "high": float(np.quantile(array, 1 - alpha)),
            "valid_replicates": int(len(array)),
        }
    return {
        "method": "patient-level percentile bootstrap",
        "confidence_level": confidence_level,
        "requested_replicates": replicates,
        "invalid_single_class_replicates": invalid,
        "metrics": intervals,
    }
