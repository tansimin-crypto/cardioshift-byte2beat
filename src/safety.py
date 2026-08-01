"""Training-only safety signals for selective prediction under site shift."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from src.data import PREDICTOR_COLUMNS
from src.modeling import Candidate, apply_calibrator, make_pipeline


def _fit_sigmoid(y_true: np.ndarray, raw_probability: np.ndarray) -> LogisticRegression:
    probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit, np.asarray(y_true, dtype=int))
    return model


def group_oof_probability(
    training: pd.DataFrame,
    candidate: Candidate,
    seed: int,
) -> np.ndarray:
    splitter = LeaveOneGroupOut()
    probability = np.full(len(training), np.nan, dtype=float)
    for fold_index, (fit_index, validation_index) in enumerate(
        splitter.split(training, training["target"], groups=training["site"])
    ):
        model = make_pipeline(candidate, seed + fold_index)
        model.fit(
            training.iloc[fit_index][PREDICTOR_COLUMNS],
            training.iloc[fit_index]["target"],
        )
        probability[validation_index] = model.predict_proba(
            training.iloc[validation_index][PREDICTOR_COLUMNS]
        )[:, 1]
    if np.isnan(probability).any():
        raise RuntimeError("group OOF probability is incomplete")
    return probability


def cross_calibrated_oof(
    y_true: np.ndarray,
    raw_probability: np.ndarray,
    seed: int,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    raw = np.asarray(raw_probability, dtype=float)
    calibrated = np.full(len(y), np.nan, dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_index, validation_index in splitter.split(raw, y):
        calibrator = _fit_sigmoid(y[fit_index], raw[fit_index])
        calibrated[validation_index] = apply_calibrator(
            calibrator,
            raw[validation_index],
        )
    if np.isnan(calibrated).any():
        raise RuntimeError("cross-calibrated OOF probability is incomplete")
    return calibrated


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    rank = int(np.ceil((len(values) + 1) * (1 - alpha))) - 1
    rank = min(max(rank, 0), len(values) - 1)
    return float(values[rank])


def class_conditional_thresholds(
    y_true: np.ndarray,
    calibrated_probability: np.ndarray,
    alpha: float,
) -> dict[int, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(calibrated_probability, dtype=float)
    score_for_true_class = np.where(y == 1, 1 - p, p)
    return {
        class_value: conformal_quantile(
            score_for_true_class[y == class_value],
            alpha,
        )
        for class_value in (0, 1)
    }


def prediction_sets(
    calibrated_probability: np.ndarray,
    thresholds: dict[int, float],
) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    sizes: list[int] = []
    for probability in np.asarray(calibrated_probability, dtype=float):
        members: list[str] = []
        if probability <= thresholds[0]:
            members.append("0")
        if 1 - probability <= thresholds[1]:
            members.append("1")
        labels.append(",".join(members))
        sizes.append(len(members))
    return labels, np.asarray(sizes, dtype=int)


def fit_ood_scores(
    fitted_prediction_model: Any,
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
    *,
    quantile: float,
    seed: int,
) -> dict[str, object]:
    preprocessor = fitted_prediction_model.named_steps["preprocess"]
    transformed_training = preprocessor.transform(training_features)
    transformed_testing = preprocessor.transform(testing_features)
    detector = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    detector.fit(transformed_training)
    training_score = -detector.score_samples(transformed_training)
    testing_score = -detector.score_samples(transformed_testing)
    threshold = float(np.quantile(training_score, quantile))
    return {
        "training_score": training_score,
        "testing_score": testing_score,
        "threshold": threshold,
        "is_ood": testing_score > threshold,
        "method": "IsolationForest on training-fitted prediction representation",
        "threshold_source": "outer-training in-sample score quantile",
    }


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(sorted_reference, values, side="right") / len(sorted_reference)


def safety_score(
    probability: np.ndarray,
    ood_training_score: np.ndarray,
    ood_testing_score: np.ndarray,
    training_missing_fraction: np.ndarray,
    testing_missing_fraction: np.ndarray,
    set_size: np.ndarray,
) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    ambiguity = 1 - 2 * np.abs(p - 0.5)
    ood_percentile = empirical_percentile(ood_training_score, ood_testing_score)
    missing_percentile = empirical_percentile(
        training_missing_fraction,
        testing_missing_fraction,
    )
    nonsingleton = (np.asarray(set_size) != 1).astype(float)
    return np.maximum.reduce(
        [ambiguity, ood_percentile, missing_percentile, nonsingleton]
    )


def selective_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    accepted: np.ndarray,
    conformal_contains_true: np.ndarray,
    conformal_set_size: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    accept = np.asarray(accepted, dtype=bool)
    prediction = (p >= 0.5).astype(int)
    accepted_n = int(accept.sum())
    accepted_positive = accept & (y == 1)
    false_negative = accepted_positive & (prediction == 0)
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "accepted_n": accepted_n,
        "coverage": float(accept.mean()),
        "deferral_rate": float(1 - accept.mean()),
        "selective_risk": (
            float((prediction[accept] != y[accept]).mean())
            if accepted_n
            else float("nan")
        ),
        "accepted_case_fnr": (
            float(false_negative.sum() / accepted_positive.sum())
            if accepted_positive.any()
            else float("nan")
        ),
        "empirical_conformal_coverage": float(
            np.asarray(conformal_contains_true, dtype=bool).mean()
        ),
        "mean_prediction_set_size": float(
            np.asarray(conformal_set_size, dtype=float).mean()
        ),
    }


def risk_coverage_curve(
    y_true: np.ndarray,
    probability: np.ndarray,
    score: np.ndarray,
) -> list[dict[str, float | int]]:
    y = np.asarray(y_true, dtype=int)
    prediction = (np.asarray(probability, dtype=float) >= 0.5).astype(int)
    order = np.argsort(np.asarray(score, dtype=float), kind="stable")
    points: list[dict[str, float | int]] = []
    for requested_coverage in np.linspace(0.1, 1.0, 19):
        accepted_n = max(1, int(np.floor(len(y) * requested_coverage)))
        accepted_index = order[:accepted_n]
        accepted_y = y[accepted_index]
        accepted_prediction = prediction[accepted_index]
        positive = accepted_y == 1
        points.append(
            {
                "requested_coverage": float(requested_coverage),
                "accepted_n": accepted_n,
                "empirical_coverage": float(accepted_n / len(y)),
                "selective_risk": float(
                    (accepted_prediction != accepted_y).mean()
                ),
                "accepted_case_fnr": (
                    float(
                        ((accepted_prediction == 0) & positive).sum()
                        / positive.sum()
                    )
                    if positive.any()
                    else float("nan")
                ),
            }
        )
    return points
