"""Leakage-resistant model selection and outer-fold prediction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import PREDICTOR_COLUMNS, assert_prediction_features

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


@dataclass(frozen=True)
class Candidate:
    family: str
    parameters: dict[str, Any]

    @property
    def label(self) -> str:
        parameters = ",".join(f"{key}={self.parameters[key]}" for key in sorted(self.parameters))
        return f"{self.family}({parameters})"


def candidates_from_config(config: dict[str, Any]) -> list[Candidate]:
    models = config["models"]
    candidates: list[Candidate] = []
    lr = models["logistic_regression"]
    for c_value, class_weight in product(lr["C"], lr["class_weight"]):
        candidates.append(
            Candidate(
                "logistic_regression",
                {
                    "C": c_value,
                    "class_weight": class_weight,
                    "max_iter": lr["max_iter"],
                },
            )
        )
    rf = models["random_forest"]
    for n_estimators, max_depth, min_samples_leaf, class_weight in product(
        rf["n_estimators"],
        rf["max_depth"],
        rf["min_samples_leaf"],
        rf["class_weight"],
    ):
        candidates.append(
            Candidate(
                "random_forest",
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "class_weight": class_weight,
                },
            )
        )
    hgb = models["hist_gradient_boosting"]
    for learning_rate, max_leaf_nodes, l2_regularization in product(
        hgb["learning_rate"],
        hgb["max_leaf_nodes"],
        hgb["l2_regularization"],
    ):
        candidates.append(
            Candidate(
                "hist_gradient_boosting",
                {
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                    "l2_regularization": l2_regularization,
                },
            )
        )
    return candidates


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, BaseEstimator]] = [
        (
            "impute",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="most_frequent",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_pipeline(candidate: Candidate, seed: int) -> Pipeline:
    assert_prediction_features(PREDICTOR_COLUMNS)
    if candidate.family == "logistic_regression":
        estimator: BaseEstimator = LogisticRegression(
            solver="liblinear",
            random_state=seed,
            **candidate.parameters,
        )
        scale_numeric = True
    elif candidate.family == "random_forest":
        estimator = RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **candidate.parameters,
        )
        scale_numeric = False
    elif candidate.family == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            random_state=seed,
            **candidate.parameters,
        )
        scale_numeric = False
    else:
        raise ValueError(f"unknown model family: {candidate.family}")
    return Pipeline(
        [
            ("preprocess", _preprocessor(scale_numeric)),
            ("model", estimator),
        ]
    )


def _fit_calibrator(y_true: np.ndarray, raw_probability: np.ndarray) -> LogisticRegression:
    probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    calibrator.fit(logit, np.asarray(y_true, dtype=int))
    return calibrator


def apply_calibrator(
    calibrator: LogisticRegression,
    raw_probability: np.ndarray,
) -> np.ndarray:
    probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def _candidate_score(
    frame: pd.DataFrame,
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
    candidate: Candidate,
    seed: int,
) -> dict[str, Any]:
    probabilities = np.full(len(frame), np.nan, dtype=float)
    fold_brier: list[float] = []
    fold_auroc: list[float] = []
    for fold_index, (train_index, validation_index) in enumerate(folds):
        model = make_pipeline(candidate, seed + fold_index)
        model.fit(
            frame.iloc[train_index][PREDICTOR_COLUMNS],
            frame.iloc[train_index]["target"],
        )
        probability = model.predict_proba(
            frame.iloc[validation_index][PREDICTOR_COLUMNS]
        )[:, 1]
        probabilities[validation_index] = probability
        y_validation = frame.iloc[validation_index]["target"].to_numpy()
        fold_brier.append(float(brier_score_loss(y_validation, probability)))
        fold_auroc.append(float(roc_auc_score(y_validation, probability)))
    if np.isnan(probabilities).any():
        raise RuntimeError("inner validation did not produce one probability per row")
    return {
        "candidate": candidate,
        "mean_brier": float(np.mean(fold_brier)),
        "mean_auroc": float(np.mean(fold_auroc)),
        "oof_probability": probabilities,
        "fold_brier": fold_brier,
        "fold_auroc": fold_auroc,
    }


def select_with_group_validation(
    training: pd.DataFrame,
    candidates: list[Candidate],
    seed: int,
) -> dict[str, Any]:
    splitter = LeaveOneGroupOut()
    folds = list(
        splitter.split(
            training,
            training["target"],
            groups=training["site"],
        )
    )
    scores = [
        _candidate_score(training, folds, candidate, seed + index * 100)
        for index, candidate in enumerate(candidates)
    ]
    scores.sort(
        key=lambda item: (
            item["mean_brier"],
            -item["mean_auroc"],
            item["candidate"].family != "logistic_regression",
            item["candidate"].label,
        )
    )
    return {"selected": scores[0], "scores": scores}


def fit_predict_loho_fold(
    cohort: pd.DataFrame,
    held_out_site: str,
    candidates: list[Candidate],
    seed: int,
) -> dict[str, Any]:
    training = cohort.loc[cohort["site"] != held_out_site].reset_index(drop=True)
    testing = cohort.loc[cohort["site"] == held_out_site].reset_index(drop=True)
    if testing.empty:
        raise ValueError(f"held-out site not found: {held_out_site}")
    training_ids = set(training["patient_id"])
    testing_ids = set(testing["patient_id"])
    if training_ids.intersection(testing_ids):
        raise AssertionError("outer training and test patient IDs overlap")

    selection = select_with_group_validation(training, candidates, seed)
    selected = selection["selected"]
    calibrator = _fit_calibrator(
        training["target"].to_numpy(),
        selected["oof_probability"],
    )
    final_model = make_pipeline(selected["candidate"], seed + 999_999)
    final_model.fit(training[PREDICTOR_COLUMNS], training["target"])
    raw_probability = final_model.predict_proba(testing[PREDICTOR_COLUMNS])[:, 1]
    calibrated_probability = apply_calibrator(calibrator, raw_probability)

    ledger = {
        "held_out_site": held_out_site,
        "outer_train_sites": sorted(training["site"].unique().tolist()),
        "outer_test_sites": sorted(testing["site"].unique().tolist()),
        "outer_train_patient_ids": sorted(training_ids),
        "outer_test_patient_ids": sorted(testing_ids),
        "tuning_patient_ids": sorted(training_ids),
        "calibration_patient_ids": sorted(training_ids),
        "final_fit_patient_ids": sorted(training_ids),
    }
    for stage in ("tuning", "calibration", "final_fit"):
        stage_ids = set(ledger[f"{stage}_patient_ids"])
        if stage_ids.intersection(testing_ids):
            raise AssertionError(f"held-out patients entered {stage}")

    predictions = testing[["patient_id", "site", "target", "num"]].copy()
    predictions["raw_probability"] = raw_probability
    predictions["calibrated_probability"] = calibrated_probability
    return {
        "predictions": predictions,
        "ledger": ledger,
        "selected_candidate": {
            "family": selected["candidate"].family,
            "parameters": selected["candidate"].parameters,
            "inner_mean_brier": selected["mean_brier"],
            "inner_mean_auroc": selected["mean_auroc"],
        },
        "selection_table": [
            {
                "family": score["candidate"].family,
                "parameters": score["candidate"].parameters,
                "mean_brier": score["mean_brier"],
                "mean_auroc": score["mean_auroc"],
            }
            for score in selection["scores"]
        ],
    }


def fit_predict_random_split(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    candidates: list[Candidate],
    seed: int,
) -> dict[str, Any]:
    if set(training["patient_id"]).intersection(testing["patient_id"]):
        raise AssertionError("random training and test patient IDs overlap")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(training, training["target"]))
    scores = [
        _candidate_score(training, folds, candidate, seed + index * 100)
        for index, candidate in enumerate(candidates)
    ]
    scores.sort(
        key=lambda item: (
            item["mean_brier"],
            -item["mean_auroc"],
            item["candidate"].family != "logistic_regression",
            item["candidate"].label,
        )
    )
    selected = scores[0]
    calibrator = _fit_calibrator(
        training["target"].to_numpy(),
        selected["oof_probability"],
    )
    model = make_pipeline(selected["candidate"], seed + 999_999)
    model.fit(training[PREDICTOR_COLUMNS], training["target"])
    raw_probability = model.predict_proba(testing[PREDICTOR_COLUMNS])[:, 1]
    calibrated_probability = apply_calibrator(calibrator, raw_probability)
    predictions = testing[["patient_id", "site", "target", "num"]].copy()
    predictions["raw_probability"] = raw_probability
    predictions["calibrated_probability"] = calibrated_probability
    return {
        "predictions": predictions,
        "selected_candidate": {
            "family": selected["candidate"].family,
            "parameters": selected["candidate"].parameters,
            "inner_mean_brier": selected["mean_brier"],
            "inner_mean_auroc": selected["mean_auroc"],
        },
    }
