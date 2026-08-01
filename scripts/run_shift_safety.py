"""Run E3 shift diagnostics and E5 training-only selective prediction."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import EXPECTED_SITES, PREDICTOR_COLUMNS
from src.modeling import Candidate, candidates_from_config, make_pipeline
from src.safety import (
    _fit_sigmoid,
    apply_calibrator,
    class_conditional_thresholds,
    cross_calibrated_oof,
    fit_ood_scores,
    group_oof_probability,
    prediction_sets,
    risk_coverage_curve,
    safety_score,
    selective_metrics,
)

COHORT_PATH = ROOT / "data" / "processed" / "cardioshift_cohort.csv"
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
CORE_RESULTS_PATH = ROOT / "outputs" / "metrics" / "results.json"
CORE_PREDICTIONS_PATH = ROOT / "outputs" / "predictions" / "loho_predictions.csv"
OUTPUTS = ROOT / "outputs"


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {".json", ".csv", ".md", ".py", ".yaml", ".yml"}:
        content = content.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def selected_candidate(record: dict[str, object]) -> Candidate:
    selected = record["selected_candidate"]
    return Candidate(
        family=selected["family"],
        parameters=selected["parameters"],
    )


def candidate_index(candidate: Candidate, all_candidates: list[Candidate]) -> int:
    for index, item in enumerate(all_candidates):
        if item.family == candidate.family and item.parameters == candidate.parameters:
            return index
    raise ValueError(f"selected candidate not found: {candidate}")


def site_predictability(cohort: pd.DataFrame, seed: int) -> dict[str, object]:
    numeric = [column for column in PREDICTOR_COLUMNS if column not in {
        "sex", "cp", "fbs", "restecg", "exang", "slope", "thal"
    }]
    categorical = [column for column in PREDICTOR_COLUMNS if column not in numeric]
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
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
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    predicted = cross_val_predict(
        model,
        cohort[PREDICTOR_COLUMNS],
        cohort["site"],
        cv=splitter,
        method="predict",
        n_jobs=1,
    )
    labels = list(EXPECTED_SITES)
    matrix = confusion_matrix(cohort["site"], predicted, labels=labels)
    recall = np.diag(matrix) / matrix.sum(axis=1)
    return {
        "n": int(len(cohort)),
        "method": "5-fold stratified OOF RandomForest site classifier",
        "balanced_accuracy": float(
            balanced_accuracy_score(cohort["site"], predicted)
        ),
        "chance_balanced_accuracy": 0.25,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "recall_by_site": {
            label: float(value) for label, value in zip(labels, recall)
        },
    }


def standardized_mean_differences(cohort: pd.DataFrame) -> dict[str, object]:
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    report: dict[str, object] = {}
    for site in EXPECTED_SITES:
        site_values: dict[str, object] = {}
        for feature in numeric:
            left = cohort.loc[cohort["site"] == site, feature].dropna().to_numpy()
            right = cohort.loc[cohort["site"] != site, feature].dropna().to_numpy()
            pooled_variance = (
                ((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1))
                / (len(left) + len(right) - 2)
            )
            site_values[feature] = {
                "site_n_nonmissing": int(len(left)),
                "other_n_nonmissing": int(len(right)),
                "smd_site_vs_rest": (
                    float((left.mean() - right.mean()) / np.sqrt(pooled_variance))
                    if pooled_variance > 0
                    else float("nan")
                ),
            }
        report[site] = site_values
    return report


def safety_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    generator = np.random.default_rng(seed)
    metric_names = (
        "coverage",
        "selective_risk",
        "accepted_case_fnr",
        "empirical_conformal_coverage",
        "mean_prediction_set_size",
    )
    values = {name: [] for name in metric_names}
    invalid = {name: 0 for name in metric_names}
    for _ in range(replicates):
        index = generator.integers(0, len(frame), len(frame))
        sampled = frame.iloc[index]
        metrics = selective_metrics(
            sampled["target"].to_numpy(),
            sampled["calibrated_probability"].to_numpy(),
            sampled["accepted"].to_numpy(),
            sampled["conformal_contains_true"].to_numpy(),
            sampled["conformal_set_size"].to_numpy(),
        )
        for name in metric_names:
            value = float(metrics[name])
            if np.isfinite(value):
                values[name].append(value)
            else:
                invalid[name] += 1
    alpha = (1 - confidence_level) / 2
    return {
        "method": "patient-level percentile bootstrap",
        "confidence_level": confidence_level,
        "requested_replicates": replicates,
        "metrics": {
            name: {
                "low": float(np.quantile(samples, alpha)),
                "high": float(np.quantile(samples, 1 - alpha)),
                "valid_replicates": len(samples),
                "invalid_replicates": invalid[name],
            }
            for name, samples in values.items()
        },
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    core = json.loads(CORE_RESULTS_PATH.read_text(encoding="utf-8"))
    core_predictions = pd.read_csv(CORE_PREDICTIONS_PATH)
    cohort = pd.read_csv(COHORT_PATH)
    root_seed = int(config["project"]["root_seed"])
    all_candidates = candidates_from_config(config)
    safety_config = config["safety"]
    bootstrap_config = config["validation"]["bootstrap"]

    shift = {
        "site_predictability": site_predictability(cohort, root_seed + 700_000),
        "standardized_mean_differences": standardized_mean_differences(cohort),
        "missingness_by_site": {
            site: {
                feature: float(value)
                for feature, value in group[PREDICTOR_COLUMNS].isna().mean().items()
            }
            for site, group in cohort.groupby("site", sort=False)
        },
    }

    safety_frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, object]] = []
    for fold_index, record in enumerate(core["E2_leave_one_hospital_out"]["by_site"]):
        held_out_site = record["held_out_site"]
        training = cohort.loc[cohort["site"] != held_out_site].reset_index(drop=True)
        testing = cohort.loc[cohort["site"] == held_out_site].reset_index(drop=True)
        candidate = selected_candidate(record)
        index = candidate_index(candidate, all_candidates)
        base_seed = root_seed + fold_index * 10_000
        raw_oof = group_oof_probability(
            training,
            candidate,
            base_seed + index * 100,
        )
        y_training = training["target"].to_numpy()
        cross_calibrated = cross_calibrated_oof(
            y_training,
            raw_oof,
            base_seed + 800_000,
        )
        final_calibrator = _fit_sigmoid(y_training, raw_oof)
        final_model = make_pipeline(candidate, base_seed + 999_999)
        final_model.fit(training[PREDICTOR_COLUMNS], training["target"])
        raw_test = final_model.predict_proba(testing[PREDICTOR_COLUMNS])[:, 1]
        calibrated_test = apply_calibrator(final_calibrator, raw_test)

        expected = core_predictions.loc[
            core_predictions["site"] == held_out_site,
            "calibrated_probability",
        ].to_numpy()
        if not np.allclose(calibrated_test, expected, atol=1e-12, rtol=1e-10):
            raise AssertionError(f"{held_out_site}: core prediction reproduction failed")

        conformal_thresholds = class_conditional_thresholds(
            y_training,
            cross_calibrated,
            float(safety_config["conformal_alpha"]),
        )
        sets, set_sizes = prediction_sets(calibrated_test, conformal_thresholds)
        contains_true = np.asarray(
            [
                str(target) in label.split(",") if label else False
                for target, label in zip(testing["target"], sets)
            ],
            dtype=bool,
        )

        ood = fit_ood_scores(
            final_model,
            training[PREDICTOR_COLUMNS],
            testing[PREDICTOR_COLUMNS],
            quantile=float(safety_config["ood_quantile"]),
            seed=base_seed + 900_000,
        )
        training_missing = training[PREDICTOR_COLUMNS].isna().mean(axis=1).to_numpy()
        testing_missing = testing[PREDICTOR_COLUMNS].isna().mean(axis=1).to_numpy()
        missing_threshold = float(
            np.quantile(
                training_missing,
                float(safety_config["missing_fraction_quantile"]),
            )
        )
        ambiguity_low, ambiguity_high = safety_config["probability_ambiguity_band"]
        ambiguous = (
            (calibrated_test >= float(ambiguity_low))
            & (calibrated_test <= float(ambiguity_high))
        )
        nonsingleton = set_sizes != 1
        excessive_missingness = testing_missing > missing_threshold
        accepted = ~(
            ood["is_ood"]
            | ambiguous
            | nonsingleton
            | excessive_missingness
        )
        score = safety_score(
            calibrated_test,
            ood["training_score"],
            ood["testing_score"],
            training_missing,
            testing_missing,
            set_sizes,
        )

        frame = testing[["patient_id", "site", "target", "num"]].copy()
        frame["calibrated_probability"] = calibrated_test
        frame["ood_score"] = ood["testing_score"]
        frame["ood_threshold"] = ood["threshold"]
        frame["is_ood"] = ood["is_ood"]
        frame["missing_fraction"] = testing_missing
        frame["missing_threshold"] = missing_threshold
        frame["is_ambiguous"] = ambiguous
        frame["conformal_set"] = sets
        frame["conformal_set_size"] = set_sizes
        frame["conformal_contains_true"] = contains_true
        frame["accepted"] = accepted
        frame["safety_score"] = score
        safety_frames.append(frame)

        metrics = selective_metrics(
            frame["target"].to_numpy(),
            frame["calibrated_probability"].to_numpy(),
            frame["accepted"].to_numpy(),
            frame["conformal_contains_true"].to_numpy(),
            frame["conformal_set_size"].to_numpy(),
        )
        metrics["bootstrap"] = safety_bootstrap(
            frame,
            replicates=int(bootstrap_config["replicates"]),
            seed=base_seed + 950_000,
            confidence_level=float(bootstrap_config["confidence_level"]),
        )
        fold_reports.append(
            {
                "held_out_site": held_out_site,
                "n_training": int(len(training)),
                "selected_candidate": {
                    "family": candidate.family,
                    "parameters": candidate.parameters,
                },
                "thresholds": {
                    "probability_ambiguity_band": [
                        float(ambiguity_low),
                        float(ambiguity_high),
                    ],
                    "ood_quantile": float(safety_config["ood_quantile"]),
                    "ood_threshold": float(ood["threshold"]),
                    "ood_threshold_source": ood["threshold_source"],
                    "missing_fraction_quantile": float(
                        safety_config["missing_fraction_quantile"]
                    ),
                    "missing_fraction_threshold": missing_threshold,
                    "conformal_alpha": float(safety_config["conformal_alpha"]),
                    "conformal_class_0": conformal_thresholds[0],
                    "conformal_class_1": conformal_thresholds[1],
                },
                "metrics": metrics,
                "risk_coverage_curve": risk_coverage_curve(
                    frame["target"].to_numpy(),
                    frame["calibrated_probability"].to_numpy(),
                    frame["safety_score"].to_numpy(),
                ),
            }
        )

    safety_predictions = pd.concat(safety_frames, ignore_index=True)
    if not safety_predictions["patient_id"].is_unique:
        raise AssertionError("safety predictions do not form one LOHO universe")
    pooled_metrics = selective_metrics(
        safety_predictions["target"].to_numpy(),
        safety_predictions["calibrated_probability"].to_numpy(),
        safety_predictions["accepted"].to_numpy(),
        safety_predictions["conformal_contains_true"].to_numpy(),
        safety_predictions["conformal_set_size"].to_numpy(),
    )
    pooled_metrics["bootstrap"] = safety_bootstrap(
        safety_predictions,
        replicates=int(bootstrap_config["replicates"]),
        seed=root_seed + 990_000,
        confidence_level=float(bootstrap_config["confidence_level"]),
    )
    pooled_curve = risk_coverage_curve(
        safety_predictions["target"].to_numpy(),
        safety_predictions["calibrated_probability"].to_numpy(),
        safety_predictions["safety_score"].to_numpy(),
    )

    prediction_dir = OUTPUTS / "predictions"
    metrics_dir = OUTPUTS / "metrics"
    audit_dir = OUTPUTS / "audit"
    for directory in (prediction_dir, metrics_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / "safety_loho_predictions.csv"
    prediction_path.write_text(
        safety_predictions.to_csv(index=False),
        encoding="utf-8",
    )
    result = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "shift_safety_complete",
        "reproducibility": {
            "input_sha256": raw_sha256(COHORT_PATH),
            "core_results_sha256": canonical_sha256(CORE_RESULTS_PATH),
            "core_predictions_reproduced_exactly": True,
            "root_seed": root_seed,
        },
        "E3_shift": shift,
        "E5_safety": {
            "pooled": pooled_metrics,
            "by_site": fold_reports,
            "risk_coverage_curve_pooled": pooled_curve,
            "conformal_claim": (
                "Empirical coverage only. Exchangeability does not hold "
                "automatically under hospital shift."
            ),
            "ood_limitation": (
                "The IsolationForest threshold is an outer-training in-sample "
                "quantile and may under-detect subtle shift."
            ),
        },
    }
    result_path = metrics_dir / "shift_safety_results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "shift_balanced_accuracy": shift["site_predictability"][
                    "balanced_accuracy"
                ],
                "safety_pooled": pooled_metrics,
                "result_path": str(result_path),
                "prediction_path": str(prediction_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
