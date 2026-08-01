"""Run prespecified E6 missingness/subgroup analyses and E7 runtime profiling."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import pickle
import platform
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import EXPECTED_SITES, PREDICTOR_COLUMNS
from src.metrics import expected_calibration_error
from src.modeling import Candidate, apply_calibrator, candidates_from_config, make_pipeline
from src.safety import _fit_sigmoid, group_oof_probability, prediction_sets

COHORT_PATH = ROOT / "data" / "processed" / "cardioshift_cohort.csv"
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
CORE_PATH = ROOT / "outputs" / "metrics" / "results.json"
CORE_PREDICTIONS_PATH = ROOT / "outputs" / "predictions" / "loho_predictions.csv"
SAFETY_PATH = ROOT / "outputs" / "metrics" / "shift_safety_results.json"
SAFETY_PREDICTIONS_PATH = (
    ROOT / "outputs" / "predictions" / "safety_loho_predictions.csv"
)
FROZEN_PATH = ROOT / "evidence" / "frozen_e1_e5.json"
RESULT_PATH = ROOT / "outputs" / "metrics" / "robustness_results.json"
PREDICTION_PATH = ROOT / "outputs" / "predictions" / "robustness_predictions.csv"
MODEL_DIR = ROOT / "outputs" / "models"


def sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {".json", ".csv", ".md", ".py", ".yaml", ".yml"}:
        content = content.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def candidate_from_record(record: dict[str, Any]) -> Candidate:
    selected = record["selected_candidate"]
    return Candidate(selected["family"], selected["parameters"])


def candidate_index(candidate: Candidate, candidates: list[Candidate]) -> int:
    for index, item in enumerate(candidates):
        if item.family == candidate.family and item.parameters == candidate.parameters:
            return index
    raise ValueError(f"frozen candidate is outside accepted search space: {candidate}")


def metric_set(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["target"].to_numpy(dtype=int)
    p = frame["calibrated_probability"].to_numpy(dtype=float)
    prediction = (p >= 0.5).astype(int)
    accepted = frame["accepted"].to_numpy(dtype=bool)
    accepted_positive = accepted & (y == 1)
    unique = np.unique(y)
    return {
        "n": int(len(frame)),
        "events": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)) if len(unique) == 2 else None,
        "auprc": float(average_precision_score(y, p)) if len(unique) == 2 else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10": float(expected_calibration_error(y, p, bins=10)),
        "full_coverage_error": float((prediction != y).mean()),
        "accepted_n": int(accepted.sum()),
        "safety_coverage": float(accepted.mean()),
        "selective_risk": (
            float((prediction[accepted] != y[accepted]).mean())
            if accepted.any()
            else None
        ),
        "accepted_case_fnr": (
            float(((prediction == 0) & accepted_positive).sum() / accepted_positive.sum())
            if accepted_positive.any()
            else None
        ),
        "empirical_conformal_coverage": float(
            frame["conformal_contains_true"].mean()
        ),
        "sensitivity": (
            float((prediction[y == 1] == 1).mean()) if (y == 1).any() else None
        ),
        "specificity": (
            float((prediction[y == 0] == 0).mean()) if (y == 0).any() else None
        ),
    }


def bootstrap_metrics(
    frame: pd.DataFrame,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    names = (
        "auroc",
        "auprc",
        "brier",
        "full_coverage_error",
        "safety_coverage",
        "selective_risk",
        "accepted_case_fnr",
        "empirical_conformal_coverage",
        "sensitivity",
        "specificity",
    )
    samples: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(replicates):
        sampled = frame.iloc[rng.integers(0, len(frame), len(frame))]
        metrics = metric_set(sampled)
        for name in names:
            value = metrics[name]
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
    return {
        "method": "patient-level percentile bootstrap",
        "replicates": replicates,
        "metrics": {
            name: {
                "low": float(np.quantile(values, 0.025)) if values else None,
                "high": float(np.quantile(values, 0.975)) if values else None,
                "valid_replicates": len(values),
            }
            for name, values in samples.items()
        },
    }


def mask_mcar(
    features: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    copied = features.copy(deep=True)
    rng = np.random.default_rng(seed)
    mask = rng.random(copied.shape) < fraction
    copied.iloc[:, :] = copied.mask(mask)
    row_masks = [
        ";".join(column for column, selected in zip(copied.columns, row) if selected)
        for row in mask
    ]
    return copied, row_masks


def mask_grouped(
    features: pd.DataFrame,
    *,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    copied = features.copy(deep=True)
    copied.loc[:, columns] = np.nan
    label = ";".join(columns)
    return copied, [label] * len(copied)


def peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    return int(counters.PeakWorkingSetSize) if ok else None


def predict_bundle(bundle: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    raw = bundle["model"].predict_proba(features)[:, 1]
    return apply_calibrator(bundle["calibrator"], raw)


def runtime_profile(
    model_paths: dict[str, Path],
    samples: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    warmups = 5
    repeats = 30
    cold_repeats = 5
    records: dict[str, Any] = {}
    for site, path in model_paths.items():
        cold_ms: list[float] = []
        for _ in range(cold_repeats):
            start = time.perf_counter_ns()
            with path.open("rb") as handle:
                pickle.load(handle)
            cold_ms.append((time.perf_counter_ns() - start) / 1e6)
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        one = samples[site].iloc[[0]]
        hundred = pd.concat([one] * 100, ignore_index=True)
        for _ in range(warmups):
            predict_bundle(bundle, one)
            predict_bundle(bundle, hundred)
        tracemalloc.start()
        timings: dict[str, list[float]] = {"batch_1_ms": [], "batch_100_ms": []}
        for label, frame in (("batch_1_ms", one), ("batch_100_ms", hundred)):
            for _ in range(repeats):
                start = time.perf_counter_ns()
                predict_bundle(bundle, frame)
                timings[label].append((time.perf_counter_ns() - start) / 1e6)
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        records[site] = {
            "serialized_model_bytes": path.stat().st_size,
            "serialized_model_sha256": sha256(path),
            "cold_load_ms": {
                "median": float(np.median(cold_ms)),
                "min": float(np.min(cold_ms)),
                "max": float(np.max(cold_ms)),
            },
            **{
                label: {
                    "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)),
                }
                for label, values in timings.items()
            },
            "peak_process_memory_bytes": peak_working_set_bytes(),
            "peak_python_traced_allocation_bytes": int(python_peak),
        }
    return {
        "claim_scope": "research runtime profile; not medical-device performance",
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "scikit_learn": sklearn.__version__,
        },
        "warmup_repeats": warmups,
        "measurement_repeats": repeats,
        "cold_load_repeats": cold_repeats,
        "by_fold": records,
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    core = json.loads(CORE_PATH.read_text(encoding="utf-8"))
    safety = json.loads(SAFETY_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    cohort = pd.read_csv(COHORT_PATH)
    core_predictions = pd.read_csv(CORE_PREDICTIONS_PATH)
    safety_predictions = pd.read_csv(SAFETY_PREDICTIONS_PATH)
    root_seed = int(config["project"]["root_seed"])
    candidates = candidates_from_config(config)
    mcar_repeats = int(config["stress_tests"]["mcar_repeats"])
    bootstrap_replicates = int(config["stress_tests"]["bootstrap_replicates"])

    prediction_frames: list[pd.DataFrame] = []
    bundles: dict[str, dict[str, Any]] = {}
    model_paths: dict[str, Path] = {}
    samples: dict[str, pd.DataFrame] = {}
    fold_metadata: dict[str, Any] = {}

    for fold_index, record in enumerate(core["E2_leave_one_hospital_out"]["by_site"]):
        site = record["held_out_site"]
        training = cohort.loc[cohort["site"] != site].reset_index(drop=True)
        testing = cohort.loc[cohort["site"] == site].reset_index(drop=True)
        candidate = candidate_from_record(record)
        index = candidate_index(candidate, candidates)
        base_seed = root_seed + fold_index * 10_000
        raw_oof = group_oof_probability(training, candidate, base_seed + index * 100)
        calibrator = _fit_sigmoid(training["target"].to_numpy(), raw_oof)
        model = make_pipeline(candidate, base_seed + 999_999)
        model.fit(training[PREDICTOR_COLUMNS], training["target"])

        transformed_training = model.named_steps["preprocess"].transform(
            training[PREDICTOR_COLUMNS]
        )
        detector = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=base_seed + 900_000,
            n_jobs=-1,
        )
        detector.fit(transformed_training)
        training_ood_score = -detector.score_samples(transformed_training)

        safety_record = next(
            item for item in safety["E5_safety"]["by_site"] if item["held_out_site"] == site
        )
        thresholds = safety_record["thresholds"]
        ood_threshold = float(thresholds["ood_threshold"])
        reproduced_threshold = float(
            np.quantile(training_ood_score, float(thresholds["ood_quantile"]))
        )
        if not np.isclose(ood_threshold, reproduced_threshold, atol=1e-12, rtol=1e-10):
            raise AssertionError(f"{site}: frozen OOD threshold did not reproduce")

        bundle = {
            "site": site,
            "candidate": candidate,
            "model": model,
            "calibrator": calibrator,
            "ood_detector": detector,
            "thresholds": thresholds,
        }
        bundles[site] = bundle
        samples[site] = testing[PREDICTOR_COLUMNS].copy()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"{site.lower().replace(' ', '_')}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        model_paths[site] = model_path

        expected = core_predictions.loc[
            core_predictions["site"] == site, "calibrated_probability"
        ].to_numpy()
        baseline = predict_bundle(bundle, testing[PREDICTOR_COLUMNS])
        if not np.allclose(baseline, expected, atol=1e-12, rtol=1e-10):
            raise AssertionError(f"{site}: accepted calibrated probabilities changed")
        fold_metadata[site] = {
            "candidate": {
                "family": candidate.family,
                "parameters": candidate.parameters,
            },
            "selection_source": "accepted E2 selected_candidate",
            "ood_threshold_source": thresholds["ood_threshold_source"],
            "thresholds": thresholds,
            "outer_test_patient_ids_sha256": hashlib.sha256(
                "\n".join(testing["patient_id"]).encode("utf-8")
            ).hexdigest(),
        }

        scenarios: list[tuple[str, str, int, int | None, pd.DataFrame, list[str]]] = [
            (
                "baseline",
                "baseline",
                0,
                None,
                testing[PREDICTOR_COLUMNS].copy(),
                [""] * len(testing),
            )
        ]
        for fraction in config["stress_tests"]["mcar_fractions"]:
            for repeat in range(mcar_repeats):
                seed = root_seed + 1_000_000 + fold_index * 100_000 + int(fraction * 1000) * 100 + repeat
                masked, row_masks = mask_mcar(
                    testing[PREDICTOR_COLUMNS],
                    fraction=float(fraction),
                    seed=seed,
                )
                scenarios.append(
                    (
                        "mcar",
                        f"mcar_{int(float(fraction) * 100)}",
                        repeat,
                        seed,
                        masked,
                        row_masks,
                    )
                )
        grouped = config["stress_tests"]["grouped"]
        for name, configured in grouped.items():
            columns = (
                [column for column in PREDICTOR_COLUMNS if column not in configured]
                if name == "basic_vitals_only"
                else list(configured)
            )
            masked, row_masks = mask_grouped(
                testing[PREDICTOR_COLUMNS],
                columns=columns,
            )
            scenarios.append(("grouped", name, 0, None, masked, row_masks))

        for scenario_type, scenario, repeat, mask_seed, features, row_masks in scenarios:
            probability = predict_bundle(bundle, features)
            transformed = model.named_steps["preprocess"].transform(features)
            ood_score = -detector.score_samples(transformed)
            conformal_thresholds = {
                0: float(thresholds["conformal_class_0"]),
                1: float(thresholds["conformal_class_1"]),
            }
            sets, sizes = prediction_sets(probability, conformal_thresholds)
            contains = np.asarray(
                [
                    str(target) in labels.split(",") if labels else False
                    for target, labels in zip(testing["target"], sets)
                ],
                dtype=bool,
            )
            missing_fraction = features.isna().mean(axis=1).to_numpy()
            low, high = thresholds["probability_ambiguity_band"]
            ambiguous = (probability >= float(low)) & (probability <= float(high))
            is_ood = ood_score > ood_threshold
            excessive_missing = missing_fraction > float(
                thresholds["missing_fraction_threshold"]
            )
            accepted = ~(is_ood | ambiguous | (sizes != 1) | excessive_missing)
            frame = testing[["patient_id", "site", "target"]].copy()
            frame["scenario_type"] = scenario_type
            frame["scenario"] = scenario
            frame["repeat"] = repeat
            frame["mask_seed"] = mask_seed
            frame["newly_masked_features"] = row_masks
            frame["calibrated_probability"] = probability
            frame["prediction"] = (probability >= 0.5).astype(int)
            frame["ood_score"] = ood_score
            frame["is_ood"] = is_ood
            frame["missing_fraction"] = missing_fraction
            frame["is_ambiguous"] = ambiguous
            frame["is_excessive_missingness"] = excessive_missing
            frame["conformal_set"] = sets
            frame["conformal_set_size"] = sizes
            frame["conformal_contains_true"] = contains
            frame["accepted"] = accepted
            prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTION_PATH, index=False)

    scenario_results: dict[str, Any] = {}
    for (scenario_type, scenario, repeat), frame in predictions.groupby(
        ["scenario_type", "scenario", "repeat"], sort=False
    ):
        key = f"{scenario_type}:{scenario}:repeat={repeat}"
        scenario_results[key] = {
            "scenario_type": scenario_type,
            "scenario": scenario,
            "repeat": int(repeat),
            "pooled": metric_set(frame),
            "pooled_bootstrap": bootstrap_metrics(
                frame,
                seed=root_seed + 2_000_000 + len(scenario_results),
                replicates=bootstrap_replicates,
            ),
            "by_site": {
                site: metric_set(site_frame)
                for site, site_frame in frame.groupby("site", sort=False)
            },
        }

    subgroup_source = safety_predictions.merge(
        cohort[["patient_id", "age", "sex"]],
        on="patient_id",
        how="left",
        validate="one_to_one",
    )
    subgroup_source["scenario_type"] = "subgroup"
    subgroup_source["scenario"] = "accepted_loho"
    subgroup_source["repeat"] = 0
    subgroup_source["is_excessive_missingness"] = (
        subgroup_source["missing_fraction"] > subgroup_source["missing_threshold"]
    )
    subgroup_source["prediction"] = (
        subgroup_source["calibrated_probability"] >= 0.5
    ).astype(int)
    subgroup_source["newly_masked_features"] = ""
    subgroup_source["mask_seed"] = np.nan
    subgroup_predictions = subgroup_source[
        predictions.columns.intersection(subgroup_source.columns)
    ].copy()

    subgroup_definitions: dict[str, pd.Series] = {
        "sex_0": subgroup_source["sex"] == 0,
        "sex_1": subgroup_source["sex"] == 1,
        "age_lt_50": subgroup_source["age"] < 50,
        "age_50_59": subgroup_source["age"].between(50, 59, inclusive="both"),
        "age_60_69": subgroup_source["age"].between(60, 69, inclusive="both"),
        "age_ge_70": subgroup_source["age"] >= 70,
    }
    subgroup_results: dict[str, Any] = {}
    for offset, (name, selector) in enumerate(subgroup_definitions.items()):
        frame = subgroup_source.loc[selector].copy()
        subgroup_results[name] = {
            "definition": name,
            "descriptive_only": True,
            "small_or_single_class_reason": (
                "fewer than 30 records"
                if len(frame) < 30
                else (
                    "single outcome class"
                    if frame["target"].nunique() < 2
                    else None
                )
            ),
            "metrics": metric_set(frame),
            "bootstrap": bootstrap_metrics(
                frame,
                seed=root_seed + 3_000_000 + offset,
                replicates=bootstrap_replicates,
            ),
        }

    runtime = runtime_profile(model_paths, samples)
    result = {
        "schema_version": "1.0",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model_selection": "none; accepted E2 candidate reused per LOHO fold",
            "perturbation_scope": "copied outer-test features only",
            "mcar_fractions": config["stress_tests"]["mcar_fractions"],
            "mcar_repeats": mcar_repeats,
            "mcar_seed_formula": (
                "root_seed + 1_000_000 + fold_index*100_000 "
                "+ int(fraction*1000)*100 + repeat"
            ),
            "grouped": config["stress_tests"]["grouped"],
            "bootstrap_replicates": bootstrap_replicates,
            "threshold_source_sha256": sha256(SAFETY_PATH),
        },
        "frozen_e1_e5_manifest_sha256": sha256(FROZEN_PATH),
        "fold_metadata": fold_metadata,
        "E6_robustness": {
            "scenarios": scenario_results,
            "subgroups": subgroup_results,
            "subgroup_claim": (
                "Descriptive only; these analyses do not establish fairness."
            ),
        },
        "E7_runtime": runtime,
        "artifacts": {
            "predictions": {
                "path": str(PREDICTION_PATH.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(PREDICTION_PATH),
                "rows": int(len(predictions)),
            },
            "models": {
                site: {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256(path),
                }
                for site, path in model_paths.items()
            },
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "prediction_rows": len(predictions),
                "result_path": str(RESULT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
