"""Verify E6/E7 artifacts, frozen inputs, masking, universes, and metrics."""

from __future__ import annotations

import hashlib
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_robustness import mask_mcar, metric_set
from src.data import PREDICTOR_COLUMNS

CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
COHORT_PATH = ROOT / "data" / "processed" / "cardioshift_cohort.csv"
CORE_PREDICTIONS_PATH = ROOT / "outputs" / "predictions" / "loho_predictions.csv"
SAFETY_PATH = ROOT / "outputs" / "metrics" / "shift_safety_results.json"
FROZEN_PATH = ROOT / "evidence" / "frozen_e1_e5.json"
RESULT_PATH = ROOT / "outputs" / "metrics" / "robustness_results.json"
PREDICTION_PATH = ROOT / "outputs" / "predictions" / "robustness_predictions.csv"
OUTPUT_PATH = ROOT / "evidence" / "e6_e7" / "verification.json"


def canonical_content(path: Path, content: bytes) -> bytes:
    if path.suffix.lower() in {".json", ".csv", ".md", ".py", ".yaml", ".yml"}:
        content = content.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
    return content


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(canonical_content(path, path.read_bytes())).hexdigest()


def checkout_matches(path: Path, expected: str) -> bool:
    return canonical_sha256(path) == expected


def git_blob_sha256(revision: str, relative: str) -> str:
    content = subprocess.check_output(
        ["git", "show", f"{revision}:{relative}"],
        cwd=ROOT,
    )
    return hashlib.sha256(canonical_content(ROOT / relative, content)).hexdigest()


def close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.isclose(float(left), float(right), atol=1e-12, rtol=1e-10))


def main(*, regenerated_inputs: bool = False, write: bool = True) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cohort = pd.read_csv(COHORT_PATH)
    accepted = pd.read_csv(CORE_PREDICTIONS_PATH)
    safety = json.loads(SAFETY_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_csv(PREDICTION_PATH)
    checks: list[str] = []

    if result["status"] != "complete":
        raise AssertionError("robustness result is not complete")
    if result["protocol"]["model_selection"] != (
        "none; accepted E2 candidate reused per LOHO fold"
    ):
        raise AssertionError("E6/E7 changed the model-selection protocol")
    checks.append("no model reselection")

    for relative, record in frozen["artifacts"].items():
        expected = record["sha256"]
        if git_blob_sha256(frozen["accepted_commit"], relative) != expected:
            raise AssertionError(f"accepted Git blob hash mismatch: {relative}")
        if not regenerated_inputs and not checkout_matches(ROOT / relative, expected):
            raise AssertionError(f"frozen artifact changed: {relative}")
    checks.append("accepted Git blobs match all frozen E1-E5 hashes")
    checks.append(
        "regenerated E1-E5 inputs validated semantically downstream"
        if regenerated_inputs
        else "current checkout matches all frozen E1-E5 canonical hashes"
    )

    root_seed = int(config["project"]["root_seed"])
    repeats = int(config["stress_tests"]["mcar_repeats"])
    universe = set(cohort["patient_id"])
    scenario_groups = predictions.groupby(
        ["scenario_type", "scenario", "repeat"],
        dropna=False,
        sort=False,
    )
    expected_groups = 1 + len(config["stress_tests"]["mcar_fractions"]) * repeats
    expected_groups += len(config["stress_tests"]["grouped"])
    if len(scenario_groups) != expected_groups:
        raise AssertionError("scenario count differs from prespecified protocol")

    for (scenario_type, scenario, repeat), frame in scenario_groups:
        if len(frame) != len(cohort) or set(frame["patient_id"]) != universe:
            raise AssertionError(f"{scenario}: patient universe changed")
        key = f"{scenario_type}:{scenario}:repeat={int(repeat)}"
        recorded = result["E6_robustness"]["scenarios"][key]["pooled"]
        recomputed = metric_set(frame)
        for name, value in recorded.items():
            if not close(value, recomputed[name]):
                raise AssertionError(f"{key}: metric mismatch for {name}")
    checks.append("every perturbation has the full outer-test patient universe")
    checks.append("all pooled metrics recompute from patient-level predictions")

    baseline = predictions.loc[predictions["scenario_type"] == "baseline"].sort_values(
        "patient_id"
    )
    accepted_sorted = accepted.sort_values("patient_id")
    if not np.allclose(
        baseline["calibrated_probability"],
        accepted_sorted["calibrated_probability"],
        atol=1e-12,
        rtol=1e-10,
    ):
        raise AssertionError("baseline probabilities differ from accepted E2")
    checks.append("baseline probabilities equal accepted E2 probabilities")

    for fold_index, site in enumerate(
        item["held_out_site"]
        for item in json.loads(
            (ROOT / "outputs" / "metrics" / "results.json").read_text(
                encoding="utf-8"
            )
        )["E2_leave_one_hospital_out"]["by_site"]
    ):
        source = cohort.loc[cohort["site"] == site, PREDICTOR_COLUMNS].reset_index(
            drop=True
        )
        patient_ids = cohort.loc[cohort["site"] == site, "patient_id"].reset_index(
            drop=True
        )
        for fraction in config["stress_tests"]["mcar_fractions"]:
            scenario = f"mcar_{int(float(fraction) * 100)}"
            for repeat in range(repeats):
                seed = (
                    root_seed
                    + 1_000_000
                    + fold_index * 100_000
                    + int(float(fraction) * 1000) * 100
                    + repeat
                )
                _, row_masks = mask_mcar(source, fraction=float(fraction), seed=seed)
                expected_masks = dict(zip(patient_ids, row_masks))
                observed = predictions.loc[
                    (predictions["site"] == site)
                    & (predictions["scenario"] == scenario)
                    & (predictions["repeat"] == repeat)
                ]
                if not (observed["mask_seed"] == seed).all():
                    raise AssertionError(f"{site}/{scenario}/{repeat}: seed mismatch")
                for patient_id, mask in zip(
                    observed["patient_id"],
                    observed["newly_masked_features"].fillna(""),
                ):
                    if mask != expected_masks[patient_id]:
                        raise AssertionError(
                            f"{site}/{scenario}/{repeat}: mask mismatch"
                        )
    checks.append("MCAR seeds and row-level feature masks reproduce exactly")

    grouped = config["stress_tests"]["grouped"]
    for name, configured in grouped.items():
        expected_columns = (
            [column for column in PREDICTOR_COLUMNS if column not in configured]
            if name == "basic_vitals_only"
            else list(configured)
        )
        expected_label = ";".join(expected_columns)
        observed = predictions.loc[predictions["scenario"] == name]
        if set(observed["newly_masked_features"]) != {expected_label}:
            raise AssertionError(f"{name}: grouped mask columns changed")
    checks.append("grouped masks match config exactly")

    threshold_hash = result["protocol"]["threshold_source_sha256"]
    if threshold_hash != canonical_sha256(SAFETY_PATH):
        raise AssertionError("safety threshold source hash changed")
    for site, metadata in result["fold_metadata"].items():
        expected = next(
            item["thresholds"]
            for item in safety["E5_safety"]["by_site"]
            if item["held_out_site"] == site
        )
        if metadata["thresholds"] != expected:
            raise AssertionError(f"{site}: frozen safety thresholds changed")
    checks.append("OOD, conformal, ambiguity, and missingness thresholds are frozen")

    report = {
        "schema_version": "1.0",
        "status": "pass",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verified_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "checks": checks,
        "artifact_hashes": {
            "robustness_results": canonical_sha256(RESULT_PATH),
            "robustness_predictions": canonical_sha256(PREDICTION_PATH),
        },
        "rows": int(len(predictions)),
        "scenario_groups": len(scenario_groups),
        "unresolved_blockers": [],
    }
    if write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerated-inputs", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    main(regenerated_inputs=args.regenerated_inputs, write=not args.no_write)
