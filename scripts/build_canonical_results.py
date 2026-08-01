"""Merge accepted experiment artifacts into the single canonical results source."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.gate_status import OUTPUT_PATH as GATE_STATUS_PATH, build_gate_status
except ModuleNotFoundError:
    from gate_status import OUTPUT_PATH as GATE_STATUS_PATH, build_gate_status

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "outputs" / "metrics" / "results.json"
SAFETY_PATH = ROOT / "outputs" / "metrics" / "shift_safety_results.json"
FIGURES_PATH = ROOT / "outputs" / "figures" / "figure_manifest.json"
ROBUSTNESS_PATH = ROOT / "outputs" / "metrics" / "robustness_results.json"
G1_PATH = ROOT / "data" / "gate_g1_verification.json"
G2_PATH = ROOT / "evidence" / "g2" / "verification.json"
E3_E5_PATH = ROOT / "evidence" / "e3_e5" / "verification.json"
OUTPUT_PATH = ROOT / "outputs" / "results.json"
JUDGE_PATH = ROOT / "evidence" / "judge" / "current_verdict.json"
G4_LOCAL_PATH = ROOT / "evidence" / "g4" / "local_clean_environment.json"


def sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {
        ".json",
        ".csv",
        ".md",
        ".py",
        ".yaml",
        ".yml",
        ".tf",
        ".sh",
        ".ipynb",
    }:
        content = content.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> None:
    for path in (CORE_PATH, SAFETY_PATH, ROBUSTNESS_PATH, FIGURES_PATH, G1_PATH, G2_PATH, E3_E5_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    core = json.loads(CORE_PATH.read_text(encoding="utf-8"))
    safety = json.loads(SAFETY_PATH.read_text(encoding="utf-8"))
    figures = json.loads(FIGURES_PATH.read_text(encoding="utf-8"))
    g1 = json.loads(G1_PATH.read_text(encoding="utf-8"))
    g2 = json.loads(G2_PATH.read_text(encoding="utf-8"))
    robustness = json.loads(ROBUSTNESS_PATH.read_text(encoding="utf-8"))
    e3_e5 = json.loads(E3_E5_PATH.read_text(encoding="utf-8"))
    judge = (
        json.loads(JUDGE_PATH.read_text(encoding="utf-8")) if JUDGE_PATH.exists() else None
    )
    g4_local = (
        json.loads(G4_LOCAL_PATH.read_text(encoding="utf-8")) if G4_LOCAL_PATH.exists() else None
    )
    gate_status = build_gate_status(write=True)

    random_summary = core["E1_repeated_random_split"]["summary"]
    loho = core["E2_leave_one_hospital_out"]["pooled"]
    safety_pooled = safety["E5_safety"]["pooled"]
    safety_sites = safety["E5_safety"]["by_site"]
    worst_conformal = min(
        safety_sites,
        key=lambda record: record["metrics"]["empirical_conformal_coverage"],
    )
    loho_error = 1 - loho["accuracy"]
    selective_error = safety_pooled["selective_risk"]

    canonical = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit_before_canonicalization": git_commit(),
        "project": {
            "name": "CardioShift — Know When Not to Predict",
            "study_type": "retrospective multi-center transportability benchmark",
            "outcome": "angiographic heart-disease presence status",
            "not_future_risk": True,
            "not_for_clinical_use": True,
        },
        "gate_status_schema_version": gate_status["schema_version"],
        "gate_status": {
            gate: record["status"]
            for gate, record in gate_status["gates"].items()
        },
        "gate_evidence": gate_status["gates"],
        "independent_judge": judge,
        "g4_local_environment_check": g4_local,
        "key_findings": {
            "random_split_mean_auroc": random_summary["auroc"]["mean"],
            "random_split_auroc_repeat_interval": [
                random_summary["auroc"]["repeat_percentile_2_5"],
                random_summary["auroc"]["repeat_percentile_97_5"],
            ],
            "loho_pooled_auroc": loho["auroc"],
            "loho_pooled_auroc_ci": [
                loho["bootstrap"]["metrics"]["auroc"]["low"],
                loho["bootstrap"]["metrics"]["auroc"]["high"],
            ],
            "random_minus_loho_auroc": core["comparison"][
                "random_mean_minus_loho_pooled"
            ]["auroc"],
            "random_split_mean_brier": random_summary["brier"]["mean"],
            "loho_pooled_brier": loho["brier"],
            "site_classifier_balanced_accuracy": safety["E3_shift"][
                "site_predictability"
            ]["balanced_accuracy"],
            "site_classifier_chance": safety["E3_shift"]["site_predictability"][
                "chance_balanced_accuracy"
            ],
            "full_coverage_loho_error": loho_error,
            "safety_gate_coverage": safety_pooled["coverage"],
            "safety_gate_coverage_ci": [
                safety_pooled["bootstrap"]["metrics"]["coverage"]["low"],
                safety_pooled["bootstrap"]["metrics"]["coverage"]["high"],
            ],
            "safety_gate_selective_error": selective_error,
            "safety_gate_selective_error_ci": [
                safety_pooled["bootstrap"]["metrics"]["selective_risk"]["low"],
                safety_pooled["bootstrap"]["metrics"]["selective_risk"]["high"],
            ],
            "accepted_case_fnr": safety_pooled["accepted_case_fnr"],
            "empirical_conformal_coverage": safety_pooled[
                "empirical_conformal_coverage"
            ],
            "worst_site_conformal_coverage": {
                "site": worst_conformal["held_out_site"],
                "value": worst_conformal["metrics"][
                    "empirical_conformal_coverage"
                ],
            },
            "error_reduction_among_accepted_absolute": loho_error - selective_error,
        },
        "data": {
            "rows": core["study"]["rows"],
            "sites": core["study"]["sites"],
            "input_sha256": core["reproducibility"]["input_sha256"],
        },
        "experiments": {
            "E1_E2": core,
            "E3_E5": safety,
            "E6_robustness": robustness["E6_robustness"],
            "E7_runtime": robustness["E7_runtime"],
        },
        "figures": figures,
        "limitations": [
            "Historical, small, non-contemporary data from the 1980s.",
            "Site outcome prevalence and missingness are extreme in some hospitals.",
            "The target is recorded angiographic disease presence, not a future event.",
            "No prospective clinical deployment or decision-impact study was performed.",
            "The safety gate defers a majority of cases at its prespecified operating point.",
            "Conformal coverage is empirical only under hospital shift; exchangeability is not assured.",
            "The IsolationForest threshold uses an outer-training in-sample score quantile and may under-detect subtle shift.",
            "Missingness and subgroup stress tests are descriptive and do not establish clinical robustness or fairness.",
            "Results do not support diagnosis, treatment, medication, or real-patient use.",
        ],
        "artifact_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in (
                CORE_PATH,
                SAFETY_PATH,
                FIGURES_PATH,
                G1_PATH,
                G2_PATH,
                ROBUSTNESS_PATH,
                E3_E5_PATH,
                GATE_STATUS_PATH,
            )
        },
    }
    sanitized = sanitize(canonical)
    OUTPUT_PATH.write_text(
        json.dumps(
            sanitized,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
