"""Verify E3/E5 artifacts and recompute selective metrics from predictions."""

from __future__ import annotations

import hashlib
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.safety import selective_metrics

COHORT_PATH = ROOT / "data" / "processed" / "cardioshift_cohort.csv"
CORE_PATH = ROOT / "outputs" / "metrics" / "results.json"
CORE_PREDICTIONS_PATH = ROOT / "outputs" / "predictions" / "loho_predictions.csv"
RESULT_PATH = ROOT / "outputs" / "metrics" / "shift_safety_results.json"
PREDICTIONS_PATH = ROOT / "outputs" / "predictions" / "safety_loho_predictions.csv"
EVIDENCE_PATH = ROOT / "evidence" / "e3_e5" / "verification.json"


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {".json", ".csv", ".md", ".py", ".yaml", ".yml"}:
        content = content.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main(*, write: bool = True) -> None:
    checks: list[str] = []
    for path in (
        COHORT_PATH,
        CORE_PATH,
        CORE_PREDICTIONS_PATH,
        RESULT_PATH,
        PREDICTIONS_PATH,
    ):
        require(path.exists(), f"exists: {path.relative_to(ROOT)}", checks)

    cohort = pd.read_csv(COHORT_PATH)
    core_predictions = pd.read_csv(CORE_PREDICTIONS_PATH).sort_values("patient_id")
    safety_predictions = pd.read_csv(PREDICTIONS_PATH).sort_values("patient_id")
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    require(
        result["status"] == "shift_safety_complete",
        "shift/safety result status is complete",
        checks,
    )
    require(
        result["reproducibility"]["input_sha256"] == raw_sha256(COHORT_PATH),
        "shift/safety input hash matches cohort",
        checks,
    )
    require(
        result["reproducibility"]["core_results_sha256"]
        == canonical_sha256(CORE_PATH),
        "shift/safety core result hash matches",
        checks,
    )
    require(
        result["reproducibility"]["core_predictions_reproduced_exactly"],
        "core LOHO probabilities were reproduced before safety analysis",
        checks,
    )
    require(
        len(safety_predictions) == len(cohort)
        and safety_predictions["patient_id"].is_unique,
        "safety predictions cover every cohort row once",
        checks,
    )
    require(
        set(safety_predictions["patient_id"]) == set(cohort["patient_id"]),
        "safety patient universe matches cohort",
        checks,
    )
    require(
        np.allclose(
            core_predictions["calibrated_probability"],
            safety_predictions["calibrated_probability"],
            atol=1e-12,
            rtol=1e-10,
        ),
        "safety calibrated probabilities match accepted G2 probabilities",
        checks,
    )

    expected_accepted = ~(
        safety_predictions["is_ood"].astype(bool)
        | safety_predictions["is_ambiguous"].astype(bool)
        | (safety_predictions["conformal_set_size"] != 1)
        | (
            safety_predictions["missing_fraction"]
            > safety_predictions["missing_threshold"]
        )
    )
    require(
        expected_accepted.equals(safety_predictions["accepted"].astype(bool)),
        "accepted flag exactly follows prespecified safety gate",
        checks,
    )

    recomputed = selective_metrics(
        safety_predictions["target"].to_numpy(),
        safety_predictions["calibrated_probability"].to_numpy(),
        safety_predictions["accepted"].to_numpy(),
        safety_predictions["conformal_contains_true"].to_numpy(),
        safety_predictions["conformal_set_size"].to_numpy(),
    )
    pooled = result["E5_safety"]["pooled"]
    for name in (
        "n",
        "events",
        "accepted_n",
        "coverage",
        "deferral_rate",
        "selective_risk",
        "accepted_case_fnr",
        "empirical_conformal_coverage",
        "mean_prediction_set_size",
    ):
        require(
            np.isclose(recomputed[name], pooled[name], equal_nan=True),
            f"pooled {name} recomputes from patient predictions",
            checks,
        )

    site_score = result["E3_shift"]["site_predictability"]["balanced_accuracy"]
    require(np.isfinite(site_score), "site balanced accuracy is finite", checks)
    require(
        site_score > result["E3_shift"]["site_predictability"][
            "chance_balanced_accuracy"
        ],
        "site classifier exceeds balanced chance",
        checks,
    )
    require(
        "Empirical coverage only" in result["E5_safety"]["conformal_claim"],
        "conformal claim is limited to empirical coverage",
        checks,
    )
    require(
        "in-sample" in result["E5_safety"]["ood_limitation"],
        "OOD threshold limitation is explicit",
        checks,
    )

    for fold in result["E5_safety"]["by_site"]:
        require(
            fold["thresholds"]["ood_threshold_source"]
            == "outer-training in-sample score quantile",
            f"{fold['held_out_site']}: OOD threshold source is training-only",
            checks,
        )
        require(
            fold["metrics"]["bootstrap"]["requested_replicates"] == 2000,
            f"{fold['held_out_site']}: safety metrics use 2,000 bootstraps",
            checks,
        )
        require(
            len(fold["risk_coverage_curve"]) == 19,
            f"{fold['held_out_site']}: risk-coverage curve has 19 points",
            checks,
        )

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiments": ["E3", "E5"],
        "status": "pass",
        "checks": checks,
        "artifacts": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": canonical_sha256(path),
            }
            for path in (RESULT_PATH, PREDICTIONS_PATH)
        },
        "accepted_metrics": {
            "site_balanced_accuracy": site_score,
            "coverage": pooled["coverage"],
            "selective_risk": pooled["selective_risk"],
            "accepted_case_fnr": pooled["accepted_case_fnr"],
            "empirical_conformal_coverage": pooled[
                "empirical_conformal_coverage"
            ],
        },
    }
    if write:
        EVIDENCE_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    main(write=not args.no_write)
