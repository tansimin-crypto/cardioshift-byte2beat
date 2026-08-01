"""Verify E1/E2 artifacts, outer-fold coverage, and training ledgers."""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COHORT_PATH = ROOT / "data" / "processed" / "cardioshift_cohort.csv"
RESULTS_PATH = ROOT / "outputs" / "metrics" / "results.json"
LOHO_PATH = ROOT / "outputs" / "predictions" / "loho_predictions.csv"
RANDOM_PATH = ROOT / "outputs" / "predictions" / "random_split_predictions.csv"
LEDGER_PATH = ROOT / "outputs" / "audit" / "loho_training_ledgers.json"
EVIDENCE_DIR = ROOT / "evidence" / "g2"
REPORT_PATH = EVIDENCE_DIR / "verification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main(*, write: bool = True) -> None:
    checks: list[str] = []
    for path in (COHORT_PATH, RESULTS_PATH, LOHO_PATH, RANDOM_PATH, LEDGER_PATH):
        require(path.exists(), f"exists: {path.relative_to(ROOT)}", checks)

    cohort = pd.read_csv(COHORT_PATH)
    loho = pd.read_csv(LOHO_PATH)
    random = pd.read_csv(RANDOM_PATH)
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    ledgers = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))

    require(
        results["status"] == "core_experiments_complete",
        "results status is core_experiments_complete",
        checks,
    )
    require(
        results["reproducibility"]["candidate_count"] == 18,
        "all 18 prespecified candidates were evaluated",
        checks,
    )
    require(
        results["reproducibility"]["input_sha256"] == sha256(COHORT_PATH),
        "results input hash matches canonical cohort",
        checks,
    )
    require(len(ledgers) == 4, "four LOHO training ledgers exist", checks)
    require(len(loho) == len(cohort), "LOHO predicts every cohort row once", checks)
    require(loho["patient_id"].is_unique, "LOHO patient IDs are unique", checks)
    require(
        set(loho["patient_id"]) == set(cohort["patient_id"]),
        "LOHO patient universe exactly matches cohort",
        checks,
    )
    require(
        (loho["site"] == loho["site"]).all(),
        "LOHO site values are present",
        checks,
    )
    for probability_column in ("raw_probability", "calibrated_probability"):
        values = loho[probability_column].to_numpy()
        require(np.isfinite(values).all(), f"{probability_column} is finite", checks)
        require(
            ((values >= 0) & (values <= 1)).all(),
            f"{probability_column} is within [0,1]",
            checks,
        )

    for ledger in ledgers:
        held_out_site = ledger["held_out_site"]
        test_ids = set(ledger["outer_test_patient_ids"])
        require(
            ledger["outer_test_sites"] == [held_out_site],
            f"{held_out_site}: outer test contains only held-out site",
            checks,
        )
        require(
            held_out_site not in ledger["outer_train_sites"],
            f"{held_out_site}: held-out site absent from outer training sites",
            checks,
        )
        expected_test_ids = set(
            cohort.loc[cohort["site"] == held_out_site, "patient_id"]
        )
        require(
            test_ids == expected_test_ids,
            f"{held_out_site}: ledger test IDs match cohort site",
            checks,
        )
        for stage in ("tuning", "calibration", "final_fit"):
            stage_ids = set(ledger[f"{stage}_patient_ids"])
            require(
                stage_ids.isdisjoint(test_ids),
                f"{held_out_site}: test IDs absent from {stage}",
                checks,
            )

    repeats = int(results["E1_repeated_random_split"]["repeats"])
    require(repeats == 10, "ten random-split repeats exist", checks)
    require(
        random["repeat"].nunique() == repeats,
        "random prediction file contains every repeat",
        checks,
    )
    require(
        random.groupby("repeat").size().nunique() == 1,
        "random repeat test sizes are consistent",
        checks,
    )

    pooled = results["E2_leave_one_hospital_out"]["pooled"]
    require(pooled["n"] == len(cohort), "pooled LOHO n matches cohort", checks)
    require(
        len(results["E2_leave_one_hospital_out"]["by_site"]) == 4,
        "four per-site metric records exist",
        checks,
    )
    require(
        pooled["bootstrap"]["requested_replicates"] == 2000,
        "pooled LOHO uses 2,000 bootstrap replicates",
        checks,
    )
    for record in results["E2_leave_one_hospital_out"]["by_site"]:
        require(
            record["metrics"]["bootstrap"]["requested_replicates"] == 2000,
            f"{record['held_out_site']}: uses 2,000 bootstrap replicates",
            checks,
        )

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "gate": "G2",
        "status": "pass",
        "checks": checks,
        "artifacts": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (RESULTS_PATH, LOHO_PATH, RANDOM_PATH, LEDGER_PATH)
        },
        "accepted_core_metrics": {
            "random_split_mean_auroc": results["E1_repeated_random_split"]["summary"][
                "auroc"
            ]["mean"],
            "loho_pooled_auroc": pooled["auroc"],
            "random_minus_loho_auroc": results["comparison"][
                "random_mean_minus_loho_pooled"
            ]["auroc"],
            "random_split_mean_brier": results["E1_repeated_random_split"]["summary"][
                "brier"
            ]["mean"],
            "loho_pooled_brier": pooled["brier"],
        },
    }
    if write:
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    main(write=not args.no_write)
