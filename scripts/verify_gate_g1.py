"""Independent machine checks for Gate G1 data and provenance artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import EXPECTED_SITES, PREDICTOR_COLUMNS, SOURCE_COLUMNS

AUDIT_PATH = ROOT / "data" / "audit.json"
CHECKSUMS_PATH = ROOT / "data" / "checksums.json"
COHORT_PATH = ROOT / "data" / "processed" / "cardioshift_cohort.csv"
CONTRACT_PATH = ROOT / "competition-contract.yaml"
OUTPUT_PATH = ROOT / "data" / "gate_g1_verification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main(*, write: bool = True) -> None:
    checks: list[str] = []
    for path in (AUDIT_PATH, CHECKSUMS_PATH, COHORT_PATH, CONTRACT_PATH):
        require(path.exists(), f"exists: {path.relative_to(ROOT)}", checks)

    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    require(
        contract["competition"]["mode"] == "judge_based",
        "competition mode is judge_based",
        checks,
    )
    require(
        len(contract["ambiguities"]) >= 3,
        "eligibility and submission ambiguities are recorded",
        checks,
    )

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(CHECKSUMS_PATH.read_text(encoding="utf-8"))
    cohort = pd.read_csv(COHORT_PATH)
    expected_columns = ["patient_id", "site", *SOURCE_COLUMNS, "target"]

    require(audit["status"] == "pass", "canonical data audit passes", checks)
    require(cohort.columns.tolist() == expected_columns, "schema is exact", checks)
    require(len(cohort) == audit["rows"], "row count matches audit", checks)
    require(
        tuple(cohort["site"].drop_duplicates()) == EXPECTED_SITES,
        "site mapping and order are exact",
        checks,
    )
    require(cohort["patient_id"].is_unique, "patient_id is globally unique", checks)
    require(
        cohort["target"].equals((cohort["num"] > 0).astype("int64")),
        "binary target is exactly num > 0",
        checks,
    )
    require(
        cohort[PREDICTOR_COLUMNS].isna().sum().to_dict()
        == audit["missing_by_feature"],
        "missing counts match audit",
        checks,
    )
    require(
        sum(audit["missing_by_feature"].values()) > 0,
        "raw missing values are retained",
        checks,
    )

    manifest_entry = manifest["files"]["data/processed/cardioshift_cohort.csv"]
    require(
        sha256(COHORT_PATH) == manifest_entry["sha256"] == audit["output"]["sha256"],
        "processed cohort SHA-256 matches manifest and audit",
        checks,
    )
    raw_entries = [
        "data/raw/processed.cleveland.data",
        "data/raw/processed.hungarian.data",
        "data/raw/processed.switzerland.data",
        "data/raw/processed.va.data",
    ]
    for relative in raw_entries:
        path = ROOT / relative
        require(path.exists(), f"exists: {relative}", checks)
        require(
            sha256(path) == manifest["files"][relative]["sha256"],
            f"SHA-256 matches: {relative}",
            checks,
        )

    report = {
        "gate": "G1",
        "status": "pass",
        "checks": checks,
        "cohort_sha256": sha256(COHORT_PATH),
        "rows": int(len(cohort)),
        "sites": cohort.groupby("site", sort=False).size().to_dict(),
    }
    if write:
        OUTPUT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    main(write=not args.no_write)
