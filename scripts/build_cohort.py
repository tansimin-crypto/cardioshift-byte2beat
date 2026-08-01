"""Build and audit the standardized four-center CardioShift cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data import EXPECTED_SITES, PREDICTOR_COLUMNS, build_cohort

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT = PROCESSED_DIR / "cardioshift_cohort.csv"
AUDIT = ROOT / "data" / "audit.json"
CHECKSUMS = ROOT / "data" / "checksums.json"

LEGAL_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _jsonable(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def audit_cohort(cohort: pd.DataFrame) -> dict[str, object]:
    invalid_values: dict[str, list[object]] = {}
    for column, legal in LEGAL_VALUES.items():
        observed = set(cohort[column].dropna().unique())
        invalid_values[column] = sorted(_jsonable(value) for value in observed - legal)

    exact_duplicate_rows = int(
        cohort.drop(columns=["patient_id"]).duplicated(keep=False).sum()
    )
    site_summary: dict[str, object] = {}
    for site, group in cohort.groupby("site", sort=False):
        site_summary[site] = {
            "n": int(len(group)),
            "events": int(group["target"].sum()),
            "prevalence": float(group["target"].mean()),
            "missing_cells": int(group[PREDICTOR_COLUMNS].isna().sum().sum()),
            "missing_fraction": float(group[PREDICTOR_COLUMNS].isna().mean().mean()),
            "missing_by_feature": {
                key: int(value)
                for key, value in group[PREDICTOR_COLUMNS].isna().sum().items()
            },
        }

    failures: list[str] = []
    if tuple(cohort["site"].drop_duplicates()) != EXPECTED_SITES:
        failures.append("unexpected site order or membership")
    if cohort["patient_id"].duplicated().any():
        failures.append("duplicate patient_id")
    if not cohort["target"].isin([0, 1]).all():
        failures.append("target is not binary")
    if not (cohort["target"] == (cohort["num"] > 0).astype("int8")).all():
        failures.append("target does not exactly match num > 0")
    for column, values in invalid_values.items():
        if values:
            failures.append(f"{column} has invalid values: {values}")

    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "rows": int(len(cohort)),
        "columns": cohort.columns.tolist(),
        "sites": site_summary,
        "target_events": int(cohort["target"].sum()),
        "target_prevalence": float(cohort["target"].mean()),
        "missing_by_feature": {
            key: int(value)
            for key, value in cohort[PREDICTOR_COLUMNS].isna().sum().items()
        },
        "invalid_values": invalid_values,
        "exact_duplicate_rows_ignoring_patient_id": exact_duplicate_rows,
        "notes": [
            "Exact duplicates are reported, not silently removed.",
            "No imputation, encoding, scaling, or feature selection was performed.",
        ],
    }


def main() -> None:
    cohort = build_cohort(RAW_DIR)
    audit = audit_cohort(cohort)
    if audit["status"] != "pass":
        raise RuntimeError(json.dumps(audit, indent=2))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUTPUT, index=False, na_rep="", lineterminator="\r\n")
    audit["output"] = {
        "path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256(OUTPUT),
    }
    AUDIT.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(CHECKSUMS.read_text(encoding="utf-8"))
    manifest["files"][audit["output"]["path"]] = {
        "bytes": audit["output"]["bytes"],
        "sha256": audit["output"]["sha256"],
    }
    CHECKSUMS.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
