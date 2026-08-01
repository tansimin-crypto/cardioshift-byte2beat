"""Build the offline Kaggle report notebook and its uploadable data bundle."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "CardioShift_Research_Report.ipynb"
DATASET_DIR = ROOT / "dist" / "kaggle" / "cardioshift-data"
WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def canonical_sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {".json", ".csv", ".md", ".py", ".yaml", ".yml"}:
        content = content.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def sanitize_value(value: Any) -> Any:
    """Redact machine-local paths from publishable JSON values."""
    if isinstance(value, dict):
        return {key: sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace(str(ROOT), "<REPOSITORY_ROOT>")
        normalized = normalized.replace(
            str(ROOT).replace("\\", "/"), "<REPOSITORY_ROOT>"
        )
        if WINDOWS_PATH.match(normalized):
            if normalized.lower().endswith(("python.exe", "python")):
                return "<PYTHON_EXECUTABLE>"
            return "<LOCAL_PATH_REDACTED>"
        return normalized
    return value


def publishable_results() -> dict[str, Any]:
    """Keep scientific results while omitting stale, machine-local command logs."""
    results = json.loads(
        (ROOT / "outputs" / "results.json").read_text(encoding="utf-8")
    )
    judge = results.get("independent_judge")
    if isinstance(judge, dict):
        results["independent_judge"] = {
            key: judge[key]
            for key in ("status", "review_type", "verified_at_utc")
            if key in judge
        }
        results["independent_judge"]["historical_details_omitted"] = True
    local_check = results.get("g4_local_environment_check")
    if isinstance(local_check, dict):
        results["g4_local_environment_check"] = {
            key: local_check[key]
            for key in (
                "schema_version",
                "status",
                "scope",
                "verified_at_utc",
                "verified_head",
                "pytest_counts",
            )
            if key in local_check
        }
        results["g4_local_environment_check"][
            "historical_command_logs_omitted"
        ] = True
    return sanitize_value(results)


def markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def prepare_dataset() -> dict[str, dict[str, Any]]:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    raw_dir = DATASET_DIR / "raw"
    outputs_dir = DATASET_DIR / "outputs"
    source_dir = DATASET_DIR / "source"
    for directory in (raw_dir, outputs_dir / "metrics", outputs_dir / "predictions", outputs_dir / "figures", source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copies = {
        ROOT / "data" / "raw" / "processed.cleveland.data": raw_dir / "processed.cleveland.data",
        ROOT / "data" / "raw" / "processed.hungarian.data": raw_dir / "processed.hungarian.data",
        ROOT / "data" / "raw" / "processed.switzerland.data": raw_dir / "processed.switzerland.data",
        ROOT / "data" / "raw" / "processed.va.data": raw_dir / "processed.va.data",
        ROOT / "data" / "checksums.json": DATASET_DIR / "checksums.json",
        ROOT / "outputs" / "metrics" / "robustness_results.json": outputs_dir / "metrics" / "robustness_results.json",
        ROOT / "outputs" / "predictions" / "loho_predictions.csv": outputs_dir / "predictions" / "loho_predictions.csv",
        ROOT / "outputs" / "predictions" / "random_split_predictions.csv": outputs_dir / "predictions" / "random_split_predictions.csv",
        ROOT / "outputs" / "predictions" / "safety_loho_predictions.csv": outputs_dir / "predictions" / "safety_loho_predictions.csv",
        ROOT / "outputs" / "predictions" / "robustness_predictions.csv": outputs_dir / "predictions" / "robustness_predictions.csv",
        ROOT / "src" / "__init__.py": source_dir / "src" / "__init__.py",
        ROOT / "src" / "results_access.py": source_dir / "src" / "results_access.py",
    }
    for source, target in copies.items():
        if not source.exists():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    (outputs_dir / "results.json").write_text(
        json.dumps(publishable_results(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    shutil.copy2(ROOT / "LICENSE", DATASET_DIR / "LICENSE")
    (DATASET_DIR / "SOFTWARE_LICENSE.md").write_text(
        "# Software license\n\n"
        "CardioShift source code in this bundle is licensed under the MIT License; "
        "see `LICENSE`. The UCI Heart Disease data files remain under CC BY 4.0; "
        "see `UCI_LICENSE.md`.\n",
        encoding="utf-8",
        newline="\n",
    )

    (DATASET_DIR / "DOI.txt").write_text(
        "UCI Heart Disease dataset DOI: 10.24432/C52P4X\n",
        encoding="utf-8",
    )
    (DATASET_DIR / "UCI_LICENSE.md").write_text(
        "# Dataset license\n\n"
        "The UCI Heart Disease dataset is distributed under CC BY 4.0.\n"
        "Source: https://doi.org/10.24432/C52P4X\n\n"
        "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(
        DATASET_DIR / "cardioshift_source.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted((ROOT / "src").glob("*.py")):
            archive.write(path, path.relative_to(ROOT))
        for path in sorted((ROOT / "scripts").glob("*.py")):
            archive.write(path, path.relative_to(ROOT))
        archive.write(ROOT / "configs" / "experiment.yaml", "configs/experiment.yaml")
        archive.write(ROOT / "requirements.lock", "requirements.lock")
        archive.write(ROOT / "LICENSE", "LICENSE")
        archive.writestr(
            "SOFTWARE_LICENSE.md",
            "CardioShift source code is licensed under the MIT License in LICENSE.\n"
            "Bundled UCI data is separately licensed under CC BY 4.0.\n",
        )

    files = [
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        str(path.relative_to(DATASET_DIR)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": canonical_sha256(path),
        }
        for path in sorted(files)
    }
    (DATASET_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "internet_required": False,
                "dataset_doi": "10.24432/C52P4X",
                "license": "CC BY 4.0",
                "files": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def build_notebook() -> None:
    cells = [
        markdown(
            """
# CardioShift — Know When Not to Predict

## 1. Executive Summary

This executable report evaluates transportability across four historical
hospital cohorts. Headline values below are emitted by the shared results
accessor from validated run artifacts; they are not copied into prose.
"""
        ),
        code(
            """
from pathlib import Path
import hashlib, json, os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def locate_data() -> Path:
    configured = os.environ.get("CARDIOSHIFT_DATA_DIR")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    cwd = Path.cwd()
    candidates.extend([cwd, cwd / "dist" / "kaggle" / "cardioshift-data"])
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        candidates.extend(path.parent for path in kaggle_root.rglob("manifest.json"))
    for candidate in candidates:
        if (candidate / "outputs" / "results.json").exists():
            return candidate
    raise FileNotFoundError("Set CARDIOSHIFT_DATA_DIR to the Kaggle Dataset directory")

BASE = locate_data()
SOURCE = BASE / "source"
if SOURCE.exists():
    sys.path.insert(0, str(SOURCE))
elif str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from src.results_access import ResultsAccessor

OUTPUT_DIR = (
    Path("/kaggle/working/cardioshift_outputs")
    if Path("/kaggle/working").exists()
    else Path(os.environ.get("CARDIOSHIFT_OUTPUT_DIR", Path.cwd() / "cardioshift_outputs"))
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
R = ResultsAccessor(BASE)
F = R.findings
print({
    "random_split_mean_auroc": round(F["random_split_mean_auroc"], 3),
    "loho_pooled_auroc": round(F["loho_pooled_auroc"], 3),
    "optimism_gap": round(F["random_minus_loho_auroc"], 3),
    "safety_coverage": round(F["safety_gate_coverage"], 3),
    "selective_error": round(F["safety_gate_selective_error"], 3),
})
"""
        ),
        markdown(
            """
## 2. Intended Use and Non-use

This is a retrospective research benchmark for studying hospital shift,
calibration, selective prediction, and failure modes. It does not predict
future risk and is not for diagnosis, treatment, medication, triage, or
real-patient decisions.

## 3. Dataset Provenance and License

The four processed center files come from the UCI Heart Disease dataset
(DOI 10.24432/C52P4X), licensed CC BY 4.0. The offline bundle includes the
source files, checksums, attribution, source archive, and reproducibility
manifest.
"""
        ),
        code(
            """
manifest_path = BASE / "manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    failures = []
    platform_transforms = []
    for relative, expected in manifest["files"].items():
        path = BASE / relative
        if not path.is_file():
            unpacked = path.with_suffix("")
            if Path("/kaggle/input").exists() and path.suffix.lower() == ".zip" and unpacked.is_dir():
                platform_transforms.append(relative)
                continue
            failures.append(f"{relative}: missing")
            continue
        content = path.read_bytes()
        if path.suffix.lower() in {".json", ".csv", ".md", ".py", ".yaml", ".yml"}:
            content = content.replace(b"\\r\\r\\n", b"\\n").replace(b"\\r\\n", b"\\n")
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected["sha256"]:
            failures.append(relative)
    assert not failures, failures
    print(f"Offline manifest verified: {len(manifest['files'])} files")
    print(f"Kaggle archive transforms: {platform_transforms}")
else:
    print("Local repository mode: canonical artifact hashes are verified by the test suite.")
"""
        ),
        markdown(
            """
## 4. Four-hospital cohort

The pooled cohort contains one deidentified row per record from Cleveland,
Hungary, Switzerland, and VA Long Beach. Missing values remain missing until
training-fold preprocessing.
"""
        ),
        code(
            """
hospital = R.per_hospital()
display(hospital)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(hospital["hospital"], hospital["n"], color="#3b6ea8")
ax.set_ylabel("Records")
ax.set_title("Four-hospital cohort")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cohort_by_hospital.png", dpi=140)
plt.show()
"""
        ),
        markdown(
            """
## 5. Validation design

The primary design is leave-one-hospital-out (LOHO): every patient in the test
hospital is excluded from tuning, calibration, and final fitting. Repeated
random splitting is retained only as a contrast. Random splitting estimates
within-mixture performance; it is not deployment validation for a new hospital.

## 6. Random Split vs LOHO
"""
        ),
        code(
            """
comparison = pd.DataFrame({
    "design": ["Repeated random split", "Leave-one-hospital-out"],
    "AUROC": [F["random_split_mean_auroc"], F["loho_pooled_auroc"]],
    "Brier": [F["random_split_mean_brier"], F["loho_pooled_brier"]],
})
display(comparison)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
axes[0].bar(comparison["design"], comparison["AUROC"], color=["#579c87", "#d97757"])
axes[1].bar(comparison["design"], comparison["Brier"], color=["#579c87", "#d97757"])
axes[0].set_ylim(0.5, 1.0); axes[0].set_title("AUROC (higher is better)")
axes[1].set_ylim(0.0, 0.3); axes[1].set_title("Brier (lower is better)")
for ax in axes: ax.tick_params(axis="x", rotation=18)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "random_vs_loho.png", dpi=140)
plt.show()
"""
        ),
        markdown("## 7. Per-hospital calibration"),
        code(
            """
display(hospital)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(hospital["hospital"], hospital["auroc"], color="#d97757")
ax.axhline(F["loho_pooled_auroc"], color="black", linestyle="--", label="pooled")
ax.set_ylim(0.45, 1.0); ax.set_ylabel("AUROC"); ax.legend()
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "loho_by_hospital.png", dpi=140)
plt.show()
"""
        ),
        markdown(
            """
## 8. Dataset-shift diagnostics

Hospital identity is predictable from inputs, which is evidence of dataset
shift—not a clinical prediction task. Prevalence, missingness, and standardized
feature differences vary substantially by center.
"""
        ),
        code(
            """
shift = R.canonical["experiments"]["E3_E5"]["E3_shift"]
print({
    "site_classifier_balanced_accuracy": F["site_classifier_balanced_accuracy"],
    "balanced_chance": F["site_classifier_chance"],
})
missing = pd.DataFrame(shift["missingness_by_site"]).T
display(missing)
"""
        ),
        markdown(
            """
## 9. Selective prediction

The prespecified gate combines an outer-training OOD threshold, a fixed
probability ambiguity band, class-conditional conformal sets, and a fixed
training missingness quantile. Deferral is a coverage–error tradeoff, not proof
of clinical safety.
"""
        ),
        code(
            """
print({
    "coverage": F["safety_gate_coverage"],
    "deferral_rate": 1 - F["safety_gate_coverage"],
    "selective_error": F["safety_gate_selective_error"],
    "accepted_case_fnr": F["accepted_case_fnr"],
    "empirical_conformal_coverage": F["empirical_conformal_coverage"],
})
"""
        ),
        markdown("## 10. E6 robustness"),
        code(
            """
robust = R.robustness_summary()
display(robust)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(robust["scenario"], robust["auroc"], marker="o", label="AUROC")
ax.plot(robust["scenario"], robust["safety_coverage"], marker="s", label="Safety coverage")
ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=30); ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "robustness.png", dpi=140)
plt.show()
"""
        ),
        markdown(
            """
## 11. Subgroups

Sex and age-band analyses use accepted patient-level LOHO predictions. They are
descriptive only and do not establish fairness.
"""
        ),
        code("display(R.subgroup_summary())"),
        markdown(
            """
## 12. Runtime profile

The values below are local CPU research-runtime measurements, not
medical-device benchmarks.
"""
        ),
        code(
            """
runtime = pd.DataFrame(R.runtime["by_fold"]).T
display(runtime[["serialized_model_bytes", "batch_1_ms", "batch_100_ms", "peak_process_memory_bytes"]])
print(R.runtime["environment"])
"""
        ),
        markdown(
            """
## 13. Failure case

VA Long Beach achieved the worst empirical conformal coverage. This matters
because the safety method itself can fail under hospital shift; the gate must
not be presented as a guarantee.
"""
        ),
        code(
            """
worst = F["worst_site_conformal_coverage"]
print({"hospital": worst["site"], "empirical_conformal_coverage": worst["value"]})
"""
        ),
        markdown("## 14. Limitations"),
        code(
            """
for limitation in R.limitations:
    print(f"- {limitation}")
"""
        ),
        markdown("## 15. Reproducibility manifest"),
        code(
            """
summary = {
    "canonical_results_sha256": R.sha256("outputs/results.json"),
    "input_sha256": R.canonical["data"]["input_sha256"],
    "gate_status": R.gates,
    "output_directory": str(OUTPUT_DIR),
    "internet_used": False,
}
print(json.dumps(summary, indent=2))
(OUTPUT_DIR / "notebook_run_summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
"""
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"cardioshift-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "cardioshift": {
                "internet": False,
                "generated_by": "scripts/build_kaggle_notebook.py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    manifest = prepare_dataset()
    build_notebook()
    print(
        json.dumps(
            {
                "notebook": str(NOTEBOOK_PATH),
                "dataset": str(DATASET_DIR),
                "dataset_files": len(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
