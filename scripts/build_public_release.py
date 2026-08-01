"""Build a sanitized, standalone CardioShift public release candidate."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo_cases import select_demo_cases

DIST = ROOT / "dist" / "public-release"
RESULTS = ROOT / "outputs" / "results.json"
CODER_RUNTIME = ROOT / "evidence" / "g5" / "runtime_verification.json"

ROOT_FILES = (
    ".github/workflows/ci.yml",
    "LICENSE",
    "CITATION.cff",
    "app.py",
    "competition-contract.yaml",
    "EXPERIMENT_PROTOCOL.md",
    "requirements.lock",
    "requirements-notebook.lock",
    "requirements-demo.lock",
)
PUBLIC_SCRIPTS = (
    "build_canonical_results.py",
    "build_cohort.py",
    "build_kaggle_notebook.py",
    "build_public_release.py",
    "capture_kaggle_rules.py",
    "download_data.py",
    "execute_notebook.py",
    "make_core_figures.py",
    "render_data_card.py",
    "render_scientific_docs.py",
    "run_core_experiments.py",
    "run_robustness.py",
    "run_shift_safety.py",
    "sitecustomize.py",
    "verify_coder.py",
    "verify_gate_g1.py",
    "verify_gate_g2.py",
    "verify_kaggle_rules_evidence.py",
    "verify_public_release.py",
    "verify_robustness.py",
    "verify_shift_safety.py",
)
PUBLIC_TESTS = (
    "test_app_contract.py",
    "test_ci_contract.py",
    "test_coder_contract.py",
    "test_core_artifacts.py",
    "test_data_contract.py",
    "test_figure_contract.py",
    "test_gate_g1.py",
    "test_kaggle_rules_evidence.py",
    "test_no_leakage.py",
    "test_notebook_contract.py",
    "test_public_release_contract.py",
    "test_robustness_contract.py",
    "test_safety.py",
    "test_shift_safety_artifacts.py",
)
PUBLIC_EVIDENCE = (
    "evidence/e3_e5/verification.json",
    "evidence/e6_e7/verification.json",
    "evidence/frozen_e1_e5.json",
    "evidence/g2/verification.json",
    "evidence/g3/verification.json",
    "evidence/g5/verification.json",
    "evidence/g5/runtime_verification.json",
    "evidence/kaggle/current_rules.json",
    "evidence/kaggle/raw_pages.json",
)
PUBLIC_DATA = (
    "data/audit.json",
    "data/checksums.json",
    "data/data_dictionary.md",
    "data/gate_g1_verification.json",
    "data/README.md",
    "data/processed/cardioshift_cohort.csv",
)
PUBLIC_OUTPUT_FILES = (
    "outputs/results.json",
    "outputs/metrics/results.json",
    "outputs/metrics/robustness_results.json",
    "outputs/metrics/shift_safety_results.json",
    "outputs/audit/loho_training_ledgers.json",
    "outputs/audit/random_split_selections.json",
    "outputs/predictions/loho_predictions.csv",
    "outputs/predictions/random_split_predictions.csv",
    "outputs/predictions/robustness_predictions.csv",
    "outputs/predictions/safety_loho_predictions.csv",
)
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".lock",
    ".hcl",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tf",
    ".txt",
    ".yaml",
    ".yml",
}
WINDOWS_PATH = re.compile(r"^[A-Za-z]:\\")


def canonical_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
        "Dockerfile",
        "LICENSE",
        "known_hosts",
    }:
        content = content.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
    return content


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def copy_file(relative: str) -> None:
    source = ROOT / relative
    if not source.exists():
        raise FileNotFoundError(source)
    target = DIST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(relative: str) -> None:
    source = ROOT / relative
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        DIST / relative,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", ".notebook-work", ".terraform"
        ),
    )


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace(str(ROOT), "<REPOSITORY_ROOT>")
        normalized = normalized.replace(str(ROOT).replace("\\", "/"), "<REPOSITORY_ROOT>")
        if WINDOWS_PATH.match(normalized):
            if normalized.lower().endswith(("python.exe", "python")):
                return "<PYTHON_EXECUTABLE>"
            return "<LOCAL_PATH_REDACTED>"
        return normalized
    return value


def publishable_results() -> dict[str, Any]:
    """Omit historical machine logs while preserving frozen scientific content."""
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
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


def write_sanitized_results() -> dict[str, Any]:
    sanitized = publishable_results()
    target = DIST / "outputs" / "results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(sanitized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sanitized


def render_readme(results: dict[str, Any]) -> str:
    finding = results["key_findings"]
    return f"""# CardioShift - Know When Not to Predict

CardioShift is a retrospective four-hospital research benchmark for studying
how a fixed heart-disease classifier behaves when transferred to an unseen
hospital. It is not for diagnosis, treatment, medication, triage, or
real-patient decisions.

## Main result

The same bounded model search achieved mean random-split AUROC
**{finding['random_split_mean_auroc']:.3f}**, while pooled leave-one-hospital-out
AUROC was **{finding['loho_pooled_auroc']:.3f}**. The
**{finding['random_minus_loho_auroc']:.3f}** gap is the central result.

The prespecified safety gate accepted
**{finding['safety_gate_coverage']:.1%}** of cases with
**{finding['safety_gate_selective_error']:.1%}** error among accepted cases.
This is a coverage-error tradeoff, not evidence of clinical safety.

## Included artifacts

- `app.py`, `src/`, and deidentified fixed research cases;
- complete public scientific scripts and tests;
- locked core, Notebook, and Demo dependencies;
- the generated research Notebook, report, documentation, figures, metrics,
  predictions, and audit ledgers;
- the Coder Terraform template and provider lock;
- UCI data provenance plus the offline Kaggle Dataset payload;
- Kaggle Rules provenance and content-integrity evidence.

Private judge notes, local clean-room logs, credentials, local usernames, and
absolute host paths are intentionally excluded.

## Quick verification

```powershell
python -m pip install -r requirements.lock -r requirements-notebook.lock
python -m pytest -q --ignore=tests/test_app_contract.py
python scripts/verify_public_release.py --release-root .
```

Use the separate Demo environment for Streamlit:

```powershell
python -m venv .venv-demo
.venv-demo\\Scripts\\python -m pip install -r requirements-demo.lock
.venv-demo\\Scripts\\python -c "import app; assert len(app.PAGES) == 5"
.venv-demo\\Scripts\\streamlit run app.py
```

On Linux or macOS, use `.venv-demo/bin/python` and
`.venv-demo/bin/streamlit`.

## Rebuild the cohort and scientific artifacts

The four UCI source files are available under
`dist/kaggle/cardioshift-data/raw/`. Restore them and run:

```powershell
python scripts/build_cohort.py
python scripts/verify_gate_g1.py
python scripts/run_core_experiments.py
python scripts/verify_gate_g2.py
python scripts/run_shift_safety.py
python scripts/verify_shift_safety.py
python scripts/run_robustness.py
python scripts/verify_robustness.py
```

## Data and license

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989),
*Heart Disease*, UCI Machine Learning Repository,
<https://doi.org/10.24432/C52P4X>, CC BY 4.0.

Software is MIT licensed. See `LICENSE`, `CITATION.cff`,
`competition-contract.yaml`, `EXPERIMENT_PROTOCOL.md`, and
`report/manuscript.md`.
"""


def render_writeup(results: dict[str, Any]) -> str:
    finding = results["key_findings"]
    runtime = json.loads(CODER_RUNTIME.read_text(encoding="utf-8"))
    deferral = 1.0 - finding["safety_gate_coverage"]
    worst = finding["worst_site_conformal_coverage"]
    ci_low, ci_high = finding["loho_pooled_auroc_ci"]
    return f"""# Title

CardioShift: Know When Not to Predict

# Subtitle

A four-hospital external-validation benchmark for dataset shift, calibration,
and selective prediction in retrospective heart-disease data.

## The problem

A model can look strong when patients from every hospital are mixed into both
training and testing, yet fail when transferred to a hospital it has never
seen. CardioShift measures that gap without claiming clinical deployment.

## Why random splits mislead

Across the prespecified random splits, mean AUROC was
**{finding['random_split_mean_auroc']:.3f}**. With an entire hospital held out,
pooled AUROC fell to **{finding['loho_pooled_auroc']:.3f}** (95% patient
bootstrap CI **{ci_low:.3f}-{ci_high:.3f}**), an optimism gap of
**{finding['random_minus_loho_auroc']:.3f}**.

## Four-hospital validation

The evaluation holds out Cleveland, Hungary, Switzerland, and VA Long Beach
one at a time. The held-out hospital never participates in model choice,
calibration, or final fitting.

## Dataset shift

Hospital identity was predictable from the recorded input features with
balanced accuracy **{finding['site_classifier_balanced_accuracy']:.3f}**
against balanced chance **{finding['site_classifier_chance']:.2f}**. This
supports recorded source distinguishability; it does not identify a causal
mechanism.

## Calibration and selective prediction

The prespecified gate accepted **{finding['safety_gate_coverage']:.1%}** of
held-out-hospital cases and made errors in
**{finding['safety_gate_selective_error']:.1%}** of accepted cases. The cost
was **{deferral:.1%}** deferral. This is a coverage-error tradeoff, not proof of
clinical safety.

## Robustness

The frozen model and thresholds were reused across missingness, grouped
failure, and subgroup stress tests. No stress-test outcome was used to select
a new model or operating point.

## Failure cases

The Demo includes an accepted correct case, a deferred error, and a confident
error that the gate failed to capture. Empirical conformal coverage was lowest
in **{worst['site']}** at **{worst['value']:.1%}**, illustrating that
exchangeability is not assured under hospital shift.

## Limitations

{chr(10).join(f'- {item}' for item in results['limitations'])}

## Reproducibility

The public release contains locked environments, source code, tests, figures,
canonical metrics, patient-level predictions, the generated Notebook, and a
sanitized manifest. The Notebook is designed for attached data with Internet
disabled.

## Coder integration

The Coder template requires GitHub External Auth, fetches an explicit
`repo_ref`, and fails unless the workspace HEAD equals an explicit full release
SHA. A real workspace ran the audited release
`{runtime['tested_release_sha']}` with a clean Linux worktree, locked tests,
JupyterLab {runtime['cold_start']['jupyterlab_version']}, and healthy Streamlit.
The same checks passed again after a full stop/start cycle. The machine-readable
record is `evidence/g5/runtime_verification.json`.

## Public Notebook

https://www.kaggle.com/code/simingtan/cardioshift-know-when-not-to-predict

## Public Demo or project

https://github.com/tansimin-crypto/cardioshift-byte2beat

## Manuscript

https://github.com/tansimin-crypto/cardioshift-byte2beat/blob/main/report/manuscript.md

The full local manuscript is included at `report/manuscript.md`.
"""


def render_demo_script(results: dict[str, Any]) -> str:
    finding = results["key_findings"]
    runtime = json.loads(CODER_RUNTIME.read_text(encoding="utf-8"))
    cases = select_demo_cases(ROOT)
    deferral = 1.0 - finding["safety_gate_coverage"]
    worst = finding["worst_site_conformal_coverage"]
    accepted = cases["accepted_confident_correct"]
    deferred = cases["deferred_confident_error"]
    missed = cases["confident_error_not_caught_by_gate"]
    return f"""# CardioShift 2-3 minute judge demo

## 0:00-0:10 - Problem

"Random patient splits can mix all hospitals into training and testing. We ask
what happens when the next hospital is genuinely unseen."

## 0:10-0:35 - Same model, two realities

Show Figure 1 and state:

- random-split mean AUROC: **{finding['random_split_mean_auroc']:.3f}**;
- leave-one-hospital-out pooled AUROC: **{finding['loho_pooled_auroc']:.3f}**;
- optimism gap: **{finding['random_minus_loho_auroc']:.3f}**.

## 0:35-0:55 - Hospital shift

Show the prevalence/missingness and reliability figures. Hospital
classification balanced accuracy is
**{finding['site_classifier_balanced_accuracy']:.3f}** versus
**{finding['site_classifier_chance']:.2f}** balanced chance.

## 0:55-1:30 - Three fixed cases

1. **ACCEPT:** `{accepted['case_id']}` from `{accepted['hospital']}` is an
   accepted correct case.
2. **DEFER:** `{deferred['case_id']}` from `{deferred['hospital']}` is a
   deferred model error.
3. **Gate miss:** `{missed['case_id']}` from `{missed['hospital']}` is a
   confident error that remained ACCEPT.

These are deidentified retrospective research rows, not live patient inputs.

## 1:30-1:50 - Cost of deferral

The gate accepted **{finding['safety_gate_coverage']:.1%}** and therefore
deferred **{deferral:.1%}** of cases. Error among accepted cases was
**{finding['safety_gate_selective_error']:.1%}**.

## 1:50-2:05 - Conformal failure under shift

Worst-hospital empirical conformal coverage was **{worst['value']:.1%}** in
**{worst['site']}**. The Demo labels this as a limitation, not a guarantee.

## 2:05-2:35 - Real Coder workspace (verified)

Show the recorded Coder evidence and, when available, the running local
workspace:

1. workspace build with GitHub authentication;
2. `.coder-status/tests.ok` and `.coder-status/services.ok`;
3. `.coder-status/release.sha` equal to
   `{runtime['tested_release_sha']}`;
4. JupyterLab {runtime['cold_start']['jupyterlab_version']} opened through Coder;
5. Streamlit health `ok` and Jupyter API HTTP 200;
6. the same exact SHA and health checks after stop/start recovery.

The public evidence is `evidence/g5/runtime_verification.json`.

## 2:35-2:45 - Close

"CardioShift is a retrospective research benchmark for knowing when model
confidence does not transfer. It is not for diagnosis, treatment, medication,
triage, or real-patient decisions."
"""


def render_coder_demo() -> str:
    runtime = json.loads(CODER_RUNTIME.read_text(encoding="utf-8"))
    return f"""# Verified Coder integration

CardioShift was executed in a real local Coder workspace, not inferred from a
static Terraform plan or Docker build.

## Audited run

- Workspace: `{runtime['workspace']}`
- Template version: `{runtime['template_version']}`
- Tested release SHA: `{runtime['tested_release_sha']}`
- Repository transport: read-only GitHub SSH deploy key
- Linux worktree after startup: clean
- Test marker: `/workspace/.coder-status/tests.ok`
- Service marker: `/workspace/.coder-status/services.ok`
- JupyterLab: {runtime['cold_start']['jupyterlab_version']}, API HTTP 200
- Streamlit health: `ok`
- Stop/start recovery: passed with the same exact SHA

The machine-readable evidence is
[`evidence/g5/runtime_verification.json`](evidence/g5/runtime_verification.json).
The Coder template and fail-closed startup contract are under [`coder/`](coder/).

This proves the submitted research environment and application can be created
and restarted in Coder. It does not claim a public hosted clinical service.
"""


def normalize_public_text() -> None:
    for path in DIST.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "Dockerfile", "LICENSE", "known_hosts"
        }:
            continue
        text = path.read_text(encoding="utf-8")
        path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sanitize_kaggle_payload() -> None:
    payload = DIST / "dist" / "kaggle" / "cardioshift-data"
    results_path = payload / "outputs" / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results_path.write_text(
        json.dumps(sanitize_value(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = payload / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        path = payload / relative
        content = canonical_bytes(path)
        record["bytes"] = len(content)
        record["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    archive = DIST.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    for relative in ROOT_FILES:
        copy_file(relative)
    for relative in ("src", "configs", "docs", "report", "notebooks"):
        copy_tree(relative)
    executed_notebook = DIST / "notebooks" / "CardioShift_Research_Report.executed.ipynb"
    if executed_notebook.exists():
        executed_notebook.unlink()
    shutil.rmtree(DIST / "notebooks" / ".notebook-work", ignore_errors=True)
    copy_tree("coder")
    shutil.rmtree(DIST / "coder" / ".terraform", ignore_errors=True)
    copy_tree("dist/kaggle")
    copy_tree("outputs/figures")

    for name in PUBLIC_SCRIPTS:
        copy_file(f"scripts/{name}")
    for name in PUBLIC_TESTS:
        copy_file(f"tests/{name}")
    for relative in PUBLIC_EVIDENCE:
        copy_file(relative)
    for relative in PUBLIC_DATA:
        copy_file(relative)
    for relative in PUBLIC_OUTPUT_FILES:
        if relative != "outputs/results.json":
            copy_file(relative)

    normalize_public_text()
    sanitize_kaggle_payload()
    results = write_sanitized_results()
    (DIST / "README.md").write_text(render_readme(results), encoding="utf-8")
    (DIST / "KAGGLE_WRITEUP.md").write_text(
        render_writeup(results),
        encoding="utf-8",
    )
    (DIST / "DEMO_SCRIPT.md").write_text(
        render_demo_script(results),
        encoding="utf-8",
    )
    (DIST / "CODER_DEMO.md").write_text(
        render_coder_demo(),
        encoding="utf-8",
    )
    (DIST / "RELEASE_URLS.template.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "published": False,
                "public_repository": None,
                "public_kaggle_notebook": None,
                "public_demo": None,
                "public_manuscript": None,
                "audited_release_sha": None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    files = {
        str(path.relative_to(DIST)).replace("\\", "/"): {
            "bytes": len(canonical_bytes(path)),
            "sha256": canonical_sha256(path),
        }
        for path in sorted(DIST.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit_before_export": source_commit,
        "canonical_text_newlines": "LF",
        "standalone_public_release": True,
        "submission_ready_claimed": False,
        "files": files,
    }
    (DIST / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.make_archive(str(DIST), "zip", root_dir=DIST.parent, base_dir=DIST.name)
    print(
        json.dumps(
            {
                "status": "built",
                "files": len(files),
                "source_commit": source_commit,
                "output": "dist/public-release",
                "archive": "dist/public-release.zip",
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
