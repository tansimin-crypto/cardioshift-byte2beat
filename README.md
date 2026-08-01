# CardioShift - Know When Not to Predict

CardioShift is a retrospective four-hospital research benchmark for studying
how a fixed heart-disease classifier behaves when transferred to an unseen
hospital. It is not for diagnosis, treatment, medication, triage, or
real-patient decisions.

## Main result

The same bounded model search achieved mean random-split AUROC
**0.892**, while pooled leave-one-hospital-out
AUROC was **0.789**. The
**0.102** gap is the central result.

The prespecified safety gate accepted
**40.5%** of cases with
**10.2%** error among accepted cases.
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
.venv-demo\Scripts\python -m pip install -r requirements-demo.lock
.venv-demo\Scripts\python -c "import app; assert len(app.PAGES) == 5"
.venv-demo\Scripts\streamlit run app.py
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
