# Title

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
**0.892**. With an entire hospital held out,
pooled AUROC fell to **0.789** (95% patient
bootstrap CI **0.758-0.818**), an optimism gap of
**0.102**.

## Four-hospital validation

The evaluation holds out Cleveland, Hungary, Switzerland, and VA Long Beach
one at a time. The held-out hospital never participates in model choice,
calibration, or final fitting.

## Dataset shift

Hospital identity was predictable from the recorded input features with
balanced accuracy **0.911**
against balanced chance **0.25**. This
supports recorded source distinguishability; it does not identify a causal
mechanism.

## Calibration and selective prediction

The prespecified gate accepted **40.5%** of
held-out-hospital cases and made errors in
**10.2%** of accepted cases. The cost
was **59.5%** deferral. This is a coverage-error tradeoff, not proof of
clinical safety.

## Robustness

The frozen model and thresholds were reused across missingness, grouped
failure, and subgroup stress tests. No stress-test outcome was used to select
a new model or operating point.

## Failure cases

The Demo includes an accepted correct case, a deferred error, and a confident
error that the gate failed to capture. Empirical conformal coverage was lowest
in **VA Long Beach** at **78.0%**, illustrating that
exchangeability is not assured under hospital shift.

## Limitations

- Historical, small, non-contemporary data from the 1980s.
- Site outcome prevalence and missingness are extreme in some hospitals.
- The target is recorded angiographic disease presence, not a future event.
- No prospective clinical deployment or decision-impact study was performed.
- The safety gate defers a majority of cases at its prespecified operating point.
- Conformal coverage is empirical only under hospital shift; exchangeability is not assured.
- The IsolationForest threshold uses an outer-training in-sample score quantile and may under-detect subtle shift.
- Missingness and subgroup stress tests are descriptive and do not establish clinical robustness or fairness.
- Results do not support diagnosis, treatment, medication, or real-patient use.

## Reproducibility

The public release contains locked environments, source code, tests, figures,
canonical metrics, patient-level predictions, the generated Notebook, and a
sanitized manifest. The Notebook is designed for attached data with Internet
disabled.

## Coder integration

The Coder template requires GitHub External Auth, fetches an explicit
`repo_ref`, and fails unless the workspace HEAD equals an explicit full release
SHA. A real workspace ran the audited release
`6b7aadd9806f13ebedd6a3be4b09e5d8a48c440b` with a clean Linux worktree, locked tests,
JupyterLab 4.6.2, and healthy Streamlit.
The same checks passed again after a full stop/start cycle. The machine-readable
record is `evidence/g5/runtime_verification.json`.

## Public Notebook

https://www.kaggle.com/code/simingtan/cardioshift-know-when-not-to-predict

## Public Demo or project

https://github.com/tansimin-crypto/cardioshift-byte2beat

## Manuscript

https://github.com/tansimin-crypto/cardioshift-byte2beat/blob/main/report/manuscript.md

The full local manuscript is included at `report/manuscript.md`.
