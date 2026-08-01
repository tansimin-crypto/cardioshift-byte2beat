# CardioShift Experiment Protocol

Status: prespecified before outcome-bearing experiments.

## Research question

How much does repeated random patient splitting overstate performance relative
to leave-one-hospital-out (LOHO) validation on the four UCI Heart Disease
centers, and can training-only calibration and abstention reduce confidently
wrong accepted predictions under hospital shift?

This is a retrospective benchmark of disease-presence classification. It is not
a clinical device, a diagnosis system, or a future-risk model.

## Cohort and outcome

- Sources: Cleveland, Hungary, Switzerland, and VA Long Beach processed files.
- Predictors: the 13 UCI variables listed in `competition-contract.yaml`.
- Validation-only fields: `site`, `patient_id`.
- Source outcome: `num`.
- Binary outcome: `target = 0` when `num == 0`; otherwise `target = 1`.
- Missing values are preserved at ingestion and handled inside training folds.

The data audit is frozen before modeling. Any exclusion must be documented with
the exact predicate, affected rows by site, and before/after hashes.

## Primary and secondary estimands

Primary:

- pooled patient-level LOHO AUROC, AUPRC, Brier score, and log loss;
- the same metrics per held-out hospital;
- worst-hospital performance;
- calibration intercept/slope and expected calibration error (ECE);
- selective risk and accepted-case false-negative rate over coverage.

Secondary:

- repeated stratified random-split estimates using the same model family;
- the paired difference between random-split and LOHO predictions where a
  defensible pairing is available;
- site predictability from predictors and missingness indicators;
- empirical conformal coverage and prediction-set size under hospital shift;
- robustness under prespecified missingness stress tests;
- model size and CPU latency.

Every reported metric must include sample size. Bootstrap confidence intervals
use patient-level resampling and fixed seeds; small or single-class resamples are
discarded and counted.

## Validation design

### E1: repeated random split

- 10 repeats of stratified 80/20 patient splits.
- Seeds are derived from root seed 20260729 and saved with predictions.
- All preprocessing, tuning, calibration, and threshold selection occur inside
  each training partition.
- This experiment is a comparator, not the primary external-validity result.

### E2: leave-one-hospital-out

Four outer folds, each holding out one full hospital. The held-out hospital is
never used for:

- imputer, encoder, scaler, or feature selection fitting;
- hyperparameter or model-family selection;
- probability calibration;
- conformal quantiles;
- OOD or missingness thresholds;
- classification or abstention threshold selection.

Within the three training hospitals, model selection uses group-aware inner
validation by hospital. If an inner training split cannot support a metric or
calibrator, the failure is logged; the held-out hospital is never used as a
fallback.

### Models

Prespecified model families:

1. logistic regression;
2. random forest;
3. histogram gradient boosting.

The primary model is selected by mean inner-hospital Brier score, with AUROC as
the first tie-breaker and model simplicity as the second. Hyperparameter grids
are intentionally small and declared in `configs/experiment.yaml`.

### Preprocessing

- Numeric: median imputation inside the training fold; standardization only for
  logistic regression.
- Categorical: most-frequent imputation plus one-hot encoding with unknown
  categories ignored.
- Missingness indicators are generated inside the imputer.
- `site`, `patient_id`, `num`, and `target` are explicitly rejected as predictors.

## Calibration, conformal prediction, and abstention

Calibration uses cross-fitted training-hospital predictions only. The primary
calibrator is sigmoid/Platt scaling. Isotonic calibration is excluded from the
primary analysis because individual training folds are small.

Class-conditional split conformal uses only training-hospital calibration
predictions. Under cross-hospital shift, only empirical coverage is claimed;
exchangeability-based guarantees are not claimed.

The abstention gate may use only prespecified training-derived signals:

- ambiguous calibrated probability;
- OOD score beyond a training-OOF quantile;
- non-singleton or empty conformal set;
- excessive missingness beyond a training-derived threshold.

Thresholds and operating points are fixed without viewing held-out outcomes.
The report shows the full risk-coverage curve, not only a favorable point.

## Robustness

Stress tests:

- MCAR masking at 10%, 30%, and 50%;
- no exercise variables;
- no ECG variables;
- basic vitals only.

Subgroups:

- sex;
- prespecified age bands `<50`, `50-59`, `60-69`, `>=70`.

Subgroup results always report `n`, event count, and confidence intervals. They
are descriptive; no unsupported fairness or clinical claims are permitted.

## Evidence and reporting lock

- Patient-level predictions are saved for every outer test observation.
- One canonical `outputs/results.json` supplies numbers to figures, notebook,
  app, README, and manuscript.
- All figures are code-generated.
- Negative and failed results are retained in `outputs/audit/`.
- No result language is written before its artifact exists.

## Prespecified failure conditions

Publication is blocked if:

- a held-out site enters any training decision;
- `site` enters a disease-prediction feature matrix;
- preprocessing is fit before the split;
- patient identifiers are not unique within site;
- any displayed number cannot be traced to `results.json`;
- a figure or case cannot be regenerated;
- claims imply future risk, treatment guidance, or clinical readiness.
