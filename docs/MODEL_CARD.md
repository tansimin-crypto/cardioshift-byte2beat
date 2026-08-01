# Model Card: CardioShift

## Model details

CardioShift is not one universal fitted model. It is a four-fold external
validation benchmark. In each outer fold, one hospital is held out; model family
and hyperparameters are selected only among the other three hospitals by
leave-one-training-hospital-out Brier score. Candidate families are logistic
regression, random forest, and histogram gradient boosting.

All imputation, encoding, scaling, model selection, sigmoid calibration,
conformal quantiles, OOD thresholds, and abstention thresholds are fit without
the held-out hospital. `site`, `patient_id`, `num`, and `target` are forbidden
prediction inputs.

## Intended use

Retrospective research and education about transportability, calibration,
dataset shift, and selective prediction.

## Out-of-scope use

- diagnosis or treatment;
- future or ten-year cardiovascular risk;
- medication or lifestyle recommendations;
- real-patient decision support;
- claims of clinical-grade, regulatory-ready, or guaranteed-safe performance.

## Performance

| Setting | AUROC | Brier |
|---|---:|---:|
| 10 repeated random splits, mean | 0.892 | 0.129 |
| Pooled leave-one-hospital-out | 0.789 | 0.190 |

Random-minus-LOHO AUROC: 0.102.

The safety gate accepted 40.5% of cases.
Accepted-case error was 10.2%; this
must be interpreted jointly with the 59.5%
deferral rate.

## Safety components

- sigmoid calibration from training-hospital OOF probabilities;
- class-conditional conformal sets, reported as empirical coverage only;
- IsolationForest OOD score on a training-fitted representation;
- probability ambiguity and excessive-missingness gates.

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
