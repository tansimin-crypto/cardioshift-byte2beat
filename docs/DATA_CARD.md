# CardioShift Data Card

Generated from `data/audit.json` by `scripts/render_data_card.py`.

## Intended use

Retrospective methods research on cross-hospital validation, calibration,
dataset shift, and abstention for heart-disease presence classification.
Not for diagnosis, treatment, future-risk prediction, or real patient use.

## Source and license

- Dataset: UCI Heart Disease
- DOI: `10.24432/C52P4X`
- License: CC BY 4.0
- Source URL: <https://archive.ics.uci.edu/dataset/45/heart%2Bdisease>

## Cohort summary

- Rows: 920
- Binary target events: 509
- Binary target prevalence: 0.553
- Exact duplicate rows ignoring local ID: 4

| Site | n | Events | Prevalence | Missing fraction |
|---|---:|---:|---:|---:|
| Cleveland | 303 | 139 | 0.459 | 0.002 |
| Hungary | 294 | 106 | 0.361 | 0.205 |
| Switzerland | 123 | 115 | 0.935 | 0.171 |
| VA Long Beach | 200 | 149 | 0.745 | 0.268 |

## Missing values

Original `?` values are preserved as missing. No imputation, encoding,
scaling, feature selection, exclusion, or complete-case filtering occurs
during cohort construction.

| Feature | Missing count |
|---|---:|
| age | 0 |
| ca | 611 |
| chol | 30 |
| cp | 0 |
| exang | 55 |
| fbs | 90 |
| oldpeak | 62 |
| restecg | 2 |
| sex | 0 |
| slope | 309 |
| thal | 486 |
| thalach | 55 |
| trestbps | 59 |

## Outcome

`num` is retained as the source outcome. `target` is exactly `num > 0`.
It represents recorded angiographic disease presence status, not future
cardiovascular risk.

## Known limitations

- Historical, small, non-contemporary data.
- Site-specific prevalence and missingness can be extreme.
- The local `patient_id` is a deterministic source-row key, not a clinical ID.
- Exact duplicate rows are reported rather than silently deleted.
- Cross-hospital validation does not establish prospective clinical utility.
