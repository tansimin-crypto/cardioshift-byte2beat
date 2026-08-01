# PROBAST+AI Self-Audit

This is a conservative internal bias audit, not a validated PROBAST assessment.

| Domain | Judgment | Rationale |
|---|---|---|
| Participants and data sources | High concern | Historical convenience datasets; strong site prevalence and missingness differences |
| Predictors | Some concern | Many predictors are missing in entire historical workflows; timing details are limited |
| Outcome | Some concern | Angiographic status is documented, but cross-site measurement consistency is uncertain |
| Analysis | Some concern | Leakage controls and external validation are strong; sample size is small, tuning remains data-limited, and OOD threshold is in-sample |
| Applicability | High concern for clinical use | No prospective deployment, contemporary cohort, workflow study, or clinical threshold validation |

Risk-reducing design choices:

- hospital-level external validation;
- fold-local preprocessing and selection;
- training-only calibration and thresholds;
- full prediction and ledger retention;
- uncertainty intervals and explicit negative findings;
- no clinical, treatment, or future-risk claims.
