# TRIPOD+AI Checklist Status

This is a working self-audit, not a claim of formal endorsement.

| Area | Status | Evidence |
|---|---|---|
| Background and objective | Complete | `EXPERIMENT_PROTOCOL.md` |
| Intended use and target | Complete | `competition-contract.yaml`, Model Card |
| Data source and eligibility | Partial | UCI source documented; historical collection details limited |
| Outcome definition | Complete | `num == 0` versus `num > 0` |
| Predictors | Complete | `data/data_dictionary.md` |
| Missing data | Complete for ingestion | Preserved; fold-local imputation |
| Sample size | Descriptive | Full available four-center cohort; no prospective calculation |
| Model development | Complete | `src/modeling.py` |
| Validation | Complete | Four-fold LOHO plus repeated random split |
| Performance measures | Complete | discrimination, calibration, classification, selective metrics |
| Uncertainty | Complete for primary metrics | 2,000 patient bootstrap replicates |
| Model specification | Partial | Fold-specific fitted objects are not yet exported |
| Participant flow | Partial | No exclusions; duplicate rows reported and retained |
| Limitations | Complete | `docs/ETHICS_AND_LIMITATIONS.md` |
| Registration/protocol | Project-local | `EXPERIMENT_PROTOCOL.md` predates outcome-bearing runs |
| Data/code access | Pending public release | Local evidence exists; public links not yet verified |
