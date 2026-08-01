"""Render all judge-facing scientific text from outputs/results.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "outputs" / "results.json"


def percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def interval(values: list[float], *, percent_values: bool = False) -> str:
    if percent_values:
        return f"{percent(values[0])}–{percent(values[1])}"
    return f"{values[0]:.3f}–{values[1]:.3f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def render_readme(result: dict[str, object]) -> str:
    finding = result["key_findings"]
    commit = result["code_commit_before_canonicalization"]
    return f"""
# CardioShift — Know When Not to Predict

> A model that knows when to defer may be more useful than a model that is
> always confident.

CardioShift is a retrospective four-hospital benchmark asking whether strong
random-split heart-disease classification results transfer to a hospital the
model has never seen.

## Main result

The same bounded model search reached mean AUROC
**{finding['random_split_mean_auroc']:.3f}** across 10 random splits, but pooled
leave-one-hospital-out AUROC fell to **{finding['loho_pooled_auroc']:.3f}**
(95% patient bootstrap CI {interval(finding['loho_pooled_auroc_ci'])}): an
optimism gap of **{finding['random_minus_loho_auroc']:.3f}**.

Hospital identity was itself predictable from the recorded inputs with balanced
accuracy **{finding['site_classifier_balanced_accuracy']:.3f}** versus
{finding['site_classifier_chance']:.2f} balanced chance, confirming substantial
dataset shift.

The prespecified safety gate accepted
**{percent(finding['safety_gate_coverage'])}** of held-out-hospital cases and
had **{percent(finding['safety_gate_selective_error'])}** error among accepted
cases (95% CI {interval(finding['safety_gate_selective_error_ci'], percent_values=True)}).
This is a coverage–error tradeoff, not proof of clinical safety. The system
deferred most cases, and conformal empirical coverage fell to
{percent(finding['worst_site_conformal_coverage']['value'])} in
{finding['worst_site_conformal_coverage']['site']}.

## Reproduce

```powershell
python scripts/download_data.py
python scripts/build_cohort.py
python scripts/verify_gate_g1.py
python scripts/run_core_experiments.py
python scripts/verify_gate_g2.py
python scripts/run_shift_safety.py
python scripts/verify_shift_safety.py
python scripts/make_core_figures.py
python scripts/build_canonical_results.py
python scripts/render_scientific_docs.py
python -m unittest discover -s tests -v
```

Accepted evidence commits:

- Gate G1: `222523a`
- Gate G2: `c4301e3`
- E3/E5: `af35b24`
- Figures: `a54aec5`
- Canonical source base: `{commit}`

## Scope

- Outcome: recorded angiographic heart-disease presence status.
- Primary validation: leave-one-hospital-out across Cleveland, Hungary,
  Switzerland, and VA Long Beach.
- Not future-risk prediction.
- Not a clinical device and not for diagnosis, treatment, medication, or real
  patient decisions.

## Data

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989),
*Heart Disease*, UCI Machine Learning Repository,
<https://doi.org/10.24432/C52P4X>, CC BY 4.0.

See `competition-contract.yaml`, `EXPERIMENT_PROTOCOL.md`,
`outputs/results.json`, and `report/manuscript.md`.
"""


def render_model_card(result: dict[str, object]) -> str:
    finding = result["key_findings"]
    return f"""
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
| 10 repeated random splits, mean | {finding['random_split_mean_auroc']:.3f} | {finding['random_split_mean_brier']:.3f} |
| Pooled leave-one-hospital-out | {finding['loho_pooled_auroc']:.3f} | {finding['loho_pooled_brier']:.3f} |

Random-minus-LOHO AUROC: {finding['random_minus_loho_auroc']:.3f}.

The safety gate accepted {percent(finding['safety_gate_coverage'])} of cases.
Accepted-case error was {percent(finding['safety_gate_selective_error'])}; this
must be interpreted jointly with the {percent(1 - finding['safety_gate_coverage'])}
deferral rate.

## Safety components

- sigmoid calibration from training-hospital OOF probabilities;
- class-conditional conformal sets, reported as empirical coverage only;
- IsolationForest OOD score on a training-fitted representation;
- probability ambiguity and excessive-missingness gates.

## Limitations

{chr(10).join(f'- {item}' for item in result['limitations'])}
"""


def render_ethics(result: dict[str, object]) -> str:
    finding = result["key_findings"]
    return f"""
# Ethics and Limitations

## Claim boundary

The data label records angiographic disease presence status. It is not a future
event, survival endpoint, or ten-year cardiovascular risk. CardioShift does not
provide a diagnosis, treatment recommendation, medication recommendation, or
clinical action.

## Why abstention is not automatically safety

At the prespecified operating point, the gate deferred
{percent(1 - finding['safety_gate_coverage'])} of held-out-hospital cases.
Accepted-case error was lower than the forced-answer error, but that retrospective
association does not establish patient benefit. A deployment study would need
to define who reviews deferred cases, review time, downstream harms, and whether
accepted errors concentrate in protected or clinically important groups.

## Distribution shift

Hospital identity was predictable with balanced accuracy
{finding['site_classifier_balanced_accuracy']:.3f}. This supports the presence
of systematic recording and population differences; it does not identify their
causes. Provider identifiers and missingness are treated as evidence of
association, not intent.

## Conformal limitation

Overall empirical conformal coverage was
{percent(finding['empirical_conformal_coverage'])}, but the worst site,
{finding['worst_site_conformal_coverage']['site']}, reached only
{percent(finding['worst_site_conformal_coverage']['value'])}. Standard conformal
coverage relies on exchangeability, which is not automatically satisfied when
moving to a different hospital.

## Data and subgroup limitations

{chr(10).join(f'- {item}' for item in result['limitations'])}

Sex and age subgroup analyses and prespecified missingness stress tests are
reported with sample sizes and bootstrap uncertainty. They are descriptive
only and do not establish fairness or clinical robustness.
"""


def render_reproducibility(result: dict[str, object]) -> str:
    return f"""
# Reproducibility

## Frozen inputs

- Standardized cohort SHA-256: `{result['data']['input_sha256']}`
- Rows: {result['data']['rows']}
- Sites: {', '.join(result['data']['sites'])}
- Root seed: {result['experiments']['E1_E2']['reproducibility']['root_seed']}
- Candidate configurations: {result['experiments']['E1_E2']['reproducibility']['candidate_count']}

## Evidence chain

1. `competition-contract.yaml` freezes submission and scientific constraints.
2. `EXPERIMENT_PROTOCOL.md` prespecifies validation and failure conditions.
3. `data/checksums.json` hashes official source files and the standardized
   cohort.
4. `outputs/predictions/loho_predictions.csv` contains one external prediction
   per source row.
5. `outputs/audit/loho_training_ledgers.json` records the exact patient IDs used
   for tuning, calibration, and final fitting.
6. `outputs/results.json` is the only judge-facing numeric source.
7. `outputs/figures/figure_manifest.json` maps every figure to source hashes.

## Commands

Run the command sequence in `README.md`. Every verification script fails with a
non-zero exit code on a violated invariant. Current clean-environment and Kaggle
Run All validation remain pending, so Gate G4 is not yet claimed.

## Artifact hashes

{chr(10).join(f"- `{path}`: `{digest}`" for path, digest in result['artifact_hashes'].items())}
"""


def render_tripod() -> str:
    return """
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
"""


def render_probast() -> str:
    return """
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
"""


def render_manuscript(result: dict[str, object]) -> str:
    finding = result["key_findings"]
    loho = result["experiments"]["E1_E2"]["E2_leave_one_hospital_out"]["pooled"]
    return f"""
# CardioShift: Know When Not to Predict

## A leakage-resistant four-hospital benchmark of calibration and selective prediction

### Abstract

Random patient splitting can evaluate interpolation within a mixture of data
sources rather than transport to a new hospital. We compared 10 repeated
stratified random splits with four leave-one-hospital-out (LOHO) evaluations on
920 records from the UCI Heart Disease collection. Every outer fold restricted
preprocessing, model selection, calibration, conformal quantiles, OOD thresholds,
and abstention thresholds to the three training hospitals. Mean random-split
AUROC was {finding['random_split_mean_auroc']:.3f}; pooled LOHO AUROC was
{finding['loho_pooled_auroc']:.3f} (95% patient bootstrap CI
{interval(finding['loho_pooled_auroc_ci'])}), an optimism gap of
{finding['random_minus_loho_auroc']:.3f}. A separate site classifier achieved
balanced accuracy {finding['site_classifier_balanced_accuracy']:.3f}, supporting
substantial source shift. A prespecified safety gate accepted
{percent(finding['safety_gate_coverage'])} of external cases with accepted-case
error {percent(finding['safety_gate_selective_error'])} (95% CI
{interval(finding['safety_gate_selective_error_ci'], percent_values=True)}).
However, the gate deferred most cases, and empirical conformal coverage fell to
{percent(finding['worst_site_conformal_coverage']['value'])} in
{finding['worst_site_conformal_coverage']['site']}. CardioShift therefore shows
both the scale of random-split optimism and the limits of algorithmic
abstention under real hospital shift.

### 1. Clinical question and intended use

The research question is not whether another classifier can improve internal
accuracy. It is whether performance, calibration, and confidence transfer when
the test hospital is absent from every model-development decision. The intended
use is retrospective methods research and education. The endpoint is recorded
angiographic disease presence status; it is not future cardiovascular risk.

### 2. Data

The analysis uses Cleveland (303 records), Hungary (294), Switzerland (123), and
VA Long Beach (200). The combined cohort contains 920 records and 509 positive
labels. Original missing values were retained. Hospital prevalence ranged from
36.1% to 93.5%, while the fraction of missing predictor cells ranged from 0.15%
to 26.85%. Four exact duplicate rows were reported and retained.

### 3. Validation protocol

The primary design was four-fold LOHO validation. Within each outer training
set, candidate logistic regression, random forest, and histogram gradient
boosting configurations were compared by leave-one-training-hospital-out mean
Brier score, with AUROC and simplicity as tie-breakers. Numeric and categorical
imputation, scaling where applicable, and encoding were fitted inside each
training fold. The `site` field and all identifiers and outcomes were excluded
from predictor matrices.

Sigmoid calibration used group-OOF probabilities from the three training
hospitals. Patient-level predictions and exact tuning, calibration, and final-fit
ID ledgers were saved. Metrics used 2,000 patient bootstrap resamples.

### 4. Random splitting versus hospital holdout

Across repeated random splits, mean AUROC was
{finding['random_split_mean_auroc']:.3f} and mean Brier score was
{finding['random_split_mean_brier']:.3f}. Pooled LOHO AUROC was
{finding['loho_pooled_auroc']:.3f}, and Brier score worsened to
{finding['loho_pooled_brier']:.3f}. Pooled LOHO sensitivity was
{percent(loho['sensitivity'])}, specificity {percent(loho['specificity'])}, and
ECE {loho['ece_10']:.3f}. The four held-out-hospital AUROCs ranged from
0.727 to 0.891.

These results support a narrower conclusion than “random splits are invalid.”
In this four-source dataset, random patient splitting produced substantially
more optimistic estimates than hospital-level external validation.

![Same model, two realities](../outputs/figures/01_same_model_two_realities.png)

### 5. Evidence of dataset shift

Hospital identity was predicted from patient predictors and missingness with
5-fold OOF balanced accuracy {finding['site_classifier_balanced_accuracy']:.3f}
versus balanced chance {finding['site_classifier_chance']:.2f}. This does not
establish the causes of shift, but it demonstrates that the recorded source
distributions are strongly distinguishable.

![Prevalence and missingness](../outputs/figures/02_prevalence_missingness.png)

### 6. Calibration

Calibration varied sharply by held-out hospital even after training-only sigmoid
calibration. Per-hospital ECE ranged from approximately 0.186 to 0.289. This
motivates reporting probability quality separately from ranking performance.

![Per-site reliability](../outputs/figures/03_per_site_reliability.png)

### 7. Selective prediction

The prespecified gate deferred cases when any of four training-derived signals
triggered: OOD score, ambiguity band, non-singleton or empty conformal set, or
excessive missingness. It accepted {percent(finding['safety_gate_coverage'])} of
LOHO cases. Accepted-case error was
{percent(finding['safety_gate_selective_error'])}; accepted-case false-negative
rate was {percent(finding['accepted_case_fnr'])}.

This improvement cannot be interpreted without the
{percent(1 - finding['safety_gate_coverage'])} deferral rate. The gate trades
coverage for lower error among accepted cases; it does not prove clinical safety.

![Risk coverage](../outputs/figures/04_risk_coverage.png)

### 8. Conformal and OOD limitations

Overall empirical conformal coverage was
{percent(finding['empirical_conformal_coverage'])}, but
{finding['worst_site_conformal_coverage']['site']} reached only
{percent(finding['worst_site_conformal_coverage']['value'])}. Exchangeability
does not automatically survive a hospital shift. The OOD detector also used an
outer-training in-sample threshold, which can under-detect subtle changes.

### 9. Failure case

The held-out predictions contain real high-confidence errors. The safety gate
catches some through OOD, conformal, ambiguity, or missingness triggers, but not
all. The case visual is illustrative and does not imply patient benefit.

![Confident error deferred](../outputs/figures/05_confident_error_deferred.png)

### 10. Limitations

{chr(10).join(f'- {item}' for item in result['limitations'])}

### 11. Conclusion

Random patient splitting overstated cross-hospital performance in this
four-center benchmark. Training-only calibration and abstention reduced error
among accepted cases but at the cost of deferring most cases, and conformal
coverage remained fragile in the most shifted hospital. The practical lesson is
not that uncertainty tools guarantee safety; it is that a model should be
evaluated where it is expected to fail, and allowed to expose uncertainty rather
than forced to answer every time.
"""


def main() -> None:
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    write(ROOT / "README.md", render_readme(result))
    write(ROOT / "docs" / "MODEL_CARD.md", render_model_card(result))
    write(
        ROOT / "docs" / "ETHICS_AND_LIMITATIONS.md",
        render_ethics(result),
    )
    write(ROOT / "docs" / "REPRODUCIBILITY.md", render_reproducibility(result))
    write(ROOT / "docs" / "TRIPOD_AI_CHECKLIST.md", render_tripod())
    write(ROOT / "docs" / "PROBAST_AI_AUDIT.md", render_probast())
    write(ROOT / "report" / "manuscript.md", render_manuscript(result))
    print("rendered README, model/ethics/reproducibility docs, checklists, manuscript")


if __name__ == "__main__":
    main()
