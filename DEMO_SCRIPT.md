# CardioShift 2-3 minute judge demo

## 0:00-0:10 - Problem

"Random patient splits can mix all hospitals into training and testing. We ask
what happens when the next hospital is genuinely unseen."

## 0:10-0:35 - Same model, two realities

Show Figure 1 and state:

- random-split mean AUROC: **0.892**;
- leave-one-hospital-out pooled AUROC: **0.789**;
- optimism gap: **0.102**.

## 0:35-0:55 - Hospital shift

Show the prevalence/missingness and reliability figures. Hospital
classification balanced accuracy is
**0.911** versus
**0.25** balanced chance.

## 0:55-1:30 - Three fixed cases

1. **ACCEPT:** `va_long_beach:0068` from `VA Long Beach` is an
   accepted correct case.
2. **DEFER:** `va_long_beach:0070` from `VA Long Beach` is a
   deferred model error.
3. **Gate miss:** `va_long_beach:0141` from `VA Long Beach` is a
   confident error that remained ACCEPT.

These are deidentified retrospective research rows, not live patient inputs.

## 1:30-1:50 - Cost of deferral

The gate accepted **40.5%** and therefore
deferred **59.5%** of cases. Error among accepted cases was
**10.2%**.

## 1:50-2:05 - Conformal failure under shift

Worst-hospital empirical conformal coverage was **78.0%** in
**VA Long Beach**. The Demo labels this as a limitation, not a guarantee.

## 2:05-2:35 - Real Coder workspace (verified)

Show the recorded Coder evidence and, when available, the running local
workspace:

1. workspace build with GitHub authentication;
2. `.coder-status/tests.ok` and `.coder-status/services.ok`;
3. `.coder-status/release.sha` equal to
   `6b7aadd9806f13ebedd6a3be4b09e5d8a48c440b`;
4. JupyterLab 4.6.2 opened through Coder;
5. Streamlit health `ok` and Jupyter API HTTP 200;
6. the same exact SHA and health checks after stop/start recovery.

The public evidence is `evidence/g5/runtime_verification.json`.

## 2:35-2:45 - Close

"CardioShift is a retrospective research benchmark for knowing when model
confidence does not transfer. It is not for diagnosis, treatment, medication,
triage, or real-patient decisions."
