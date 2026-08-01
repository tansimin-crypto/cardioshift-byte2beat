# Ethics and Limitations

## Claim boundary

The data label records angiographic disease presence status. It is not a future
event, survival endpoint, or ten-year cardiovascular risk. CardioShift does not
provide a diagnosis, treatment recommendation, medication recommendation, or
clinical action.

## Why abstention is not automatically safety

At the prespecified operating point, the gate deferred
59.5% of held-out-hospital cases.
Accepted-case error was lower than the forced-answer error, but that retrospective
association does not establish patient benefit. A deployment study would need
to define who reviews deferred cases, review time, downstream harms, and whether
accepted errors concentrate in protected or clinically important groups.

## Distribution shift

Hospital identity was predictable with balanced accuracy
0.911. This supports the presence
of systematic recording and population differences; it does not identify their
causes. Provider identifiers and missingness are treated as evidence of
association, not intent.

## Conformal limitation

Overall empirical conformal coverage was
90.4%, but the worst site,
VA Long Beach, reached only
78.0%. Standard conformal
coverage relies on exchangeability, which is not automatically satisfied when
moving to a different hospital.

## Data and subgroup limitations

- Historical, small, non-contemporary data from the 1980s.
- Site outcome prevalence and missingness are extreme in some hospitals.
- The target is recorded angiographic disease presence, not a future event.
- No prospective clinical deployment or decision-impact study was performed.
- The safety gate defers a majority of cases at its prespecified operating point.
- Conformal coverage is empirical only under hospital shift; exchangeability is not assured.
- The IsolationForest threshold uses an outer-training in-sample score quantile and may under-detect subtle shift.
- Missingness and subgroup stress tests are descriptive and do not establish clinical robustness or fairness.
- Results do not support diagnosis, treatment, medication, or real-patient use.

Sex and age subgroup analyses and prespecified missingness stress tests are
reported with sample sizes and bootstrap uncertainty. They are descriptive
only and do not establish fairness or clinical robustness.
