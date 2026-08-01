"""Render the human-readable Data Card from the canonical data audit."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "audit.json"
OUTPUT = ROOT / "docs" / "DATA_CARD.md"


def render(audit: dict[str, object]) -> str:
    site_rows = [
        (
            f"| {site} | {values['n']} | {values['events']} | "
            f"{values['prevalence']:.3f} | {values['missing_fraction']:.3f} |"
        )
        for site, values in audit["sites"].items()
    ]
    missing_rows = [
        f"| {feature} | {count} |"
        for feature, count in audit["missing_by_feature"].items()
    ]
    return "\n".join(
        [
            "# CardioShift Data Card",
            "",
            "Generated from `data/audit.json` by `scripts/render_data_card.py`.",
            "",
            "## Intended use",
            "",
            "Retrospective methods research on cross-hospital validation, calibration,",
            "dataset shift, and abstention for heart-disease presence classification.",
            "Not for diagnosis, treatment, future-risk prediction, or real patient use.",
            "",
            "## Source and license",
            "",
            "- Dataset: UCI Heart Disease",
            "- DOI: `10.24432/C52P4X`",
            "- License: CC BY 4.0",
            "- Source URL: <https://archive.ics.uci.edu/dataset/45/heart%2Bdisease>",
            "",
            "## Cohort summary",
            "",
            f"- Rows: {audit['rows']}",
            f"- Binary target events: {audit['target_events']}",
            f"- Binary target prevalence: {audit['target_prevalence']:.3f}",
            (
                "- Exact duplicate rows ignoring local ID: "
                f"{audit['exact_duplicate_rows_ignoring_patient_id']}"
            ),
            "",
            "| Site | n | Events | Prevalence | Missing fraction |",
            "|---|---:|---:|---:|---:|",
            *site_rows,
            "",
            "## Missing values",
            "",
            "Original `?` values are preserved as missing. No imputation, encoding,",
            "scaling, feature selection, exclusion, or complete-case filtering occurs",
            "during cohort construction.",
            "",
            "| Feature | Missing count |",
            "|---|---:|",
            *missing_rows,
            "",
            "## Outcome",
            "",
            "`num` is retained as the source outcome. `target` is exactly `num > 0`.",
            "It represents recorded angiographic disease presence status, not future",
            "cardiovascular risk.",
            "",
            "## Known limitations",
            "",
            "- Historical, small, non-contemporary data.",
            "- Site-specific prevalence and missingness can be extreme.",
            "- The local `patient_id` is a deterministic source-row key, not a clinical ID.",
            "- Exact duplicate rows are reported rather than silently deleted.",
            "- Cross-hospital validation does not establish prospective clinical utility.",
            "",
        ]
    )


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "pass":
        raise RuntimeError("data audit is not passing")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(audit), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
