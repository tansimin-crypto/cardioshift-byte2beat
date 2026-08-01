"""Evidence-backed Streamlit experience for CardioShift judges."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.demo_cases import select_demo_cases
from src.results_access import ResultsAccessor

ROOT = Path(__file__).resolve().parent
PAGES = (
    "The Result",
    "Why Hospitals Differ",
    "Know When Not to Predict",
    "Robustness and Limitations",
    "Reproduce",
)


@st.cache_resource
def accessor() -> ResultsAccessor:
    return ResultsAccessor(ROOT)


def metric_row(results: ResultsAccessor) -> None:
    finding = results.findings
    columns = st.columns(3)
    columns[0].metric(
        "Random-split AUROC",
        f"{finding['random_split_mean_auroc']:.3f}",
    )
    columns[1].metric("LOHO AUROC", f"{finding['loho_pooled_auroc']:.3f}")
    columns[2].metric(
        "Optimism gap",
        f"{finding['random_minus_loho_auroc']:.3f}",
    )


def page_result(results: ResultsAccessor) -> None:
    st.header("The Result")
    metric_row(results)
    st.info(
        "Random splitting estimates performance inside a mixed hospital "
        "distribution. It is not validation for transport to a new hospital."
    )
    hospital = results.per_hospital().set_index("hospital")
    st.subheader("Leave-one-hospital-out performance")
    st.dataframe(hospital, use_container_width=True)
    st.bar_chart(hospital["auroc"])


def page_shift(results: ResultsAccessor) -> None:
    st.header("Why Hospitals Differ")
    shift = results.canonical["experiments"]["E3_E5"]["E3_shift"]
    finding = results.findings
    st.metric(
        "Hospital classifier balanced accuracy",
        f"{finding['site_classifier_balanced_accuracy']:.3f}",
        help=(
            "A dataset-shift diagnostic only. This classifier is not a "
            "clinical model."
        ),
    )
    st.caption(
        f"Balanced chance is {finding['site_classifier_chance']:.2f}. "
        "Predictable site identity indicates that recorded inputs differ by center."
    )
    st.subheader("Missingness by hospital")
    st.dataframe(pd.DataFrame(shift["missingness_by_site"]).T)
    st.subheader("Standardized mean differences")
    rows = []
    for hospital, features in shift["standardized_mean_differences"].items():
        for feature, record in features.items():
            rows.append(
                {
                    "hospital": hospital,
                    "feature": feature,
                    "SMD vs rest": record["smd_site_vs_rest"],
                }
            )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def page_cases(results: ResultsAccessor) -> None:
    st.header("Know When Not to Predict")
    st.warning(
        "These are fixed, deidentified historical research records—not a form "
        "for entering health information."
    )
    cases = select_demo_cases(ROOT)
    labels = {
        "accepted_confident_correct": "Accepted, confident, correct",
        "deferred_confident_error": "Deferred, confident error",
        "confident_error_not_caught_by_gate": "Confident error not caught by gate",
        "heavy_missingness_case": "Heavy-missingness case",
    }
    for key, title in labels.items():
        case = cases[key]
        with st.expander(title, expanded=key == "confident_error_not_caught_by_gate"):
            left, right = st.columns(2)
            left.write(
                {
                    "case_id": case["case_id"],
                    "hospital": case["hospital"],
                    "model_probability": case["model_probability"],
                    "prediction_set": case["prediction_set"],
                    "decision": case["decision"],
                }
            )
            right.write(
                {
                    "OOD flag": case["ood_flag"],
                    "ambiguity flag": case["ambiguity_flag"],
                    "missingness flag": case["missingness_flag"],
                    "actual research label": case["research_label"],
                    "prediction correct": case["correct"],
                }
            )
    finding = results.findings
    st.caption(
        "The actual label is shown only to explain retrospective failure modes. "
        f"Overall gate coverage was {finding['safety_gate_coverage']:.1%}."
    )


def page_robustness(results: ResultsAccessor) -> None:
    st.header("Robustness and Limitations")
    summary = results.robustness_summary().set_index("scenario")
    st.subheader("Missingness stress tests")
    st.dataframe(summary, use_container_width=True)
    st.line_chart(summary[["auroc", "safety_coverage"]])
    st.subheader("Subgroup sample sizes")
    subgroups = results.subgroup_summary().set_index("subgroup")
    st.dataframe(
        subgroups[["n", "events", "auroc", "safety_coverage", "interpretation"]]
    )
    finding = results.findings
    st.error(
        f"Worst-site empirical conformal coverage occurred in "
        f"{finding['worst_site_conformal_coverage']['site']}: "
        f"{finding['worst_site_conformal_coverage']['value']:.1%}. "
        "The safety method itself can fail under hospital shift."
    )
    st.write(
        f"The gate deferred {1 - finding['safety_gate_coverage']:.1%} of records."
    )
    for limitation in results.limitations:
        st.write(f"- {limitation}")


def page_reproduce(results: ResultsAccessor) -> None:
    st.header("Reproduce")
    canonical = results.canonical
    st.code(
        "\n".join(
            [
                "python -m pip install -r requirements.lock",
                "python scripts/verify_gate_g1.py",
                "python scripts/verify_gate_g2.py",
                "python scripts/verify_shift_safety.py",
                "python scripts/verify_robustness.py",
                "python -m pytest -q",
                "streamlit run app.py",
            ]
        ),
        language="bash",
    )
    st.write(
        {
            "commit recorded by canonical builder": canonical[
                "code_commit_before_canonicalization"
            ],
            "data SHA-256": canonical["data"]["input_sha256"],
            "results SHA-256": results.sha256("outputs/results.json"),
            "Notebook": "notebooks/CardioShift_Research_Report.ipynb",
            "gate status": results.gates,
        }
    )
    st.success(
        "Coder runtime verification passed for the audited release: locked tests, "
        "JupyterLab, Streamlit, exact-SHA checkout, and stop/start recovery. "
        "See evidence/g5/runtime_verification.json."
    )
    st.error(
        "Research use only. Not for diagnosis, treatment, medication, triage, "
        "or real-patient decisions."
    )


def main() -> None:
    st.set_page_config(page_title="CardioShift", layout="wide")
    st.title("CardioShift")
    st.caption("Know When Not to Predict")
    page = st.sidebar.selectbox("Judge path", PAGES)
    results = accessor()
    handlers = {
        "The Result": page_result,
        "Why Hospitals Differ": page_shift,
        "Know When Not to Predict": page_cases,
        "Robustness and Limitations": page_robustness,
        "Reproduce": page_reproduce,
    }
    handlers[page](results)


if __name__ == "__main__":
    main()
