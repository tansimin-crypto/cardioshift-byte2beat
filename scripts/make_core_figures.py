"""Generate the six prespecified judge-facing figures from accepted artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "outputs" / "metrics" / "results.json"
SAFETY_PATH = ROOT / "outputs" / "metrics" / "shift_safety_results.json"
LOHO_PATH = ROOT / "outputs" / "predictions" / "loho_predictions.csv"
SAFETY_PREDICTIONS_PATH = (
    ROOT / "outputs" / "predictions" / "safety_loho_predictions.csv"
)
COHORT_PATH = ROOT / "data" / "processed" / "cardioshift_cohort.csv"
FIGURE_DIR = ROOT / "outputs" / "figures"
MANIFEST_PATH = FIGURE_DIR / "figure_manifest.json"

COLORS = {
    "random": "#4C78A8",
    "loho": "#E45756",
    "safe": "#2A9D8F",
    "muted": "#6B7280",
    "ink": "#17202A",
}


def sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {
        ".json",
        ".csv",
        ".md",
        ".py",
        ".yaml",
        ".yml",
        ".tf",
        ".sh",
        ".ipynb",
    }:
        content = content.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def save_figure(figure: plt.Figure, filename: str) -> Path:
    path = FIGURE_DIR / filename
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def figure_one(core: dict[str, object]) -> Path:
    metrics = [
        ("auroc", "AUROC", "higher is better"),
        ("auprc", "AUPRC", "higher is better"),
        ("brier", "Brier score", "lower is better"),
        ("ece_10", "ECE (10 bins)", "lower is better"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
    for axis, (key, title, direction) in zip(axes, metrics):
        random_summary = core["E1_repeated_random_split"]["summary"][key]
        loho = core["E2_leave_one_hospital_out"]["pooled"]
        random_value = random_summary["mean"]
        loho_value = loho[key]
        loho_ci = loho["bootstrap"]["metrics"][key]
        random_error = np.asarray(
            [
                [random_value - random_summary["repeat_percentile_2_5"]],
                [random_summary["repeat_percentile_97_5"] - random_value],
            ]
        )
        loho_error = np.asarray(
            [
                [loho_value - loho_ci["low"]],
                [loho_ci["high"] - loho_value],
            ]
        )
        axis.bar(
            [0, 1],
            [random_value, loho_value],
            color=[COLORS["random"], COLORS["loho"]],
            width=0.65,
        )
        axis.errorbar(
            [0],
            [random_value],
            yerr=random_error,
            fmt="none",
            ecolor=COLORS["ink"],
            capsize=4,
            lw=1.2,
        )
        axis.errorbar(
            [1],
            [loho_value],
            yerr=loho_error,
            fmt="none",
            ecolor=COLORS["ink"],
            capsize=4,
            lw=1.2,
        )
        axis.set_xticks([0, 1], ["Random\nsplit", "Hospital\nholdout"])
        axis.set_title(title, fontsize=11, weight="bold")
        axis.text(
            0.5,
            1.02,
            direction,
            transform=axis.transAxes,
            ha="center",
            fontsize=8,
            color=COLORS["muted"],
        )
        axis.grid(axis="y", alpha=0.2)
        for position, value in enumerate((random_value, loho_value)):
            axis.text(
                position,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
            )
    figure.suptitle(
        "Same model search, two realities",
        fontsize=16,
        weight="bold",
        color=COLORS["ink"],
        y=1.08,
    )
    figure.text(
        0.5,
        -0.03,
        "Random: 10-repeat distribution interval. Hospital holdout: 95% patient bootstrap CI.",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    return save_figure(figure, "01_same_model_two_realities.png")


def figure_two(cohort: pd.DataFrame) -> Path:
    features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]
    sites = cohort["site"].drop_duplicates().tolist()
    prevalence = cohort.groupby("site", sort=False)["target"].mean().reindex(sites)
    missing = (
        cohort.groupby("site", sort=False)[features]
        .apply(lambda frame: frame.isna().mean())
        .reindex(sites)
    )
    figure = plt.figure(figsize=(12, 7))
    grid = figure.add_gridspec(2, 1, height_ratios=[1, 2.2], hspace=0.35)
    axis_top = figure.add_subplot(grid[0])
    axis_bottom = figure.add_subplot(grid[1])
    bars = axis_top.bar(
        sites,
        prevalence.values,
        color=["#4C78A8", "#72B7B2", "#F2CF5B", "#E45756"],
    )
    axis_top.set_ylim(0, 1)
    axis_top.set_ylabel("Disease-presence prevalence")
    axis_top.set_title(
        "Hospitals differ in outcome prevalence and measurement patterns",
        fontsize=15,
        weight="bold",
    )
    axis_top.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, prevalence.values):
        axis_top.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.1%}",
            ha="center",
            weight="bold",
        )
    image = axis_bottom.imshow(
        missing.to_numpy(),
        cmap="magma_r",
        aspect="auto",
        vmin=0,
        vmax=max(0.55, float(missing.max().max())),
    )
    axis_bottom.set_yticks(range(len(sites)), sites)
    axis_bottom.set_xticks(
        range(len(features)),
        features,
        rotation=45,
        ha="right",
    )
    axis_bottom.set_title("Missing fraction by hospital and predictor")
    for row in range(len(sites)):
        for column in range(len(features)):
            value = missing.iloc[row, column]
            if value >= 0.05:
                axis_bottom.text(
                    column,
                    row,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.3 else "black",
                )
    colorbar = figure.colorbar(image, ax=axis_bottom, fraction=0.025, pad=0.02)
    colorbar.set_label("Missing fraction")
    return save_figure(figure, "02_prevalence_missingness.png")


def reliability_points(frame: pd.DataFrame, bins: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values("calibrated_probability")
    groups = pd.qcut(
        ordered["calibrated_probability"],
        q=min(bins, len(ordered)),
        duplicates="drop",
    )
    summary = ordered.groupby(groups, observed=True).agg(
        predicted=("calibrated_probability", "mean"),
        observed=("target", "mean"),
        n=("target", "size"),
    )
    return (
        summary["predicted"].to_numpy(),
        summary["observed"].to_numpy(),
        summary["n"].to_numpy(),
    )


def figure_three(loho: pd.DataFrame) -> Path:
    sites = loho["site"].drop_duplicates().tolist()
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for axis, site in zip(axes.ravel(), sites):
        predicted, observed, n = reliability_points(loho.loc[loho["site"] == site])
        axis.plot([0, 1], [0, 1], "--", color=COLORS["muted"], lw=1)
        axis.plot(
            predicted,
            observed,
            marker="o",
            color=COLORS["loho"],
            lw=2,
        )
        for x_value, y_value, count in zip(predicted, observed, n):
            axis.text(x_value, y_value, f" {count}", fontsize=7)
        axis.set_title(site, weight="bold")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    figure.supxlabel("Mean calibrated probability")
    figure.supylabel("Observed event fraction")
    figure.suptitle(
        "Calibration does not transfer uniformly across hospitals",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Labels show patients per quantile bin; diagonal denotes ideal calibration.",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    return save_figure(figure, "03_per_site_reliability.png")


def figure_four(safety: dict[str, object]) -> Path:
    curve = pd.DataFrame(safety["E5_safety"]["risk_coverage_curve_pooled"])
    pooled = safety["E5_safety"]["pooled"]
    figure, axis = plt.subplots(figsize=(7.6, 5))
    axis.plot(
        curve["empirical_coverage"],
        curve["selective_risk"],
        marker="o",
        color=COLORS["safe"],
        lw=2.5,
        label="Outcome-blind uncertainty ranking",
    )
    axis.scatter(
        [pooled["coverage"]],
        [pooled["selective_risk"]],
        s=100,
        color=COLORS["loho"],
        zorder=5,
        label="Prespecified safety gate",
    )
    axis.annotate(
        f"gate: {pooled['coverage']:.1%} coverage\n{pooled['selective_risk']:.1%} error",
        (pooled["coverage"], pooled["selective_risk"]),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"]},
    )
    axis.set_xlabel("Accepted-case coverage")
    axis.set_ylabel("Error among accepted cases")
    axis.set_title(
        "Abstention trades coverage for lower accepted-case error",
        fontsize=14,
        weight="bold",
    )
    axis.grid(alpha=0.25)
    axis.set_xlim(0.05, 1.02)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    return save_figure(figure, "04_risk_coverage.png")


def failure_reasons(row: pd.Series) -> list[str]:
    reasons = []
    if bool(row["is_ood"]):
        reasons.append("OOD score above training threshold")
    if bool(row["is_ambiguous"]):
        reasons.append("probability in ambiguity band")
    if int(row["conformal_set_size"]) != 1:
        reasons.append(f"conformal set = {row['conformal_set'] or 'empty'}")
    if row["missing_fraction"] > row["missing_threshold"]:
        reasons.append("missingness above training threshold")
    return reasons


def figure_five(safety_predictions: pd.DataFrame) -> Path:
    prediction = (safety_predictions["calibrated_probability"] >= 0.5).astype(int)
    wrong = prediction != safety_predictions["target"]
    confidence = np.abs(safety_predictions["calibrated_probability"] - 0.5) * 2
    candidates = safety_predictions.loc[
        wrong & ~safety_predictions["accepted"].astype(bool)
    ].copy()
    candidates["confidence"] = confidence.loc[candidates.index]
    row = candidates.sort_values("confidence", ascending=False).iloc[0]
    reasons = failure_reasons(row)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    naive, safe = axes
    naive.axis("off")
    safe.axis("off")
    predicted_label = int(row["calibrated_probability"] >= 0.5)
    naive.text(0.02, 0.92, "Always-answer model", fontsize=14, weight="bold")
    naive.text(
        0.02,
        0.72,
        f"{row['calibrated_probability']:.1%} probability of class 1",
        fontsize=18,
        weight="bold",
        color=COLORS["loho"],
    )
    naive.text(0.02, 0.54, f"Forced prediction: {predicted_label}", fontsize=12)
    naive.text(0.02, 0.42, f"Recorded outcome: {int(row['target'])}", fontsize=12)
    naive.text(
        0.02,
        0.25,
        "Confident and wrong",
        fontsize=16,
        weight="bold",
        color=COLORS["loho"],
    )
    safe.text(0.02, 0.92, "CardioShift gate", fontsize=14, weight="bold")
    safe.text(
        0.02,
        0.72,
        "DEFER",
        fontsize=24,
        weight="bold",
        color=COLORS["safe"],
    )
    safe.text(
        0.02,
        0.58,
        "\n".join(f"• {reason}" for reason in reasons),
        fontsize=11,
        va="top",
    )
    safe.text(
        0.02,
        0.14,
        f"De-identified source row: {row['patient_id']} ({row['site']})",
        fontsize=8,
        color=COLORS["muted"],
    )
    figure.suptitle(
        "A real held-out-hospital failure the safety gate catches",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Illustrative retrospective error, not evidence of clinical benefit.",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    return save_figure(figure, "05_confident_error_deferred.png")


def figure_six(core: dict[str, object]) -> Path:
    records = core["E2_leave_one_hospital_out"]["by_site"]
    sites = [record["held_out_site"] for record in records]
    auroc = [record["metrics"]["auroc"] for record in records]
    brier = [record["metrics"]["brier"] for record in records]
    pooled = core["E2_leave_one_hospital_out"]["pooled"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].barh(sites, auroc, color=COLORS["random"])
    axes[0].axvline(
        pooled["auroc"],
        color=COLORS["ink"],
        linestyle="--",
        label=f"pooled {pooled['auroc']:.3f}",
    )
    axes[0].set_xlim(0.5, 1)
    axes[0].set_title("AUROC")
    axes[0].legend(frameon=False)
    axes[1].barh(sites, brier, color=COLORS["loho"])
    axes[1].axvline(
        pooled["brier"],
        color=COLORS["ink"],
        linestyle="--",
        label=f"pooled {pooled['brier']:.3f}",
    )
    axes[1].set_title("Brier score")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Pooled performance hides the weakest hospital",
        fontsize=15,
        weight="bold",
    )
    return save_figure(figure, "06_worst_hospital_vs_pooled.png")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
        }
    )
    core = json.loads(CORE_PATH.read_text(encoding="utf-8"))
    safety = json.loads(SAFETY_PATH.read_text(encoding="utf-8"))
    cohort = pd.read_csv(COHORT_PATH)
    loho = pd.read_csv(LOHO_PATH)
    safety_predictions = pd.read_csv(SAFETY_PREDICTIONS_PATH)
    paths = [
        figure_one(core),
        figure_two(cohort),
        figure_three(loho),
        figure_four(safety),
        figure_five(safety_predictions),
        figure_six(core),
    ]
    manifest = {
        "source_artifacts": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in (
                CORE_PATH,
                SAFETY_PATH,
                LOHO_PATH,
                SAFETY_PREDICTIONS_PATH,
                COHORT_PATH,
            )
        },
        "figures": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
