"""Select fixed, deidentified research cases from accepted safety predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _confidence(probability: pd.Series) -> pd.Series:
    return (probability - 0.5).abs()


def select_demo_cases(base_dir: Path) -> dict[str, dict[str, object]]:
    frame = pd.read_csv(
        Path(base_dir)
        / "outputs"
        / "predictions"
        / "safety_loho_predictions.csv"
    )
    frame["prediction"] = (frame["calibrated_probability"] >= 0.5).astype(int)
    frame["correct"] = frame["prediction"] == frame["target"]
    frame["confidence"] = _confidence(frame["calibrated_probability"])

    selectors = {
        "accepted_confident_correct": frame["accepted"] & frame["correct"],
        "deferred_confident_error": (~frame["accepted"]) & (~frame["correct"]),
        "confident_error_not_caught_by_gate": frame["accepted"] & (~frame["correct"]),
        "heavy_missingness_case": frame["missing_fraction"]
        == frame["missing_fraction"].max(),
    }
    selected: dict[str, dict[str, object]] = {}
    for label, selector in selectors.items():
        candidates = frame.loc[selector].copy()
        if candidates.empty:
            raise RuntimeError(f"required research case is unavailable: {label}")
        sort_column = (
            "missing_fraction" if label == "heavy_missingness_case" else "confidence"
        )
        row = candidates.sort_values(sort_column, ascending=False).iloc[0]
        selected[label] = {
            "case_id": str(row["patient_id"]),
            "hospital": str(row["site"]),
            "model_probability": float(row["calibrated_probability"]),
            "prediction_set": str(row["conformal_set"]),
            "ood_flag": bool(row["is_ood"]),
            "ambiguity_flag": bool(row["is_ambiguous"]),
            "missingness_flag": bool(
                row["missing_fraction"] > row["missing_threshold"]
            ),
            "missing_fraction": float(row["missing_fraction"]),
            "decision": "ACCEPT" if bool(row["accepted"]) else "DEFER",
            "research_label": int(row["target"]),
            "prediction": int(row["prediction"]),
            "correct": bool(row["correct"]),
        }
    return selected
