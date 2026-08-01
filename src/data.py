"""Data contracts and pure cohort-building utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

SOURCE_COLUMNS = [
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
    "num",
]

PREDICTOR_COLUMNS = SOURCE_COLUMNS[:-1]
FORBIDDEN_PREDICTORS = {"site", "patient_id", "num", "target"}
EXPECTED_SITES = ("Cleveland", "Hungary", "Switzerland", "VA Long Beach")


@dataclass(frozen=True)
class CenterSource:
    site: str
    filename: str


CENTER_SOURCES = (
    CenterSource("Cleveland", "processed.cleveland.data"),
    CenterSource("Hungary", "processed.hungarian.data"),
    CenterSource("Switzerland", "processed.switzerland.data"),
    CenterSource("VA Long Beach", "processed.va.data"),
)


def read_center(path: Path, site: str) -> pd.DataFrame:
    """Read one processed UCI center without fitting any transformation."""
    frame = pd.read_csv(
        path,
        header=None,
        names=SOURCE_COLUMNS,
        na_values=["?"],
        keep_default_na=True,
    )
    for column in SOURCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame.insert(0, "site", site)
    site_key = site.lower().replace(" ", "_")
    frame.insert(
        0,
        "patient_id",
        [f"{site_key}:{index:04d}" for index in range(1, len(frame) + 1)],
    )
    if frame["num"].isna().any():
        raise ValueError(f"{site}: source outcome num contains missing values")
    if not frame["num"].isin([0, 1, 2, 3, 4]).all():
        invalid = sorted(frame.loc[~frame["num"].isin([0, 1, 2, 3, 4]), "num"].unique())
        raise ValueError(f"{site}: invalid num values: {invalid}")
    frame["target"] = (frame["num"] > 0).astype("int8")
    return frame


def build_cohort(raw_dir: Path) -> pd.DataFrame:
    """Combine the four centers while preserving missing values and provenance."""
    frames = [
        read_center(raw_dir / source.filename, source.site)
        for source in CENTER_SOURCES
    ]
    cohort = pd.concat(frames, ignore_index=True, verify_integrity=True)
    expected_columns = ["patient_id", "site", *SOURCE_COLUMNS, "target"]
    if cohort.columns.tolist() != expected_columns:
        raise AssertionError("standardized schema drifted")
    if cohort["patient_id"].duplicated().any():
        raise ValueError("patient_id is not globally unique")
    return cohort


def assert_prediction_features(columns: Iterable[str]) -> list[str]:
    """Reject leakage-prone columns at the model boundary."""
    selected = list(columns)
    forbidden = sorted(FORBIDDEN_PREDICTORS.intersection(selected))
    if forbidden:
        raise ValueError(f"forbidden prediction columns: {forbidden}")
    unknown = sorted(set(selected).difference(PREDICTOR_COLUMNS))
    if unknown:
        raise ValueError(f"unknown prediction columns: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate prediction columns")
    return selected
