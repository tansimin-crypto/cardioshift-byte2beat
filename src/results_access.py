"""Single read-only accessor shared by the Notebook and Streamlit demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class ResultsAccessor:
    """Read canonical metrics and validated patient-level artifacts."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.results_path = self.base_dir / "outputs" / "results.json"
        self._results = json.loads(self.results_path.read_text(encoding="utf-8"))

    @property
    def canonical(self) -> dict[str, Any]:
        return self._results

    @property
    def findings(self) -> dict[str, Any]:
        return self._results["key_findings"]

    @property
    def gates(self) -> dict[str, str]:
        return self._results["gate_status"]

    @property
    def limitations(self) -> list[str]:
        return list(self._results["limitations"])

    @property
    def robustness(self) -> dict[str, Any]:
        return self._results["experiments"]["E6_robustness"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self._results["experiments"]["E7_runtime"]

    def prediction_frame(self, name: str) -> pd.DataFrame:
        allowed = {
            "loho": "loho_predictions.csv",
            "random": "random_split_predictions.csv",
            "safety": "safety_loho_predictions.csv",
            "robustness": "robustness_predictions.csv",
        }
        if name not in allowed:
            raise KeyError(f"unknown prediction artifact: {name}")
        return pd.read_csv(
            self.base_dir / "outputs" / "predictions" / allowed[name]
        )

    def per_hospital(self) -> pd.DataFrame:
        rows = []
        for record in self._results["experiments"]["E1_E2"][
            "E2_leave_one_hospital_out"
        ]["by_site"]:
            rows.append(
                {
                    "hospital": record["held_out_site"],
                    **{
                        key: record["metrics"][key]
                        for key in ("n", "events", "auroc", "brier")
                    },
                }
            )
        return pd.DataFrame(rows)

    def robustness_summary(self) -> pd.DataFrame:
        rows = []
        for record in self.robustness["scenarios"].values():
            if record["repeat"] != 0:
                continue
            rows.append(
                {
                    "scenario": record["scenario"],
                    **{
                        key: record["pooled"][key]
                        for key in (
                            "auroc",
                            "brier",
                            "safety_coverage",
                            "selective_risk",
                        )
                    },
                }
            )
        return pd.DataFrame(rows)

    def subgroup_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "subgroup": name,
                    **record["metrics"],
                    "interpretation": record["small_or_single_class_reason"],
                }
                for name, record in self.robustness["subgroups"].items()
            ]
        )

    def sha256(self, relative: str) -> str:
        content = (self.base_dir / relative).read_bytes()
        content = content.replace(b"\r\n", b"\n")
        return hashlib.sha256(content).hexdigest()
