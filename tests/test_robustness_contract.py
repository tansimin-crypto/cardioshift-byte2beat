from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd
import yaml


class RobustnessContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.result_path = (
            self.root / "outputs" / "metrics" / "robustness_results.json"
        )
        self.prediction_path = (
            self.root / "outputs" / "predictions" / "robustness_predictions.csv"
        )
        if not self.result_path.exists() or not self.prediction_path.exists():
            self.skipTest("E6/E7 artifacts have not been generated")
        self.result = json.loads(self.result_path.read_text(encoding="utf-8"))
        self.predictions = pd.read_csv(self.prediction_path)
        self.config = yaml.safe_load(
            (self.root / "configs" / "experiment.yaml").read_text(encoding="utf-8")
        )

    def test_protocol_forbids_model_reselection(self) -> None:
        self.assertEqual(
            self.result["protocol"]["model_selection"],
            "none; accepted E2 candidate reused per LOHO fold",
        )
        self.assertEqual(len(self.result["fold_metadata"]), 4)

    def test_all_prespecified_scenarios_exist(self) -> None:
        scenarios = set(self.predictions["scenario"])
        self.assertTrue({"mcar_10", "mcar_30", "mcar_50"}.issubset(scenarios))
        self.assertTrue(
            set(self.config["stress_tests"]["grouped"]).issubset(scenarios)
        )
        for scenario in ("mcar_10", "mcar_30", "mcar_50"):
            repeats = self.predictions.loc[
                self.predictions["scenario"] == scenario, "repeat"
            ].nunique()
            self.assertEqual(
                repeats,
                self.config["stress_tests"]["mcar_repeats"],
            )

    def test_each_scenario_repeat_has_same_patient_universe(self) -> None:
        baseline = set(
            self.predictions.loc[
                self.predictions["scenario_type"] == "baseline", "patient_id"
            ]
        )
        for _, frame in self.predictions.groupby(
            ["scenario_type", "scenario", "repeat"]
        ):
            self.assertEqual(set(frame["patient_id"]), baseline)

    def test_evidence_hashes_current_outputs(self) -> None:
        evidence_path = self.root / "evidence" / "e6_e7" / "verification.json"
        if not evidence_path.exists():
            self.skipTest("E6/E7 verifier has not run")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        for key, path in (
            ("robustness_results", self.result_path),
            ("robustness_predictions", self.prediction_path),
        ):
            content = path.read_bytes().replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                evidence["artifact_hashes"][key],
            )

    def test_runtime_profile_is_research_only(self) -> None:
        runtime = self.result["E7_runtime"]
        self.assertIn("not medical-device", runtime["claim_scope"])
        self.assertEqual(set(runtime["by_fold"]), set(self.result["fold_metadata"]))
        for record in runtime["by_fold"].values():
            self.assertGreater(record["serialized_model_bytes"], 0)
            self.assertGreater(record["batch_1_ms"]["median"], 0)
            self.assertGreater(record["batch_100_ms"]["median"], 0)


if __name__ == "__main__":
    unittest.main()
