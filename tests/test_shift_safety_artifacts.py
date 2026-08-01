from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.safety import selective_metrics


class ShiftSafetyArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.result_path = (
            self.root / "outputs" / "metrics" / "shift_safety_results.json"
        )
        if not self.result_path.exists():
            self.skipTest("shift/safety experiments have not been run")

    def test_accepted_flag_is_outcome_independent_formula(self) -> None:
        frame = pd.read_csv(
            self.root
            / "outputs"
            / "predictions"
            / "safety_loho_predictions.csv"
        )
        expected = ~(
            frame["is_ood"].astype(bool)
            | frame["is_ambiguous"].astype(bool)
            | (frame["conformal_set_size"] != 1)
            | (frame["missing_fraction"] > frame["missing_threshold"])
        )
        self.assertTrue(expected.equals(frame["accepted"].astype(bool)))

    def test_pooled_metrics_recompute(self) -> None:
        frame = pd.read_csv(
            self.root
            / "outputs"
            / "predictions"
            / "safety_loho_predictions.csv"
        )
        result = json.loads(self.result_path.read_text(encoding="utf-8"))
        expected = selective_metrics(
            frame["target"].to_numpy(),
            frame["calibrated_probability"].to_numpy(),
            frame["accepted"].to_numpy(),
            frame["conformal_contains_true"].to_numpy(),
            frame["conformal_set_size"].to_numpy(),
        )
        observed = result["E5_safety"]["pooled"]
        for name, value in expected.items():
            self.assertTrue(
                np.isclose(value, observed[name], equal_nan=True),
                msg=name,
            )

    def test_core_results_hash_uses_canonical_text_bytes(self) -> None:
        from scripts.run_shift_safety import canonical_sha256

        result = json.loads(self.result_path.read_text(encoding="utf-8"))
        core_path = self.root / "outputs" / "metrics" / "results.json"
        self.assertEqual(
            result["reproducibility"]["core_results_sha256"],
            canonical_sha256(core_path),
        )

    def test_claims_include_shift_limitations(self) -> None:
        result = json.loads(self.result_path.read_text(encoding="utf-8"))
        self.assertIn("Empirical coverage only", result["E5_safety"]["conformal_claim"])
        self.assertIn("in-sample", result["E5_safety"]["ood_limitation"])


if __name__ == "__main__":
    unittest.main()
