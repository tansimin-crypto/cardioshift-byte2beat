from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


class CoreArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.results_path = self.root / "outputs" / "metrics" / "results.json"
        if not self.results_path.exists():
            self.skipTest("core experiments have not been run")

    def test_loho_prediction_universe_is_exact(self) -> None:
        cohort = pd.read_csv(
            self.root / "data" / "processed" / "cardioshift_cohort.csv"
        )
        loho = pd.read_csv(
            self.root / "outputs" / "predictions" / "loho_predictions.csv"
        )
        self.assertTrue(loho["patient_id"].is_unique)
        self.assertEqual(set(loho["patient_id"]), set(cohort["patient_id"]))

    def test_training_ledgers_exclude_held_out_patients(self) -> None:
        ledgers = json.loads(
            (
                self.root / "outputs" / "audit" / "loho_training_ledgers.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(ledgers), 4)
        for ledger in ledgers:
            test_ids = set(ledger["outer_test_patient_ids"])
            self.assertEqual(ledger["outer_test_sites"], [ledger["held_out_site"]])
            self.assertNotIn(ledger["held_out_site"], ledger["outer_train_sites"])
            for stage in ("tuning", "calibration", "final_fit"):
                self.assertTrue(
                    test_ids.isdisjoint(ledger[f"{stage}_patient_ids"]),
                    msg=f"{ledger['held_out_site']}:{stage}",
                )

    def test_results_have_sample_sizes_and_bootstrap(self) -> None:
        results = json.loads(self.results_path.read_text(encoding="utf-8"))
        pooled = results["E2_leave_one_hospital_out"]["pooled"]
        self.assertEqual(pooled["n"], 920)
        self.assertEqual(pooled["bootstrap"]["requested_replicates"], 2000)
        for site in results["E2_leave_one_hospital_out"]["by_site"]:
            self.assertGreater(site["metrics"]["n"], 0)
            self.assertEqual(
                site["metrics"]["bootstrap"]["requested_replicates"],
                2000,
            )


if __name__ == "__main__":
    unittest.main()
