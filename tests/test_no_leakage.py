from __future__ import annotations

import unittest

import pandas as pd
import yaml

from src.data import EXPECTED_SITES, PREDICTOR_COLUMNS
from src.modeling import candidates_from_config, fit_predict_loho_fold


class LeakageTests(unittest.TestCase):
    def synthetic_cohort(self) -> pd.DataFrame:
        rows = []
        for site_index, site in enumerate(EXPECTED_SITES):
            for row_index in range(24):
                target = row_index % 2
                rows.append(
                    {
                        "patient_id": f"{site_index}:{row_index}",
                        "site": site,
                        "age": 40 + row_index,
                        "sex": row_index % 2,
                        "cp": 1 + row_index % 4,
                        "trestbps": 110 + row_index,
                        "chol": 180 + row_index,
                        "fbs": row_index % 2,
                        "restecg": row_index % 3,
                        "thalach": 170 - row_index,
                        "exang": row_index % 2,
                        "oldpeak": row_index / 10,
                        "slope": 1 + row_index % 3,
                        "ca": row_index % 4,
                        "thal": (3, 6, 7)[row_index % 3],
                        "num": target,
                        "target": target,
                    }
                )
        return pd.DataFrame(rows)

    def test_outer_site_never_enters_training_stages(self) -> None:
        config = {
            "models": {
                "logistic_regression": {
                    "C": [1.0],
                    "class_weight": [None],
                    "max_iter": 500,
                },
                "random_forest": {
                    "n_estimators": [10],
                    "max_depth": [3],
                    "min_samples_leaf": [2],
                    "class_weight": ["balanced"],
                },
                "hist_gradient_boosting": {
                    "learning_rate": [0.08],
                    "max_leaf_nodes": [7],
                    "l2_regularization": [1.0],
                },
            }
        }
        cohort = self.synthetic_cohort()
        result = fit_predict_loho_fold(
            cohort,
            "Switzerland",
            candidates_from_config(config),
            seed=17,
        )
        ledger = result["ledger"]
        test_ids = set(ledger["outer_test_patient_ids"])
        self.assertEqual(ledger["outer_test_sites"], ["Switzerland"])
        self.assertNotIn("Switzerland", ledger["outer_train_sites"])
        for stage in ("tuning", "calibration", "final_fit"):
            self.assertTrue(
                test_ids.isdisjoint(ledger[f"{stage}_patient_ids"]),
                msg=stage,
            )
        self.assertEqual(
            set(result["predictions"]["patient_id"]),
            test_ids,
        )

    def test_model_boundary_is_allowlist_only(self) -> None:
        self.assertNotIn("site", PREDICTOR_COLUMNS)
        self.assertNotIn("patient_id", PREDICTOR_COLUMNS)
        self.assertNotIn("num", PREDICTOR_COLUMNS)
        self.assertNotIn("target", PREDICTOR_COLUMNS)


if __name__ == "__main__":
    unittest.main()
