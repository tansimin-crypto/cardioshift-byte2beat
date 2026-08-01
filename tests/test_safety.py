from __future__ import annotations

import unittest

import numpy as np

from src.safety import (
    class_conditional_thresholds,
    conformal_quantile,
    prediction_sets,
    selective_metrics,
)


class SafetyTests(unittest.TestCase):
    def test_conformal_quantile_is_finite_and_observed(self) -> None:
        scores = np.asarray([0.1, 0.2, 0.3, 0.4])
        value = conformal_quantile(scores, alpha=0.2)
        self.assertIn(value, scores)

    def test_prediction_sets_follow_class_thresholds(self) -> None:
        labels, sizes = prediction_sets(
            np.asarray([0.05, 0.5, 0.95]),
            {0: 0.2, 1: 0.2},
        )
        self.assertEqual(labels, ["0", "", "1"])
        self.assertEqual(sizes.tolist(), [1, 0, 1])

    def test_class_conditional_thresholds_use_true_class_scores(self) -> None:
        y = np.asarray([0, 0, 0, 1, 1, 1])
        probability = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        thresholds = class_conditional_thresholds(y, probability, alpha=0.2)
        self.assertAlmostEqual(thresholds[0], 0.3)
        self.assertAlmostEqual(thresholds[1], 0.3)

    def test_selective_metrics_only_score_accepted_cases(self) -> None:
        metrics = selective_metrics(
            y_true=np.asarray([0, 1, 1, 0]),
            probability=np.asarray([0.1, 0.2, 0.9, 0.8]),
            accepted=np.asarray([True, False, True, False]),
            conformal_contains_true=np.asarray([True, True, True, False]),
            conformal_set_size=np.asarray([1, 2, 1, 0]),
        )
        self.assertEqual(metrics["accepted_n"], 2)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["selective_risk"], 0.0)
        self.assertEqual(metrics["accepted_case_fnr"], 0.0)
        self.assertEqual(metrics["empirical_conformal_coverage"], 0.75)


if __name__ == "__main__":
    unittest.main()
