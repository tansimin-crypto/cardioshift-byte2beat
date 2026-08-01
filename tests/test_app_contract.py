from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path

from src.demo_cases import select_demo_cases
from src.results_access import ResultsAccessor


class AppContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.source = (self.root / "app.py").read_text(encoding="utf-8")

    def test_app_import_smoke(self) -> None:
        module = importlib.import_module("app")
        self.assertEqual(len(module.PAGES), 5)

    def test_no_real_patient_input_form(self) -> None:
        forbidden = (
            "text_input(",
            "number_input(",
            "date_input(",
            "file_uploader(",
            "camera_input(",
        )
        for token in forbidden:
            self.assertNotIn(token, self.source)

    def test_headline_metrics_are_not_hardcoded(self) -> None:
        tree = ast.parse(self.source)
        numeric = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        }
        for value in (0.892, 0.789, 0.102, 0.405, 0.102):
            self.assertNotIn(value, numeric)
        results = ResultsAccessor(self.root)
        self.assertIn("random_split_mean_auroc", results.findings)

    def test_required_failure_cases_are_present(self) -> None:
        cases = select_demo_cases(self.root)
        self.assertEqual(
            set(cases),
            {
                "accepted_confident_correct",
                "deferred_confident_error",
                "confident_error_not_caught_by_gate",
                "heavy_missingness_case",
            },
        )
        self.assertFalse(cases["confident_error_not_caught_by_gate"]["correct"])
        self.assertEqual(
            cases["confident_error_not_caught_by_gate"]["decision"],
            "ACCEPT",
        )

    def test_medical_nonuse_language_is_explicit(self) -> None:
        lowered = self.source.lower()
        for phrase in ("not for diagnosis", "treatment", "real-patient"):
            self.assertIn(phrase, lowered)


if __name__ == "__main__":
    unittest.main()
