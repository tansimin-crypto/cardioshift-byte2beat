from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class CIContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = (
            Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
        )
        self.source = self.path.read_text(encoding="utf-8")
        self.workflow = yaml.safe_load(self.source)

    def test_python_matrix_and_separate_demo_job(self) -> None:
        self.assertIn('"3.11"', self.source)
        self.assertIn('"3.12"', self.source)
        self.assertIn("core", self.workflow["jobs"])
        self.assertIn("demo", self.workflow["jobs"])

    def test_required_checks_are_present(self) -> None:
        for token in (
            "requirements.lock",
            "build_cohort.py",
            "verify_gate_g1.py",
            "verify_gate_g2.py",
            "verify_shift_safety.py",
            "verify_robustness.py",
            "compileall",
            "execute_notebook.py",
            "verify_gate_g3.py",
            "test_notebook_contract.py",
            "test_figure_contract.py",
            "test_canonical_results.py",
            "import app",
        ):
            self.assertIn(token, self.source)

    def test_coder_preflight_is_validation_only(self) -> None:
        self.assertIn("coder-preflight", self.workflow["jobs"])
        for token in (
            "hashicorp/setup-terraform@v3",
            "terraform -chdir=coder fmt -check",
            "terraform -chdir=coder init -backend=false -input=false",
            "terraform -chdir=coder validate -no-color",
            "docker build -f coder/Dockerfile",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("terraform -chdir=coder apply", self.source)

    def test_ci_does_not_publish_or_submit(self) -> None:
        lowered = self.source.lower()
        for forbidden in (
            "kaggle kernels push",
            "kaggle competitions submit",
            "coder templates push",
            "terraform -chdir=coder apply",
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
