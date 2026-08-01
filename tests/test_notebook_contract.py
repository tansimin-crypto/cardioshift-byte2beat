from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path


class NotebookContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.path = self.root / "notebooks" / "CardioShift_Research_Report.ipynb"
        if not self.path.exists():
            self.skipTest("Notebook has not been built")
        self.notebook = json.loads(self.path.read_text(encoding="utf-8"))
        self.source = "\n".join(
            "".join(cell.get("source", [])) for cell in self.notebook["cells"]
        )

    def test_notebook_has_all_required_sections(self) -> None:
        for section in (
            "Executive Summary",
            "Intended Use and Non-use",
            "Dataset Provenance and License",
            "Four-hospital cohort",
            "Validation design",
            "Random Split vs LOHO",
            "Per-hospital calibration",
            "Dataset-shift diagnostics",
            "Selective prediction",
            "E6 robustness",
            "Subgroups",
            "Runtime profile",
            "Failure case",
            "Limitations",
            "Reproducibility manifest",
        ):
            self.assertIn(section, self.source)

    def test_notebook_is_offline_and_portable(self) -> None:
        self.assertFalse(self.notebook["metadata"]["cardioshift"]["internet"])
        self.assertNotRegex(self.source, r"[A-Za-z]:\\\\")
        self.assertNotIn("requests.get", self.source)
        self.assertNotIn("urlopen", self.source)
        self.assertIn("/kaggle/input", self.source)
        self.assertIn("CARDIOSHIFT_DATA_DIR", self.source)

    def test_headline_numbers_come_from_accessor(self) -> None:
        self.assertIn("ResultsAccessor", self.source)
        self.assertIn("F = R.findings", self.source)
        markdown_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for forbidden in ("0.892", "0.789", "0.102", "40.5%", "10.2%"):
            self.assertNotIn(forbidden, markdown_source)

    def test_notebook_is_generated_and_reasonable_size(self) -> None:
        self.assertEqual(
            self.notebook["metadata"]["cardioshift"]["generated_by"],
            "scripts/build_kaggle_notebook.py",
        )
        self.assertLess(self.path.stat().st_size, 500_000)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_every_code_cell_compiles(self) -> None:
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell.get("source", []))
                with self.subTest(cell=index):
                    ast.parse(source, filename=f"notebook-cell-{index}")

    def test_dataset_manifest_has_required_payload(self) -> None:
        manifest_path = (
            self.root / "dist" / "kaggle" / "cardioshift-data" / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["internet_required"])
        files = set(manifest["files"])
        for name in (
            "raw/processed.cleveland.data",
            "raw/processed.hungarian.data",
            "raw/processed.switzerland.data",
            "raw/processed.va.data",
            "UCI_LICENSE.md",
            "DOI.txt",
            "checksums.json",
            "cardioshift_source.zip",
            "outputs/results.json",
        ):
            self.assertIn(name, files)


if __name__ == "__main__":
    unittest.main()
