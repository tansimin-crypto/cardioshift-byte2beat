from __future__ import annotations

import json
import unittest
from pathlib import Path


class FigureContractTests(unittest.TestCase):
    def test_manifest_lists_six_nonempty_figures_when_generated(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "outputs" / "figures" / "figure_manifest.json"
        if not manifest_path.exists():
            self.skipTest("figures have not been generated")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["figures"]), 6)
        for filename, record in manifest["figures"].items():
            path = manifest_path.parent / filename
            self.assertTrue(path.exists(), msg=filename)
            self.assertGreater(record["bytes"], 10_000, msg=filename)


if __name__ == "__main__":
    unittest.main()
