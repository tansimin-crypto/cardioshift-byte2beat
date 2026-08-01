from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import (
    PREDICTOR_COLUMNS,
    SOURCE_COLUMNS,
    assert_prediction_features,
    read_center,
)


class DataContractTests(unittest.TestCase):
    def _source_row(self) -> list[object]:
        return [63, 1, 4, 145, 233, 1, 0, 150, 0, 2.3, 3, 0, 6, 1]

    def test_read_center_preserves_missing_and_derives_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "center.data"
            row = self._source_row()
            row[4] = "?"
            path.write_text(",".join(map(str, row)) + "\n", encoding="utf-8")
            frame = read_center(path, "Cleveland")

        self.assertEqual(frame.loc[0, "patient_id"], "cleveland:0001")
        self.assertTrue(pd.isna(frame.loc[0, "chol"]))
        self.assertEqual(int(frame.loc[0, "target"]), 1)
        self.assertEqual(frame.columns.tolist(), ["patient_id", "site", *SOURCE_COLUMNS, "target"])

    def test_invalid_source_outcome_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "center.data"
            row = self._source_row()
            row[-1] = 9
            path.write_text(",".join(map(str, row)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid num"):
                read_center(path, "Hungary")

    def test_prediction_boundary_rejects_site_and_outcome(self) -> None:
        for forbidden in ("site", "patient_id", "num", "target"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    assert_prediction_features([*PREDICTOR_COLUMNS, forbidden])

    def test_prespecified_predictors_pass(self) -> None:
        self.assertEqual(assert_prediction_features(PREDICTOR_COLUMNS), PREDICTOR_COLUMNS)


if __name__ == "__main__":
    unittest.main()
