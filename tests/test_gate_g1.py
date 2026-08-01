from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_cohort import audit_cohort
from src.data import (
    EXPECTED_SITES,
    PREDICTOR_COLUMNS,
    SOURCE_COLUMNS,
    assert_prediction_features,
    build_cohort,
    read_center,
)


class GateG1Tests(unittest.TestCase):
    @staticmethod
    def source_row() -> list[object]:
        return [63, 1, 4, 145, 233, 1, 0, 150, 0, 2.3, 3, 0, 6, 1]

    def test_parser_preserves_missing_and_derives_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "center.data"
            row = self.source_row()
            row[4] = "?"
            path.write_text(",".join(map(str, row)) + "\n", encoding="utf-8")
            frame = read_center(path, "Cleveland")
        self.assertTrue(pd.isna(frame.loc[0, "chol"]))
        self.assertEqual(int(frame.loc[0, "target"]), 1)

    def test_all_sites_are_mapped_and_ids_are_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            filenames = (
                "processed.cleveland.data",
                "processed.hungarian.data",
                "processed.switzerland.data",
                "processed.va.data",
            )
            for offset, filename in enumerate(filenames):
                row = self.source_row()
                row[0] = 50 + offset
                (raw_dir / filename).write_text(
                    ",".join(map(str, row)) + "\n",
                    encoding="utf-8",
                )
            cohort = build_cohort(raw_dir)
        self.assertEqual(tuple(cohort["site"]), EXPECTED_SITES)
        self.assertTrue(cohort["patient_id"].is_unique)

    def test_invalid_label_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "center.data"
            row = self.source_row()
            row[-1] = 7
            path.write_text(",".join(map(str, row)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid num"):
                read_center(path, "Hungary")

    def test_site_and_outcome_are_rejected_as_predictors(self) -> None:
        for forbidden in ("site", "patient_id", "num", "target"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    assert_prediction_features([*PREDICTOR_COLUMNS, forbidden])

    def test_duplicate_rows_are_reported_not_dropped(self) -> None:
        rows = []
        for index, site in enumerate(EXPECTED_SITES, start=1):
            row = {
                "patient_id": f"id:{index}",
                "site": site,
                **dict(zip(SOURCE_COLUMNS, self.source_row())),
                "target": 1,
            }
            rows.append(row)
        rows.append({**rows[0], "patient_id": "id:duplicate"})
        cohort = pd.DataFrame(rows)
        audit = audit_cohort(cohort)
        self.assertEqual(len(cohort), 5)
        self.assertEqual(audit["exact_duplicate_rows_ignoring_patient_id"], 2)

    def test_canonical_artifacts_agree_when_present(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cohort_path = root / "data" / "processed" / "cardioshift_cohort.csv"
        audit_path = root / "data" / "audit.json"
        if not cohort_path.exists():
            self.skipTest("canonical data has not been built")
        cohort = pd.read_csv(cohort_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(len(cohort), audit["rows"])
        self.assertEqual(
            cohort[PREDICTOR_COLUMNS].isna().sum().to_dict(),
            audit["missing_by_feature"],
        )


if __name__ == "__main__":
    unittest.main()
