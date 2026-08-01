from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_kaggle_rules_evidence import rules_evidence_verified


class KaggleRulesEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        evidence_dir = self.root / "evidence" / "kaggle"
        self.summary_path = evidence_dir / "current_rules.json"
        self.raw_path = evidence_dir / "raw_pages.json"
        self.summary = json.loads(
            self.summary_path.read_text(encoding="utf-8")
        )
        self.raw = json.loads(self.raw_path.read_text(encoding="utf-8"))

    def _verify(self, summary: dict, raw: dict) -> bool:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "current_rules.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            (directory / "raw_pages.json").write_text(
                json.dumps(raw, indent=2) + "\n",
                encoding="utf-8",
            )
            return rules_evidence_verified(
                directory / "current_rules.json"
            )

    def test_current_capture_passes(self) -> None:
        self.assertTrue(rules_evidence_verified(self.summary_path))

    def test_summary_cannot_forge_hash_and_length_without_raw_content(self) -> None:
        summary = json.loads(json.dumps(self.summary))
        record = summary["captured_pages"]["rules"]
        record["characters"] = 1
        record["sha256_utf8"] = hashlib.sha256(b"x").hexdigest()
        self.assertFalse(self._verify(summary, self.raw))

    def test_raw_content_change_fails_against_unchanged_summary(self) -> None:
        raw = json.loads(json.dumps(self.raw))
        raw["pages"]["rules"] += "\nchanged"
        self.assertFalse(self._verify(self.summary, raw))

    def test_missing_raw_page_fails(self) -> None:
        raw = json.loads(json.dumps(self.raw))
        del raw["pages"]["foundational-rules"]
        self.assertFalse(self._verify(self.summary, raw))

    def test_credential_like_key_fails_closed(self) -> None:
        raw = json.loads(json.dumps(self.raw))
        raw["access_token"] = "not-a-real-token"
        self.assertFalse(self._verify(self.summary, raw))

    def test_mismatched_provenance_fails_closed(self) -> None:
        summary = json.loads(json.dumps(self.summary))
        summary["source"]["kaggle_cli_version"] = "Kaggle CLI 0.0.0"
        self.assertFalse(self._verify(summary, self.raw))


if __name__ == "__main__":
    unittest.main()
