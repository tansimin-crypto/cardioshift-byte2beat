from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_public_release import (
    REQUIRED_FILES,
    non_ssh_email_addresses,
    scan_release,
    verify_manifest,
)


class PublicReleaseContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.release = (
            self.root
            if (self.root / "MANIFEST.json").is_file()
            else self.root / "dist" / "public-release"
        )

    def test_release_exists_with_required_files(self) -> None:
        self.assertTrue(self.release.is_dir())
        for relative in REQUIRED_FILES:
            with self.subTest(relative=relative):
                self.assertTrue((self.release / relative).is_file())

    def test_manifest_is_exact_and_hashes_match(self) -> None:
        failures, count = verify_manifest(self.release)
        self.assertEqual(failures, [])
        self.assertGreater(count, 40)

    def test_sanitization_scan_passes(self) -> None:
        self.assertEqual(scan_release(self.release), [])
    def test_committed_verification_status_is_internally_consistent(self) -> None:
        path = (
            self.root / "evidence" / "public_release" / "verification.json"
        )
        if not path.is_file():
            self.skipTest("verification evidence is outside standalone package")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["status"] == "pass":
            self.assertEqual(report["sanitization_scan"], "pass")
            self.assertEqual(report["failures"], [])



    def test_urls_record_published_release(self) -> None:
        urls = json.loads(
            (self.release / "RELEASE_URLS.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIs(urls["published"], True)
        for key in (
            "public_repository",
            "public_kaggle_notebook",
            "public_demo",
            "public_manuscript",
            "audited_release_sha",
        ):
            self.assertTrue(urls[key], msg=key)

    def test_git_ssh_url_is_not_misclassified_as_email(self) -> None:
        text = 'repo_url="${VALUE:-git@github.com:owner/repository.git}"'
        self.assertEqual(non_ssh_email_addresses(text), set())
        self.assertEqual(
            non_ssh_email_addresses("contact=person" + "@" + "example.com"),
            {"person" + "@" + "example.com"},
        )

    def test_executed_notebook_is_not_public(self) -> None:
        self.assertFalse(
            (
                self.release
                / "notebooks"
                / "CardioShift_Research_Report.executed.ipynb"
            ).exists()
        )
    def test_negative_absolute_path_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "leak.txt").write_text(
                "local=" + "C:" + "\\Users\\Example\\secret.txt\n",
                encoding="utf-8",
            )
            failures = scan_release(root)
            self.assertTrue(
                any("absolute Windows path" in item for item in failures)
            )


if __name__ == "__main__":
    unittest.main()
