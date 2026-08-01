from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.verify_coder import (
    run_terraform_checks,
    validate_coder_sources,
    validated_runtime_evidence,
)


class CoderContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.coder = self.root / "coder"
        self.main_tf = (self.coder / "main.tf").read_text(encoding="utf-8")
        self.startup = (self.coder / "startup.sh").read_text(encoding="utf-8")
        self.dockerfile = (self.coder / "Dockerfile").read_text(encoding="utf-8")

    def issues(
        self,
        *,
        main_tf: str | None = None,
        startup: str | None = None,
        dockerfile: str | None = None,
    ) -> list[str]:
        return validate_coder_sources(
            self.main_tf if main_tf is None else main_tf,
            self.startup if startup is None else startup,
            self.dockerfile if dockerfile is None else dockerfile,
        )

    def test_required_files_exist(self) -> None:
        for name in (
            "main.tf",
            ".terraform.lock.hcl",
            "Dockerfile",
            "startup.sh",
            "README_CODER.md",
            "known_hosts",
        ):
            path = self.coder / name
            self.assertTrue(path.exists(), msg=name)
            self.assertGreater(path.stat().st_size, 0, msg=name)

    def test_jupyterlab_is_locked_for_runtime(self) -> None:
        lock = (self.root / "requirements-notebook.lock").read_text(encoding="utf-8")
        self.assertRegex(lock, r"(?m)^jupyterlab==[0-9]+\.[0-9]+\.[0-9]+$")

    def test_submission_bundle_includes_known_hosts_when_built(self) -> None:
        bundle = self.root / "dist" / "submission"
        if not bundle.exists():
            self.skipTest("submission bundle has not been built")
        self.assertTrue((bundle / "coder" / "known_hosts").is_file())
    def test_expected_sha_default_is_fail_closed_placeholder(self) -> None:
        self.assertIn(
            'default      = "0000000000000000000000000000000000000000"',
            self.main_tf,
        )

    def test_current_static_contract_passes(self) -> None:
        self.assertEqual(self.issues(), [])

    def test_private_repo_without_external_auth_fails(self) -> None:
        mutated = self.main_tf.replace(
            'data "coder_external_auth" "github"',
            'data "coder_workspace" "removed_external_auth"',
        )
        self.assertIn("missing_github_external_auth", self.issues(main_tf=mutated))

    def test_read_only_ssh_clone_contract_is_required(self) -> None:
        mutated = self.dockerfile.replace("openssh-client", "disabled")
        self.assertIn(
            "missing_read_only_ssh_clone",
            self.issues(dockerfile=mutated),
        )

    def test_runtime_verifiers_are_read_only(self) -> None:
        mutated = self.startup.replace(
            "scripts/verify_shift_safety.py --no-write",
            "scripts/verify_shift_safety.py",
        )
        self.assertIn(
            "runtime_verifiers_may_write_tracked_evidence",
            self.issues(startup=mutated),
        )

    def test_runtime_status_is_outside_git_worktree(self) -> None:
        mutated = self.startup.replace(
            'status_dir="${workspace_root}/.coder-status"',
            'status_dir="${repo_dir}/.coder-status"',
        )
        self.assertIn(
            "status_markers_inside_git_worktree",
            self.issues(startup=mutated),
        )

    def test_no_checkout_clone_is_not_treated_as_dirty(self) -> None:
        mutated = self.startup.replace("[[ -f .git/index ]]", "true")
        self.assertIn("missing_dirty_tree_guard", self.issues(startup=mutated))

    def test_expected_sha_mismatch_guard_is_required(self) -> None:
        mutated = self.startup.replace(
            '"${fetched_sha}" != "${expected_release_sha}"',
            '"${fetched_sha}" != "disabled"',
        )
        self.assertIn("missing_ref_sha_match", self.issues(startup=mutated))

    def test_stale_persistent_clone_must_fetch(self) -> None:
        mutated = self.startup.replace("git fetch --prune --no-tags", "git status")
        self.assertIn("missing_persistent_fetch", self.issues(startup=mutated))

    def test_app_absence_must_fail_before_services(self) -> None:
        mutated = self.startup.replace("  app.py\n", "")
        self.assertIn("missing_required_app_guard", self.issues(startup=mutated))

    def test_jupyter_health_unavailable_must_fail(self) -> None:
        mutated = self.startup.replace("http://127.0.0.1:8888/api", "disabled")
        self.assertIn(
            "missing_jupyter_health_failure",
            self.issues(startup=mutated),
        )

    def test_streamlit_health_unavailable_must_fail(self) -> None:
        mutated = self.startup.replace(
            "http://127.0.0.1:8501/_stcore/health",
            "disabled",
        )
        self.assertIn(
            "missing_streamlit_health_failure",
            self.issues(startup=mutated),
        )

    def test_terraform_unavailable_cannot_pass_preflight(self) -> None:
        result = run_terraform_checks(None, root=self.root)
        self.assertFalse(result["available"])
        self.assertFalse(result["all_passed"])
        self.assertTrue(
            all(
                record["status"] == "not_run_terraform_unavailable"
                for record in result["commands"]
            )
        )

    def test_no_credentials_or_fixed_coder_url(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in self.coder.iterdir()
            if path.is_file()
        )
        self.assertNotRegex(
            source,
            r"(?i)(access_token|password|private_key)\s*=\s*[\"'][^\"']+[\"']",
        )
        self.assertNotIn("coder.example.com", source)

    def test_real_runtime_evidence_remains_accepted(self) -> None:
        runtime = validated_runtime_evidence()
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime["status"], "pass")
        self.assertEqual(len(runtime["tested_release_sha"]), 40)

    def test_evidence_does_not_claim_runtime_pass(self) -> None:
        path = self.root / "evidence" / "g5" / "verification.json"
        if not path.exists():
            self.skipTest("Coder verifier has not run")
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence["runtime_verified"]:
            self.assertEqual(evidence["status"], "pass")
            self.assertFalse(evidence["unresolved_blockers"])
        else:
            self.assertEqual(
                evidence["status"],
                "implementation_complete_runtime_verification_pending",
            )
            self.assertTrue(evidence["unresolved_blockers"])


if __name__ == "__main__":
    unittest.main()
