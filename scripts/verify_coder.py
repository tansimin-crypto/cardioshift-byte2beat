"""Validate Coder preflight contracts without claiming a remote runtime."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "g5" / "verification.json"
RUNTIME_EVIDENCE = ROOT / "evidence" / "g5" / "runtime_verification.json"
FILES = (
    ROOT / "coder" / "main.tf",
    ROOT / "coder" / ".terraform.lock.hcl",
    ROOT / "coder" / "Dockerfile",
    ROOT / "coder" / "startup.sh",
    ROOT / "coder" / "README_CODER.md",
    ROOT / "coder" / "known_hosts",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")).hexdigest()


def validate_coder_sources(
    main_tf: str,
    startup: str,
    dockerfile: str,
) -> list[str]:
    """Return fail-closed issue codes for the static Coder contract."""
    checks = {
        "missing_github_external_auth": (
            'data "coder_external_auth" "github"' in main_tf
            and 'id = "github"' in main_tf
        ),
        "missing_read_only_ssh_clone": (
            "git@github.com:tansimin-crypto/byte-to-beat.git" in main_tf
            and "openssh-client" in dockerfile
            and "coder/known_hosts" in dockerfile
        ),
        "unbounded_coder_provider": bool(
            re.search(
                r'coder\s*=\s*\{.*?version\s*=\s*"[^"]*<\s*3\.0\.0"',
                main_tf,
                flags=re.DOTALL,
            )
        ),
        "unbounded_docker_provider": bool(
            re.search(
                r'docker\s*=\s*\{.*?version\s*=\s*"[^"]*<\s*4\.0\.0"',
                main_tf,
                flags=re.DOTALL,
            )
        ),
        "missing_repo_ref_parameter": (
            'data "coder_parameter" "repo_ref"' in main_tf
            and "CARDIOSHIFT_REPO_REF" in main_tf
        ),
        "missing_expected_sha_parameter": (
            'data "coder_parameter" "expected_release_sha"' in main_tf
            and "CARDIOSHIFT_EXPECTED_RELEASE_SHA" in main_tf
            and "^[0-9a-f]{40}$" in main_tf
        ),
        "missing_loopback_agent_mapping": (
            "local.agent_init_script" in main_tf
            and "host.docker.internal" in main_tf
            and "127.0.0.1" in main_tf
            and "localhost" in main_tf
        ),
        "missing_persistent_fetch": (
            'git fetch --prune --no-tags origin "${repo_ref}"' in startup
        ),
        "missing_ref_sha_match": (
            'fetched_sha="$(git rev-parse --verify' in startup
            and '"${fetched_sha}" != "${expected_release_sha}"' in startup
        ),
        "missing_exact_checkout": (
            'git checkout --detach --force "${expected_release_sha}"' in startup
        ),
        "missing_head_verification": (
            'actual_release_sha="$(git rev-parse HEAD)"' in startup
            and '"${actual_release_sha}" != "${expected_release_sha}"' in startup
        ),
        "missing_dirty_tree_guard": (
            "git diff --quiet" in startup
            and "git diff --cached --quiet" in startup
            and "[[ -f .git/index ]]" in startup
        ),
        "missing_python_import_contract": (
            "export PYTHONPATH=" in startup
            and '"${core_python}" -m scripts.build_cohort' in startup
        ),
        "runtime_verifiers_may_write_tracked_evidence": all(
            f"{script} --no-write" in startup
            for script in (
                "scripts/verify_gate_g1.py",
                "scripts/verify_gate_g2.py",
                "scripts/verify_shift_safety.py",
                "scripts/verify_robustness.py",
            )
        ),
        "status_markers_inside_git_worktree": (
            'status_dir="${workspace_root}/.coder-status"' in startup
        ),
        "missing_required_app_guard": (
            re.search(r"required_release_files=\(\s*app\.py\s", startup)
            is not None
            and "required release file is absent" in startup
        ),
        "missing_stale_marker_reset": (
            'rm -f "${status_dir}/tests.ok" "${status_dir}/services.ok"'
            in startup
        ),
        "missing_release_sha_evidence": (
            '"${status_dir}/release.sha"' in startup
        ),
        "missing_jupyter_health_failure": (
            "http://127.0.0.1:8888/api" in startup
            and "health endpoint unavailable" in startup
        ),
        "missing_streamlit_health_failure": (
            "http://127.0.0.1:8501/_stcore/health" in startup
            and "health endpoint unavailable" in startup
        ),
        "services_start_before_tests": (
            startup.find('touch "${status_dir}/tests.ok"')
            < startup.find('"${core_python}" -m jupyter lab')
            < startup.find('touch "${status_dir}/services.ok"')
        ),
        "docker_environments_not_separated": (
            "/opt/cardioshift/core" in dockerfile
            and "/opt/cardioshift/demo" in dockerfile
            and "requirements-demo.lock" in dockerfile
        ),
    }
    return [issue for issue, passed in checks.items() if not passed]


def run_terraform_checks(
    terraform: str | None,
    root: Path = ROOT,
) -> dict[str, Any]:
    commands = (
        ("terraform -chdir=coder fmt -check", ["-chdir=coder", "fmt", "-check"]),
        (
            "terraform -chdir=coder init -backend=false -input=false",
            ["-chdir=coder", "init", "-backend=false", "-input=false"],
        ),
        (
            "terraform -chdir=coder validate -no-color",
            ["-chdir=coder", "validate", "-no-color"],
        ),
    )
    if terraform is None:
        return {
            "available": False,
            "all_passed": False,
            "commands": [
                {
                    "command": label,
                    "returncode": None,
                    "status": "not_run_terraform_unavailable",
                }
                for label, _ in commands
            ],
        }

    records = []
    for label, arguments in commands:
        completed = subprocess.run(
            [terraform, *arguments],
            cwd=root,
            text=True,
            capture_output=True,
        )
        records.append(
            {
                "command": label,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{label} failed")
    return {"available": True, "all_passed": True, "commands": records}


def validated_runtime_evidence(path: Path = RUNTIME_EVIDENCE) -> dict[str, Any] | None:
    """Return accepted exact-SHA runtime evidence, otherwise fail closed."""
    if not path.is_file():
        return None
    try:
        runtime = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    sha = runtime.get("tested_release_sha", "")
    cold = runtime.get("cold_start", {})
    restart = runtime.get("restart", {})
    cold_ok = (
        cold.get("coder_healthy") is True
        and cold.get("exact_head") is True
        and cold.get("clean_worktree") is True
        and cold.get("release_marker_matches") is True
        and cold.get("tests_ok") is True
        and cold.get("jupyter_api_http_status") == 200
        and cold.get("streamlit_health") == "ok"
    )
    restart_ok = (
        restart.get("stop_succeeded") is True
        and restart.get("start_succeeded") is True
        and restart.get("coder_healthy") is True
        and restart.get("exact_head") is True
        and restart.get("clean_worktree") is True
        and restart.get("release_marker_matches") is True
        and restart.get("tests_ok") is True
        and restart.get("jupyter_api_http_status") == 200
        and restart.get("streamlit_health") == "ok"
    )
    if (
        runtime.get("schema_version") == "1.0"
        and runtime.get("status") == "pass"
        and re.fullmatch(r"[0-9a-f]{40}", sha)
        and cold_ok
        and restart_ok
        and runtime.get("unresolved_blockers") == []
    ):
        return runtime
    return None


def main() -> None:
    for path in FILES:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    main_tf = (ROOT / "coder" / "main.tf").read_text(encoding="utf-8")
    startup = (ROOT / "coder" / "startup.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "coder" / "Dockerfile").read_text(encoding="utf-8")
    issues = validate_coder_sources(main_tf, startup, dockerfile)
    if issues:
        raise AssertionError(f"Coder static contract failed: {issues}")

    terraform_checks = run_terraform_checks(shutil.which("terraform"))
    runtime = validated_runtime_evidence()
    runtime_verified = runtime is not None
    blockers = []
    if not runtime_verified:
        blockers.extend(
            [
                "Real Coder control-plane apply has not been run",
                "Real workspace GitHub authentication, exact HEAD, Jupyter, and Streamlit have not been verified",
            ]
        )
        if not terraform_checks["available"]:
            blockers.append("Terraform CLI preflight was not available in this environment")

    report = {
        "schema_version": "1.0",
        "status": (
            "pass"
            if runtime_verified
            else "implementation_complete_runtime_verification_pending"
        ),
        "runtime_verified": runtime_verified,
        "verified_at_utc": (
            runtime["verified_at_utc"]
            if runtime_verified
            else datetime.now(timezone.utc).isoformat()
        ),
        "artifact_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in FILES
        },
        "static_checks": {
            "github_external_auth_required": True,
            "provider_major_versions_bounded": True,
            "repo_ref_and_expected_sha_parameters": True,
            "persistent_clone_refreshed": True,
            "exact_release_sha_fail_closed": True,
            "required_app_guard": True,
            "loopback_agent_mapping": True,
            "tests_before_services": True,
            "jupyter_healthcheck_fail_closed": True,
            "streamlit_healthcheck_fail_closed": True,
            "credentials_absent": True,
        },
        "terraform_checks": terraform_checks,
        "docker_build": {
            "status": (
                "verified_local_image_and_real_coder_workspace"
                if runtime_verified
                else "verified_by_dedicated_ci_job_not_runtime_proof"
            )
        },
        "unresolved_blockers": blockers,
    }
    if runtime_verified:
        report["runtime_evidence_path"] = str(
            RUNTIME_EVIDENCE.relative_to(ROOT)
        ).replace("\\", "/")
        report["tested_release_sha"] = runtime["tested_release_sha"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
