"""Verify the standalone CardioShift public release and optional clean rooms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".hcl",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tf",
    ".txt",
    ".yaml",
    ".yml",
    ".lock",
}
WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.-])"
)
def non_ssh_email_addresses(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in EMAIL.finditer(text)
        if match.group(0).lower().lstrip("-") != "git@github.com"
    }

SECRET_ASSIGNMENT = re.compile(
    r"(?i)(access[_-]?token|api[_-]?key|client[_-]?secret|"
    r"refresh[_-]?token|authorization|cookie|password)"
    r"\s*[\"']?\s*[:=]\s*[\"'][^\"']+[\"']"
)
FORBIDDEN_PARTS = {
    ".git",
    ".pytest_cache",
    ".terraform",
    ".tools",
    ".venv",
    "__pycache__",
}
FORBIDDEN_PREFIXES = (
    "evidence/g4/",
    "evidence/judge/",
)
REQUIRED_FILES = (
    "README.md",
    "KAGGLE_WRITEUP.md",
    "DEMO_SCRIPT.md",
    "CODER_DEMO.md",
    "RELEASE_URLS.json",
    "MANIFEST.json",
    "app.py",
    "requirements.lock",
    "requirements-notebook.lock",
    "requirements-demo.lock",
    "outputs/results.json",
    "outputs/predictions/safety_loho_predictions.csv",
    "notebooks/CardioShift_Research_Report.ipynb",
    "coder/main.tf",
    "coder/.terraform.lock.hcl",
    "evidence/kaggle/current_rules.json",
    "evidence/kaggle/raw_pages.json",
)


def canonical_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
        "Dockerfile",
        "LICENSE",
        "known_hosts",
    }:
        content = content.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
    return content


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    passed = completed.returncode == 0
    local_path = re.compile(r"[A-Za-z]:\\[^\s\"']+")
    display_command = local_path.sub("<LOCAL_PATH>", " ".join(command))
    return {
        "command": display_command,
        "returncode": completed.returncode,
        "stdout_tail": "" if passed else local_path.sub(
            "<LOCAL_PATH>", completed.stdout[-4000:]
        ),
        "stderr_tail": "" if passed else local_path.sub(
            "<LOCAL_PATH>", completed.stderr[-4000:]
        ),
        "passed": passed,
    }


def verify_manifest(root: Path) -> tuple[list[str], int]:
    failures: list[str] = []
    manifest_path = root / "MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"invalid MANIFEST.json: {exc}"], 0
    records = manifest.get("files")
    if not isinstance(records, dict):
        return ["MANIFEST files must be an object"], 0
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if set(records) != actual:
        missing = sorted(set(records) - actual)
        extra = sorted(actual - set(records))
        failures.append(f"manifest mismatch missing={missing} extra={extra}")
    for relative, record in records.items():
        path = root / relative
        if not path.is_file() or not isinstance(record, dict):
            continue
        if record.get("bytes") != len(canonical_bytes(path)):
            failures.append(f"byte count mismatch: {relative}")
        if record.get("sha256") != canonical_sha256(path):
            failures.append(f"hash mismatch: {relative}")
    if manifest.get("standalone_public_release") is not True:
        failures.append("standalone_public_release is not true")
    if manifest.get("submission_ready_claimed") is not False:
        failures.append("submission_ready_claimed must remain false")
    return failures, len(records)


def scan_release(root: Path) -> list[str]:
    failures: list[str] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            failures.append(f"missing required file: {relative}")
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if any(part in FORBIDDEN_PARTS for part in path.relative_to(root).parts):
            failures.append(f"forbidden path: {relative}")
        if any(relative.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            failures.append(f"private evidence included: {relative}")
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "Dockerfile":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            failures.append(f"non-UTF-8 public text file: {relative}")
            continue
        if WINDOWS_ABSOLUTE.search(text):
            failures.append(f"absolute Windows path found: {relative}")
        email_addresses = non_ssh_email_addresses(text)
        if email_addresses:
            failures.append(f"email address found: {relative}")
        if SECRET_ASSIGNMENT.search(text):
            failures.append(f"credential-like assignment found: {relative}")
    return failures


def run_local_contracts(root: Path) -> list[dict[str, object]]:
    checks = [
        _run(
            [
                sys.executable,
                "-c",
                (
                    "import json; from pathlib import Path; "
                    "p=Path('outputs/results.json'); "
                    "d=json.loads(p.read_text(encoding='utf-8')); "
                    "assert d['key_findings']['loho_pooled_auroc'] > 0"
                ),
            ],
            cwd=root,
        ),
        _run(
            [
                sys.executable,
                "-c",
                (
                    "import json; from pathlib import Path; "
                    "n=json.loads(Path('notebooks/CardioShift_Research_Report.ipynb')"
                    ".read_text(encoding='utf-8')); "
                    "assert n['nbformat'] == 4 and len(n['cells']) > 0"
                ),
            ],
            cwd=root,
        ),
        _run(
            [
                sys.executable,
                "scripts/verify_kaggle_rules_evidence.py",
            ],
            cwd=root,
        ),
    ]
    return checks


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _venv_streamlit(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "streamlit.exe"
    return venv / "bin" / "streamlit"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_full_clean_rooms(root: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    clean_env = dict(os.environ)
    clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="cardioshift-public-") as tmp:
        temp = Path(tmp)
        core = temp / "core"
        demo = temp / "demo"
        checks.append(_run([sys.executable, "-m", "venv", str(core)], cwd=root))
        core_python = _venv_python(core)
        if not checks[-1]["passed"]:
            return checks
        checks.append(
            _run(
                [
                    str(core_python),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements.lock",
                    "-r",
                    "requirements-notebook.lock",
                ],
                cwd=root,
                timeout=900,
            )
        )
        if not checks[-1]["passed"]:
            return checks
        checks.append(
            _run(
                [
                    str(core_python),
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--ignore=tests/test_app_contract.py",
                ],
                cwd=root,
                env=clean_env,
                timeout=900,
            )
        )
        checks.append(_run([sys.executable, "-m", "venv", str(demo)], cwd=root))
        demo_python = _venv_python(demo)
        if not checks[-1]["passed"]:
            return checks
        checks.append(
            _run(
                [
                    str(demo_python),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements-demo.lock",
                ],
                cwd=root,
                timeout=900,
            )
        )
        if not checks[-1]["passed"]:
            return checks
        checks.append(
            _run(
                [
                    str(demo_python),
                    "-c",
                    "import app; assert len(app.PAGES) == 5",
                ],
                cwd=root,
                env=clean_env,
            )
        )
        if not checks[-1]["passed"]:
            return checks

        port = _free_port()
        command = [
            str(_venv_streamlit(demo)),
            "run",
            "app.py",
            "--server.headless=true",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ]
        process = subprocess.Popen(
            command,
            cwd=root,
            env=clean_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        healthy = False
        output = ""
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/_stcore/health",
                        timeout=2,
                    ) as response:
                        if response.status == 200:
                            healthy = True
                            break
                except OSError:
                    time.sleep(0.5)
        finally:
            process.terminate()
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate(timeout=10)
        checks.append(
            {
                "command": "streamlit health smoke",
                "returncode": 0 if healthy else 1,
                "stdout_tail": "" if healthy else output[-4000:],
                "stderr_tail": "",
                "passed": healthy,
            }
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("dist/public-release"),
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evidence/public_release/verification.json"),
    )
    args = parser.parse_args()
    requested_root = args.release_root.resolve()
    extracted: tempfile.TemporaryDirectory[str] | None = None
    archive_sha256 = None
    if requested_root.is_file() and requested_root.suffix.lower() == ".zip":
        archive_sha256 = hashlib.sha256(requested_root.read_bytes()).hexdigest()
        extracted = tempfile.TemporaryDirectory(prefix="cardioshift-unpack-")
        shutil.unpack_archive(str(requested_root), extracted.name, "zip")
        entries = list(Path(extracted.name).iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            extracted.cleanup()
            raise SystemExit("release archive must contain one top-level directory")
        root = entries[0]
    else:
        root = requested_root

    failures: list[str] = []
    manifest_failures, manifest_files = verify_manifest(root)
    failures.extend(manifest_failures)
    initial_scan_failures = scan_release(root)
    failures.extend(initial_scan_failures)
    local_checks = run_local_contracts(root)
    failures.extend(
        f"local contract failed: {check['command']}"
        for check in local_checks
        if not check["passed"]
    )
    clean_room_checks: list[dict[str, object]] = []
    if args.full and not failures:
        clean_room_checks = run_full_clean_rooms(root)
        failures.extend(
            f"clean-room check failed: {check['command']}"
            for check in clean_room_checks
            if not check["passed"]
        )
    final_scan_failures = scan_release(root)
    failures.extend(
        f"post-validation scan failed: {failure}"
        for failure in final_scan_failures
        if failure not in initial_scan_failures
    )


    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_root": args.release_root.as_posix(),
        "archive_sha256": archive_sha256,
        "archive_unpacked": archive_sha256 is not None,
        "full_clean_room_requested": args.full,
        "status": "pass" if not failures else "fail",
        "submission_ready_claimed": False,
        "manifest_files": manifest_files,
        "sanitization_scan": "pass" if not final_scan_failures else "fail",
        "local_contracts": local_checks,
        "clean_room_checks": clean_room_checks,
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if extracted is not None:
        extracted.cleanup()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
