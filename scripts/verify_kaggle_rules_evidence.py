"""Validate raw and derived Byte2Beat Kaggle Rules evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REQUIRED_PAGES = (
    "Submission Requirements",
    "rules",
    "foundational-rules",
)
REQUIRED_ARTIFACTS = {
    "Kaggle Writeup",
    "attached Public Notebook",
    "attached Public Project Link",
    "written report or manuscript",
}
EXPECTED_COMMAND = (
    "kaggle competitions pages byte-2-beat --content --format json"
)
EXPECTED_AUTHORITY = "current_signed_in_kaggle_api_via_official_cli"
FORBIDDEN_SECRET_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "client_secret",
    "cookie",
    "credentials",
    "kaggle_key",
    "password",
    "refresh_token",
}
CONTENT_ANCHORS = {
    "Submission Requirements": (
        "Kaggle Writeup",
        "Attached Public Notebook",
        "Attached Project Link",
        "Written report/manuscript",
        "must utilize Coder in your submission demo",
    ),
    "rules": (
        "maximum Team size is five (5)",
        "each Team may submit one (1) Submission only",
        "integrate coder into your submission demo",
        "You may use data other than the Competition Data",
    ),
    "foundational-rules": (
        "registered account holder at Kaggle.com",
        "older of 18 years old or the age of majority",
    ),
}


def _sha256_utf8(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_SECRET_KEYS:
                return True
            if _contains_secret_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def rules_evidence_verified(path: Path) -> bool:
    """Return True only when the summary is reproducible from raw content."""
    raw_path = path.with_name("raw_pages.json")
    if not path.exists() or not raw_path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return False

    if _contains_secret_key(summary) or _contains_secret_key(raw):
        return False
    if not _valid_timestamp(summary.get("captured_at_utc")):
        return False
    if summary.get("schema_version") != "2.0":
        return False
    if raw.get("schema_version") != "1.0":
        return False
    if summary.get("competition_slug") != "byte-2-beat":
        return False
    if raw.get("competition_slug") != "byte-2-beat":
        return False
    if raw.get("captured_at_utc") != summary.get("captured_at_utc"):
        return False
    if raw.get("source_command") != EXPECTED_COMMAND:
        return False

    source = summary.get("source", {})
    if not isinstance(source, dict):
        return False
    if source.get("authority") != EXPECTED_AUTHORITY:
        return False
    if source.get("command") != EXPECTED_COMMAND:
        return False
    if source.get("raw_pages") != "evidence/kaggle/raw_pages.json":
        return False
    if source.get("kaggle_cli_version") != raw.get("kaggle_cli_version"):
        return False
    if not isinstance(raw.get("kaggle_cli_version"), str):
        return False

    raw_pages = raw.get("pages")
    summary_pages = summary.get("captured_pages")
    if not isinstance(raw_pages, dict) or not isinstance(summary_pages, dict):
        return False
    if set(raw_pages) != set(REQUIRED_PAGES):
        return False
    if set(summary_pages) != set(REQUIRED_PAGES):
        return False
    for name in REQUIRED_PAGES:
        content = raw_pages.get(name)
        record = summary_pages.get(name)
        if not isinstance(content, str) or not content:
            return False
        if not isinstance(record, dict):
            return False
        if record.get("characters") != len(content):
            return False
        if record.get("sha256_utf8") != _sha256_utf8(content):
            return False
        lowered = content.casefold()
        if not all(anchor.casefold() in lowered for anchor in CONTENT_ANCHORS[name]):
            return False

    resolved = summary.get("resolved_requirements", {})
    if not isinstance(resolved, dict):
        return False
    eligibility = resolved.get("eligibility", {})
    teams = resolved.get("teams", {})
    external_data = resolved.get("external_data", {})
    coder = resolved.get("coder", {})
    submission = resolved.get("submission", {})
    notebook_network = resolved.get("notebook_network", {})
    artifacts = submission.get("required_artifacts", [])
    return (
        summary.get("resolution_status") == "verified"
        and "content-integrity" in summary.get("verification_scope", "")
        and eligibility.get("student_only") is False
        and teams.get("maximum_team_size") == 5
        and teams.get("hackathon_final_submissions_per_team") == 1
        and external_data.get("allowed") is True
        and coder.get("required_for_cash_prize") is True
        and coder.get("acceptance_evidence_defined_by_rules") is False
        and isinstance(artifacts, list)
        and REQUIRED_ARTIFACTS.issubset(set(artifacts))
        and submission.get("track_selection_required") is True
        and submission.get("drafts_do_not_count") is True
        and notebook_network.get("specific_rule_found") is False
    )


if __name__ == "__main__":
    default = (
        Path(__file__).resolve().parents[1]
        / "evidence"
        / "kaggle"
        / "current_rules.json"
    )
    if not rules_evidence_verified(default):
        raise SystemExit("Kaggle rules evidence verification failed")
    print("Kaggle rules evidence verification passed")
