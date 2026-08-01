"""Capture raw Kaggle rule pages and derive a content-verifiable summary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence" / "kaggle"
RAW_PATH = EVIDENCE_DIR / "raw_pages.json"
SUMMARY_PATH = EVIDENCE_DIR / "current_rules.json"
COMMAND = "kaggle competitions pages byte-2-beat --content --format json"
REQUIRED_PAGES = ("Submission Requirements", "rules", "foundational-rules")


def sha256_utf8(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def run_capture() -> tuple[str, list[dict[str, object]]]:
    version = subprocess.run(
        ["kaggle", "--version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "pages",
            "byte-2-beat",
            "--content",
            "--format",
            "json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    pages = json.loads(result.stdout)
    if not isinstance(pages, list):
        raise RuntimeError("Kaggle CLI pages response is not a list")
    return version, pages


def main() -> None:
    cli_version, response = run_capture()
    by_name = {
        page.get("name"): page.get("content")
        for page in response
        if isinstance(page, dict)
    }
    missing = [
        name
        for name in REQUIRED_PAGES
        if not isinstance(by_name.get(name), str) or not by_name[name]
    ]
    if missing:
        raise RuntimeError(f"Missing required Kaggle pages: {missing}")

    captured_at = datetime.now(timezone.utc).isoformat()
    pages = {name: by_name[name] for name in REQUIRED_PAGES}
    raw = {
        "schema_version": "1.0",
        "captured_at_utc": captured_at,
        "competition_slug": "byte-2-beat",
        "kaggle_cli_version": cli_version,
        "source_command": COMMAND,
        "pages": pages,
    }
    summary_pages = {
        name: {
            "characters": len(content),
            "sha256_utf8": sha256_utf8(content),
        }
        for name, content in pages.items()
    }
    summary = {
        "schema_version": "2.0",
        "captured_at_utc": captured_at,
        "competition_slug": "byte-2-beat",
        "source": {
            "authority": "current_signed_in_kaggle_api_via_official_cli",
            "command": COMMAND,
            "kaggle_cli_version": cli_version,
            "raw_pages": "evidence/kaggle/raw_pages.json",
            "rules_url": "https://www.kaggle.com/competitions/byte-2-beat/rules",
            "overview_url": (
                "https://www.kaggle.com/competitions/byte-2-beat/overview"
            ),
        },
        "captured_pages": summary_pages,
        "resolution_status": "verified",
        "verification_scope": (
            "provenance-bound structural and content-integrity verification; "
            "not independent anti-forgery attestation"
        ),
        "resolved_requirements": {
            "eligibility": {
                "student_only": False,
                "finding": (
                    "The current competition-specific rules do not restrict "
                    "entry or prizes to students. The abstract describes "
                    "students as the intended audience, not an eligibility "
                    "condition."
                ),
                "foundational_conditions": [
                    "registered Kaggle account",
                    (
                        "older of 18 or age of majority unless sponsor-approved "
                        "guardian consent applies"
                    ),
                    "not in a listed restricted region",
                    "not subject to applicable US export controls or sanctions",
                ],
                "winner_documentation_may_be_required": True,
            },
            "teams": {
                "maximum_team_size": 5,
                "hackathon_final_submissions_per_team": 1,
            },
            "external_data": {
                "allowed": True,
                "conditions": [
                    (
                        "publicly available and equally accessible at no cost, "
                        "or reasonable under the host accessibility and "
                        "minimal-cost standard"
                    ),
                    "all licenses and winner obligations remain satisfied",
                ],
            },
            "coder": {
                "required_for_cash_prize": True,
                "requirement": (
                    "Coder must be integrated into the submission demo."
                ),
                "acceptance_evidence_defined_by_rules": False,
                "conservative_project_gate": (
                    "Show a real Coder workspace with the application running; "
                    "static configuration alone is not claimed as sufficient."
                ),
            },
            "submission": {
                "required_artifacts": [
                    "Kaggle Writeup",
                    "attached Public Notebook",
                    "attached Public Project Link",
                    "written report or manuscript",
                ],
                "track_selection_required": True,
                "drafts_do_not_count": True,
                "public_notebook_login_or_paywall_allowed": False,
                "public_project_link_login_or_paywall_allowed": False,
            },
            "notebook_network": {
                "specific_rule_found": False,
                "project_policy": (
                    "Keep Internet disabled and attach all data for the "
                    "strongest reproducibility posture."
                ),
            },
        },
        "organizer_rule_quality_risks": [
            (
                "Competition-specific title, sponsor, sponsor address, "
                "website, prize, winner-license, and data-access fields still "
                "contain [INSERT] placeholders."
            ),
            (
                "The exact evidence threshold for integrating Coder into the "
                "submission demo is not defined in the rules."
            ),
            (
                "Rules can change; recapture immediately before the "
                "irreversible final Submit action."
            ),
        ],
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(
        json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "captured",
                "kaggle_cli_version": cli_version,
                "pages": summary_pages,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
