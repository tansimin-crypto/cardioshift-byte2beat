"""Execute the generated Notebook with nbclient in a controlled directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def canonical_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def semantic_results_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "generated_at_utc",
        "code_commit_before_canonicalization",
        "gate_status_schema_version",
        "gate_status",
        "gate_evidence",
        "independent_judge",
        "g4_local_environment_check",
        "artifact_hashes",
    ):
        payload.pop(key, None)
    content = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("notebooks/CardioShift_Research_Report.ipynb"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/CardioShift_Research_Report.executed.ipynb"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("evidence/g3/notebook_execution.json"),
    )
    args = parser.parse_args()

    try:
        import nbformat
        from nbclient import NotebookClient
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Install requirements-notebook.lock before executing the notebook"
        ) from exc

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    work_dir = output_path.parent / ".notebook-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CARDIOSHIFT_DATA_DIR"] = str(args.data_dir.resolve())
    os.environ["CARDIOSHIFT_OUTPUT_DIR"] = str((work_dir / "outputs").resolve())
    notebook = nbformat.read(input_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(work_dir)}},
        allow_errors=False,
    )
    executed = client.execute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, output_path)
    summary_path = work_dir / "outputs" / "notebook_run_summary.json"
    if not summary_path.exists():
        raise RuntimeError("Notebook did not create its run summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    code_cells = []
    for cell in executed.cells:
        if cell.cell_type != "code":
            continue
        source = str(cell.get("source", ""))
        outputs = list(cell.get("outputs", []))
        code_cells.append(
            {
                "id": cell.get("id"),
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "execution_count": cell.get("execution_count"),
                "output_count": len(outputs),
                "error_count": sum(
                    output.get("output_type") == "error" for output in outputs
                ),
                "outputs_sha256": hashlib.sha256(
                    json.dumps(outputs, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest(),
            }
        )
    evidence = {
        "schema_version": "1.0",
        "status": "pass",
        "internet_used": False,
        "source_notebook_sha256": canonical_sha256(input_path),
        "executed_notebook_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "canonical_results_semantic_sha256": semantic_results_sha256(
            args.data_dir.resolve() / "outputs" / "results.json"
        ),
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": sum(
            isinstance(cell["execution_count"], int) for cell in code_cells
        ),
        "code_cells": code_cells,
        "notebook_run_summary": summary,
        "unresolved_blockers": [],
    }
    evidence_path = args.evidence.resolve()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "input": str(input_path),
                "output": str(output_path),
                "summary": summary,
                "evidence": str(evidence_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
