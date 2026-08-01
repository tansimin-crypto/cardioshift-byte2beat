"""Run E1 random-split and E2 leave-one-hospital-out experiments."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import EXPECTED_SITES
from src.metrics import bootstrap_intervals, evaluate_binary
from src.modeling import (
    candidates_from_config,
    fit_predict_loho_fold,
    fit_predict_random_split,
)

COHORT_PATH = ROOT / "data" / "processed" / "cardioshift_cohort.csv"
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
OUTPUTS = ROOT / "outputs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def summarize_repeats(metrics: list[dict[str, object]]) -> dict[str, object]:
    names = ("auroc", "auprc", "brier", "log_loss", "ece_10")
    summary: dict[str, object] = {}
    for name in names:
        values = np.asarray([item[name] for item in metrics], dtype=float)
        summary[name] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "repeat_percentile_2_5": float(np.quantile(values, 0.025)),
            "repeat_percentile_97_5": float(np.quantile(values, 0.975)),
        }
    return summary


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    root_seed = int(config["project"]["root_seed"])
    cohort = pd.read_csv(COHORT_PATH)
    candidates = candidates_from_config(config)
    prediction_dir = OUTPUTS / "predictions"
    audit_dir = OUTPUTS / "audit"
    metrics_dir = OUTPUTS / "metrics"
    for directory in (prediction_dir, audit_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    loho_frames: list[pd.DataFrame] = []
    loho_folds: list[dict[str, object]] = []
    ledgers: list[dict[str, object]] = []
    for fold_index, held_out_site in enumerate(EXPECTED_SITES):
        result = fit_predict_loho_fold(
            cohort,
            held_out_site,
            candidates,
            root_seed + fold_index * 10_000,
        )
        predictions = result["predictions"]
        predictions["experiment"] = "leave_one_hospital_out"
        predictions["repeat"] = fold_index
        loho_frames.append(predictions)
        ledgers.append(result["ledger"])
        fold_metrics = evaluate_binary(
            predictions["target"].to_numpy(),
            predictions["calibrated_probability"].to_numpy(),
        )
        fold_metrics["bootstrap"] = bootstrap_intervals(
            predictions["target"].to_numpy(),
            predictions["calibrated_probability"].to_numpy(),
            replicates=int(config["validation"]["bootstrap"]["replicates"]),
            seed=root_seed + 50_000 + fold_index,
            confidence_level=float(
                config["validation"]["bootstrap"]["confidence_level"]
            ),
        )
        loho_folds.append(
            {
                "held_out_site": held_out_site,
                "selected_candidate": result["selected_candidate"],
                "metrics": fold_metrics,
            }
        )
        (audit_dir / f"selection_loho_{fold_index}.json").write_text(
            json.dumps(result["selection_table"], indent=2, default=json_default) + "\n",
            encoding="utf-8",
        )

    loho_predictions = pd.concat(loho_frames, ignore_index=True)
    if loho_predictions["patient_id"].nunique() != len(cohort):
        raise AssertionError("LOHO did not predict every patient exactly once")
    loho_predictions.to_csv(prediction_dir / "loho_predictions.csv", index=False)
    (audit_dir / "loho_training_ledgers.json").write_text(
        json.dumps(ledgers, indent=2) + "\n",
        encoding="utf-8",
    )
    loho_pooled = evaluate_binary(
        loho_predictions["target"].to_numpy(),
        loho_predictions["calibrated_probability"].to_numpy(),
    )
    loho_pooled["bootstrap"] = bootstrap_intervals(
        loho_predictions["target"].to_numpy(),
        loho_predictions["calibrated_probability"].to_numpy(),
        replicates=int(config["validation"]["bootstrap"]["replicates"]),
        seed=root_seed + 60_000,
        confidence_level=float(config["validation"]["bootstrap"]["confidence_level"]),
    )

    random_splitter = StratifiedShuffleSplit(
        n_splits=int(config["validation"]["random_split"]["repeats"]),
        test_size=float(config["validation"]["random_split"]["test_size"]),
        random_state=root_seed,
    )
    random_frames: list[pd.DataFrame] = []
    random_metrics: list[dict[str, object]] = []
    random_selections: list[dict[str, object]] = []
    for repeat, (train_index, test_index) in enumerate(
        random_splitter.split(cohort, cohort["target"])
    ):
        result = fit_predict_random_split(
            cohort.iloc[train_index].reset_index(drop=True),
            cohort.iloc[test_index].reset_index(drop=True),
            candidates,
            root_seed + 100_000 + repeat * 10_000,
        )
        predictions = result["predictions"]
        predictions["experiment"] = "repeated_random_split"
        predictions["repeat"] = repeat
        random_frames.append(predictions)
        random_metrics.append(
            evaluate_binary(
                predictions["target"].to_numpy(),
                predictions["calibrated_probability"].to_numpy(),
            )
        )
        random_selections.append(
            {
                "repeat": repeat,
                "selected_candidate": result["selected_candidate"],
            }
        )
    random_predictions = pd.concat(random_frames, ignore_index=True)
    random_predictions.to_csv(
        prediction_dir / "random_split_predictions.csv",
        index=False,
    )
    (audit_dir / "random_split_selections.json").write_text(
        json.dumps(random_selections, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    results = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "core_experiments_complete",
        "study": {
            "outcome": "angiographic heart-disease presence status",
            "not_future_risk": True,
            "rows": int(len(cohort)),
            "sites": list(EXPECTED_SITES),
        },
        "reproducibility": {
            "root_seed": root_seed,
            "input_path": str(COHORT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "input_sha256": sha256(COHORT_PATH),
            "config_sha256": sha256(CONFIG_PATH),
            "python": sys.version,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "candidate_count": len(candidates),
        },
        "E1_repeated_random_split": {
            "repeats": len(random_metrics),
            "test_n_each": [item["n"] for item in random_metrics],
            "metrics_by_repeat": random_metrics,
            "summary": summarize_repeats(random_metrics),
        },
        "E2_leave_one_hospital_out": {
            "pooled": loho_pooled,
            "by_site": loho_folds,
        },
        "comparison": {
            "random_mean_minus_loho_pooled": {
                name: float(
                    results_value["mean"] - loho_pooled[name]
                )
                for name, results_value in summarize_repeats(random_metrics).items()
            },
            "note": (
                "Random-split repeat variation and LOHO patient bootstrap answer "
                "different uncertainty questions; no paired causal claim is made."
            ),
        },
    }
    results_path = metrics_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "results": str(results_path),
                "loho_pooled": loho_pooled,
                "random_summary": results["E1_repeated_random_split"]["summary"],
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
