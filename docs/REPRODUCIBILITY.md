# Reproducibility

## Frozen inputs

- Standardized cohort SHA-256: `961eca9f38cd639e8d4be527326e3168d3c3c8af230fd4e154d5d6ea4da4f0e9`
- Rows: 920
- Sites: Cleveland, Hungary, Switzerland, VA Long Beach
- Root seed: 20260729
- Candidate configurations: 18

## Evidence chain

1. `competition-contract.yaml` freezes submission and scientific constraints.
2. `EXPERIMENT_PROTOCOL.md` prespecifies validation and failure conditions.
3. `data/checksums.json` hashes official source files and the standardized
   cohort.
4. `outputs/predictions/loho_predictions.csv` contains one external prediction
   per source row.
5. `outputs/audit/loho_training_ledgers.json` records the exact patient IDs used
   for tuning, calibration, and final fitting.
6. `outputs/results.json` is the only judge-facing numeric source.
7. `outputs/figures/figure_manifest.json` maps every figure to source hashes.

## Commands

Run the command sequence in `README.md`. Every verification script fails with a
non-zero exit code on a violated invariant. Current clean-environment and Kaggle
Run All validation remain pending, so Gate G4 is not yet claimed.

## Artifact hashes

- `data/gate_g1_verification.json`: `e51a9b6cd8759bee80358efe6690800568e504cfa89e519e28353ed287dca1a2`
- `evidence/e3_e5/verification.json`: `0ef96921da42f31e3660730febae2da3c51799d7b70cc1734002c4a2549111de`
- `evidence/g2/verification.json`: `56f4b6e97d536152c420c69b3d9066e906185f147717bb1869673a36d7cc2f0f`
- `evidence/gates/status.json`: `8f802630dceddaf00f22ed87882521ee109994c039e3bb5c901c6994bacd56e6`
- `outputs/figures/figure_manifest.json`: `ea9098ada2c6d1118f1d51663c6606544c85ad79e53ff1c539dc7028c3311770`
- `outputs/metrics/results.json`: `7cb40312bf91695b6085a12b71db709608ec6215bf6c6f13b8946e56d50bd3c4`
- `outputs/metrics/robustness_results.json`: `11dbaf1c5f6a87e4fb36ef7d49754dca6b3befedb18aa738fb37195bb16604fc`
- `outputs/metrics/shift_safety_results.json`: `f10ba94744c0afae21308202fed26f347ac93a038090f61b1bef8e6d7a8ec54c`
