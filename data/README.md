# Data

The source dataset is the UCI Heart Disease dataset
(DOI `10.24432/C52P4X`, CC BY 4.0).

Run:

```powershell
python scripts/download_data.py
python scripts/build_cohort.py
```

Generated local artifacts:

- `data/raw/heart+disease.zip`: immutable downloaded archive;
- `data/raw/processed.*.data`: the four selected center files;
- `data/processed/cardioshift_cohort.csv`: standardized analysis cohort;
- `data/checksums.json`: SHA-256 hashes and byte sizes;
- `data/audit.json`: schema, counts, missingness, ranges, and validation results.

`?` in the original files is parsed as missing. No imputation, encoding,
scaling, or feature selection occurs during cohort construction.
