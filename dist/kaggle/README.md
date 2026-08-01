# CardioShift Kaggle assets

`cardioshift-data/` is the uploadable offline Kaggle Dataset payload. It
contains the four UCI processed center files, CC BY 4.0 attribution and DOI,
checksums, canonical run artifacts, patient-level prediction artifacts, and a
source archive.

Build both the payload and Notebook:

```bash
python scripts/build_kaggle_notebook.py
```

The generated Notebook uses no network access. On Kaggle, attach the uploaded
Dataset and keep Internet disabled. Public upload and Run All remain manual
release steps.
