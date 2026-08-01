# Verified Coder integration

CardioShift was executed in a real local Coder workspace, not inferred from a
static Terraform plan or Docker build.

## Audited run

- Workspace: `admin/cardioshift-final-submit`
- Template version: `rc3-3d99904-jupyter`
- Tested release SHA: `6b7aadd9806f13ebedd6a3be4b09e5d8a48c440b`
- Repository transport: read-only GitHub SSH deploy key
- Linux worktree after startup: clean
- Test marker: `/workspace/.coder-status/tests.ok`
- Service marker: `/workspace/.coder-status/services.ok`
- JupyterLab: 4.6.2, API HTTP 200
- Streamlit health: `ok`
- Stop/start recovery: passed with the same exact SHA

The machine-readable evidence is
[`evidence/g5/runtime_verification.json`](evidence/g5/runtime_verification.json).
The Coder template and fail-closed startup contract are under [`coder/`](coder/).

This proves the submitted research environment and application can be created
and restarted in Coder. It does not claim a public hosted clinical service.
