# CardioShift Coder workspace

This template builds a reproducible Python 3.12 workspace with separate locked
environments:

- `/opt/cardioshift/core`: scientific pipeline, tests, and Notebook execution;
- `/opt/cardioshift/demo`: Streamlit and its compatible pandas dependency.

The separation prevents Demo dependencies from changing the accepted
scientific environment.

## Required deployment configuration

The private repository is cloned over SSH with the Coder user's generated key.
Register the public output of `coder publickey` as a **read-only** repository
Deploy Key. Never enable write access. The image contains GitHub's published
ED25519 host key, and strict host verification remains enabled.

The template also retains the Coder GitHub External Auth declaration:

```hcl
data "coder_external_auth" "github" {
  id = "github"
}
```

This preserves the required GitHub integration prompt, but the private clone
does not depend on an OAuth token with repository scope. No access token,
private key, or password is stored in this repository, Terraform output,
startup logs, or workspace files.

## Release parameters

Every Stage A run must set:

- `repo_ref`: branch, tag, or commit fetched from `origin`;
- `expected_release_sha`: the exact lowercase 40-character release commit.

For the final audited run, set both values to the audited code commit SHA. The
all-zero default is an intentionally non-runnable, fail-closed placeholder and
is not final release evidence.

On every start, including a restart with an existing persistent volume,
`startup.sh`:

1. refuses a non-Git directory or tracked local changes;
2. fetches `repo_ref` from `origin`;
3. requires that fetched ref to equal `expected_release_sha`;
4. checks out that commit in detached mode;
5. requires `git rev-parse HEAD` to match exactly;
6. verifies required release files before tests or services;
7. removes stale success markers and records `/workspace/.coder-status/release.sha` outside the Git worktree;
8. rebuilds the cohort and runs all static gates and tests;
9. starts JupyterLab and Streamlit only after tests pass;
10. fails if either health endpoint remains unavailable.

This prevents a stale persistent clone or an outdated default branch from
silently running.

## Local preflight

No real workspace is created by these commands:

```text
terraform -chdir=coder fmt -check
terraform -chdir=coder init -backend=false
terraform -chdir=coder validate
docker build -f coder/Dockerfile -t cardioshift-coder:preflight .
python scripts/verify_coder.py
```

The local Docker template rewrites loopback occurrences inside the generated
Coder agent init script to `host.docker.internal`. Non-loopback Coder access
URLs are unchanged, and the Docker host-gateway mapping is present.

## Evidence boundary

Terraform validation and Docker build are preflight checks, not runtime proof.
A real Coder server workspace subsequently passed Gate G5 for the exact audited
release recorded in `evidence/g5/runtime_verification.json`. The accepted run
proved:

- GitHub integration is presented and the read-only SSH Deploy Key clone succeeds;
- restart authentication remains valid without logging credentials;
- the actual workspace HEAD equals the audited release SHA;
- `/workspace/.coder-status/tests.ok` and `/workspace/.coder-status/services.ok` exist;
- JupyterLab and Streamlit open through Coder.

The evidence also records a successful full stop/start recovery. This runtime
proof applies only to the recorded release SHA; later product changes require a
new exact-SHA run before they may inherit the claim.
