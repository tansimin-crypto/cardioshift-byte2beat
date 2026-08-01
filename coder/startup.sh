#!/usr/bin/env bash
set -euo pipefail

workspace_root="/workspace"
repo_dir="${workspace_root}/byte-to-beat"
status_dir="${workspace_root}/.coder-status"
repo_url="${CARDIOSHIFT_REPO_URL:-git@github.com:tansimin-crypto/byte-to-beat.git}"
repo_ref="${CARDIOSHIFT_REPO_REF:-codex/submission-closure}"
expected_release_sha="${CARDIOSHIFT_EXPECTED_RELEASE_SHA:-}"
core_python="/opt/cardioshift/core/bin/python"
demo_streamlit="/opt/cardioshift/demo/bin/streamlit"
demo_python="/opt/cardioshift/demo/bin/python"

fail() {
  echo "CardioShift startup failed: $*" >&2
  exit 1
}

[[ -n "${repo_ref}" ]] || fail "CARDIOSHIFT_REPO_REF is required"
[[ "${expected_release_sha}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "CARDIOSHIFT_EXPECTED_RELEASE_SHA must be a full lowercase Git SHA"

mkdir -p "${workspace_root}"
if [[ -e "${repo_dir}" && ! -d "${repo_dir}/.git" ]]; then
  fail "${repo_dir} exists but is not a Git repository"
fi
if [[ ! -d "${repo_dir}/.git" ]]; then
  git clone --no-checkout "${repo_url}" "${repo_dir}"
fi

cd "${repo_dir}"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
git remote set-url origin "${repo_url}"
if [[ -f .git/index ]] && { ! git diff --quiet || ! git diff --cached --quiet; }; then
  fail "persistent repository has tracked local changes; refusing to overwrite"
fi

git fetch --prune --no-tags origin "${repo_ref}"
fetched_sha="$(git rev-parse --verify 'FETCH_HEAD^{commit}')"
if [[ "${fetched_sha}" != "${expected_release_sha}" ]]; then
  fail "repo_ref resolved to ${fetched_sha}, expected ${expected_release_sha}"
fi

git checkout --detach --force "${expected_release_sha}"
actual_release_sha="$(git rev-parse HEAD)"
if [[ "${actual_release_sha}" != "${expected_release_sha}" ]]; then
  fail "checked-out HEAD ${actual_release_sha} does not match expected release SHA"
fi

required_release_files=(
  app.py
  scripts/build_cohort.py
  scripts/verify_gate_g1.py
  src/results_access.py
  tests/test_app_contract.py
)
for required_file in "${required_release_files[@]}"; do
  [[ -f "${required_file}" ]] || fail "required release file is absent: ${required_file}"
done

mkdir -p "${status_dir}"
rm -f "${status_dir}/tests.ok" "${status_dir}/services.ok"
printf '%s\n' "${actual_release_sha}" >"${status_dir}/release.sha"

if [[ ! -f data/raw/processed.cleveland.data ]]; then
  mkdir -p data/raw
  cp dist/kaggle/cardioshift-data/raw/*.data data/raw/
fi

"${core_python}" -m scripts.build_cohort
"${core_python}" scripts/verify_gate_g1.py --no-write
"${core_python}" scripts/verify_gate_g2.py --no-write
"${core_python}" scripts/verify_shift_safety.py --no-write
"${core_python}" scripts/verify_robustness.py --no-write
"${core_python}" -m pytest -q --ignore=tests/test_app_contract.py
"${demo_python}" -m pytest -q tests/test_app_contract.py
touch "${status_dir}/tests.ok"

nohup "${core_python}" -m jupyter lab \
  --ip=127.0.0.1 \
  --port=8888 \
  --no-browser \
  --ServerApp.token='' \
  --ServerApp.password='' \
  >"${status_dir}/jupyter.log" 2>&1 &

nohup "${demo_streamlit}" run app.py \
  --server.address=127.0.0.1 \
  --server.port=8501 \
  --server.headless=true \
  --browser.gatherUsageStats=false \
  >"${status_dir}/streamlit.log" 2>&1 &

for endpoint in \
  "http://127.0.0.1:8888/api" \
  "http://127.0.0.1:8501/_stcore/health"; do
  healthy=false
  for _ in $(seq 1 60); do
    if curl --fail --silent --show-error "${endpoint}" >/dev/null; then
      healthy=true
      break
    fi
    sleep 1
  done
  if [[ "${healthy}" != true ]]; then
    fail "health endpoint unavailable: ${endpoint}"
  fi
done

touch "${status_dir}/services.ok"
echo "Release SHA: ${actual_release_sha}"
echo "JupyterLab: /@${CODER_WORKSPACE_OWNER_NAME:-owner}/${CODER_WORKSPACE_NAME:-workspace}/apps/jupyter"
echo "CardioShift: /@${CODER_WORKSPACE_OWNER_NAME:-owner}/${CODER_WORKSPACE_NAME:-workspace}/apps/cardioshift"
echo "Logs: ${status_dir}"
