#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  pixi run -e dev commit-release -- [options]

Options:
  --source-branch <name>  Branch containing the reviewed changes (default: main)
  --branch <name>         Deployment branch to update (default: release)
  --remote <name>         Git remote name (default: origin)
  --skip-smoke            Skip the local CLI help smoke test
  -h, --help              Show this help

This command commits the files produced by `bump`, pushes the source branch,
then fast-forwards the deployment branch from the remote source branch and pushes it.
It does not create a tag, GitHub Release, or package publication.
EOF
}

SOURCE_BRANCH="main"
TARGET_BRANCH="release"
REMOTE_NAME="origin"
RUN_SMOKE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      ;;
    --source-branch)
      SOURCE_BRANCH="${2:-}"
      shift 2
      ;;
    --branch)
      TARGET_BRANCH="${2:-}"
      shift 2
      ;;
    --remote)
      REMOTE_NAME="${2:-}"
      shift 2
      ;;
    --skip-smoke)
      RUN_SMOKE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${SOURCE_BRANCH}" ]]; then
  echo "Run commit-release from ${SOURCE_BRANCH}; current branch is ${CURRENT_BRANCH}." >&2
  exit 1
fi

VERSION_FILE="src/rolypoly/__init__.py"
ENV_FILE="src/setup/env_big.yaml"

if ! git diff --cached --quiet; then
  echo "The index already contains staged changes. Commit or unstage them first." >&2
  exit 1
fi

UNEXPECTED_CHANGES="$(git status --porcelain | awk '
  substr($0, 4) != "src/rolypoly/__init__.py" &&
  substr($0, 4) != "src/setup/env_big.yaml" { print }
')"
if [[ -n "${UNEXPECTED_CHANGES}" ]]; then
  echo "Unexpected working-tree changes; commit-release only accepts bump output:" >&2
  echo "${UNEXPECTED_CHANGES}" >&2
  exit 1
fi

if git diff --quiet -- "${VERSION_FILE}" || git diff --quiet -- "${ENV_FILE}"; then
  echo "Expected modified files from 'pixi run -e dev bump' were not both found." >&2
  exit 1
fi

NEW_VERSION="$(grep -Eo "__version__\s*=\s*['\"][^'\"]+['\"]" "${VERSION_FILE}" | head -n 1 | sed -E "s/.*['\"]([^'\"]+)['\"]/\1/")"
if [[ -z "${NEW_VERSION}" || ! "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Could not parse a release version from ${VERSION_FILE}." >&2
  exit 1
fi
if ! grep -Fq "rolypoly-tk >=${NEW_VERSION},<1" "${ENV_FILE}"; then
  echo "${ENV_FILE} does not contain the expected rolypoly-tk ${NEW_VERSION} constraint." >&2
  exit 1
fi

git fetch "${REMOTE_NAME}" "${SOURCE_BRANCH}" "${TARGET_BRANCH}" --tags
if git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/${SOURCE_BRANCH}" &&
   ! git merge-base --is-ancestor "${REMOTE_NAME}/${SOURCE_BRANCH}" "${SOURCE_BRANCH}"; then
  echo "${SOURCE_BRANCH} is behind or has diverged from ${REMOTE_NAME}/${SOURCE_BRANCH}. Reconcile it before releasing." >&2
  exit 1
fi

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  mkdir -p testing_folder/outputs
  pytest -q src/tests/test_cli_help_smoke.py
fi

git add "${VERSION_FILE}" "${ENV_FILE}"
git commit -m "release: bump version to v${NEW_VERSION}"
git push "${REMOTE_NAME}" "${SOURCE_BRANCH}"
git fetch "${REMOTE_NAME}" "${SOURCE_BRANCH}"
echo "Committed v${NEW_VERSION} and pushed ${REMOTE_NAME}/${SOURCE_BRANCH}."

if git show-ref --verify --quiet "refs/heads/${TARGET_BRANCH}"; then
  git checkout "${TARGET_BRANCH}"
elif git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/${TARGET_BRANCH}"; then
  git checkout -b "${TARGET_BRANCH}" "${REMOTE_NAME}/${TARGET_BRANCH}"
else
  git checkout -b "${TARGET_BRANCH}" "${SOURCE_BRANCH}"
fi

if git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/${TARGET_BRANCH}"; then
  git pull --ff-only "${REMOTE_NAME}" "${TARGET_BRANCH}"
fi
git merge --ff-only "${REMOTE_NAME}/${SOURCE_BRANCH}"
git push "${REMOTE_NAME}" "${TARGET_BRANCH}"
git checkout "${SOURCE_BRANCH}"

echo "Promoted v${NEW_VERSION} to ${REMOTE_NAME}/${TARGET_BRANCH}."
echo "No tag, GitHub Release, or package publication was performed."
echo "After the branch smoke workflow passes, run: pixi run -e dev publish-release"
