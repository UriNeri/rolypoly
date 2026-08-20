#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  pixi run -e dev publish-release -- [options]

Options:
  --branch <name>  Deployment branch that was prepared (default: release)
  --remote <name>  Git remote name (default: origin)
  --dry-run        Validate and print the gh command without creating a release
  -h, --help       Show this help

This command creates the versioned GitHub Release. Publishing that release
triggers the GitHub Actions TestPyPI -> smoke install -> PyPI workflow.
EOF
}

TARGET_BRANCH="release"
REMOTE_NAME="origin"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      ;;
    --branch)
      TARGET_BRANCH="${2:-}"
      shift 2
      ;;
    --remote)
      REMOTE_NAME="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit or stash changes before publishing." >&2
  exit 1
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required to create the GitHub Release." >&2
  exit 1
fi

VERSION_FILE="src/rolypoly/__init__.py"
LOCAL_VERSION="$(grep -Eo "__version__\s*=\s*['\"][^'\"]+['\"]" "${VERSION_FILE}" | head -n 1 | sed -E "s/.*['\"]([^'\"]+)['\"]/\1/")"
if [[ -z "${LOCAL_VERSION}" || ! "${LOCAL_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Could not parse a release version from ${VERSION_FILE}." >&2
  exit 1
fi
RELEASE_TAG="v${LOCAL_VERSION}"

git fetch "${REMOTE_NAME}" "${TARGET_BRANCH}" --tags
REMOTE_REF="refs/remotes/${REMOTE_NAME}/${TARGET_BRANCH}"
if ! git show-ref --verify --quiet "${REMOTE_REF}"; then
  echo "Remote deployment branch not found: ${REMOTE_NAME}/${TARGET_BRANCH}" >&2
  exit 1
fi

REMOTE_VERSION_TEXT="$(git show "${REMOTE_NAME}/${TARGET_BRANCH}:${VERSION_FILE}")"
REMOTE_VERSION="$(printf '%s\n' "${REMOTE_VERSION_TEXT}" | grep -Eo "__version__\s*=\s*['\"][^'\"]+['\"]" | head -n 1 | sed -E "s/.*['\"]([^'\"]+)['\"]/\1/")"
if [[ "${REMOTE_VERSION}" != "${LOCAL_VERSION}" ]]; then
  echo "Version mismatch: local=${LOCAL_VERSION}, ${REMOTE_NAME}/${TARGET_BRANCH}=${REMOTE_VERSION:-unparseable}." >&2
  echo "Run commit-release before publish-release." >&2
  exit 1
fi

GH_COMMAND=(
  gh release create "${RELEASE_TAG}"
  --target "${TARGET_BRANCH}"
  --title "${RELEASE_TAG}"
  --generate-notes
)

gh auth status

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Validated release %s at %s/%s. Would run:' "${RELEASE_TAG}" "${REMOTE_NAME}" "${TARGET_BRANCH}"
  printf ' %q' "${GH_COMMAND[@]}"
  printf '\n'
  echo "No tag, GitHub Release, or package publication was performed."
  exit 0
fi

if gh release view "${RELEASE_TAG}" >/dev/null 2>&1; then
  echo "GitHub Release ${RELEASE_TAG} already exists; nothing to publish."
  exit 0
fi

"${GH_COMMAND[@]}"
echo "Created GitHub Release ${RELEASE_TAG} targeting ${TARGET_BRANCH}."
echo "The release workflow will build, publish to TestPyPI, smoke-test, then publish the same artifacts to PyPI."
