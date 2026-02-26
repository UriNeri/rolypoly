#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  pixi run -e dev bump-commit-publish -- [options]

Options:
  --bump <major|minor|micro|patch|X.Y.Z>   Version bump type or explicit version (default: micro)
  --branch <name>                           Deployment branch to push (default: release)
  --remote <name>                           Git remote name (default: origin)
  --skip-smoke                              Skip local help-smoke test before commit
  --allow-dirty                             Allow running with uncommitted changes
  -h, --help                                Show this help

Notes:
  - This command bumps version locally in src/rolypoly/__init__.py, refreshes src/setup/env_big.yaml,
    commits release files, and pushes to the deployment branch.
  - Pushing to the deployment branch triggers the GitHub Actions publish workflow.
EOF
}

BUMP_SPEC="micro"
TARGET_BRANCH="release"
REMOTE_NAME="origin"
RUN_SMOKE=1
ALLOW_DIRTY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      ;;
    --bump)
      BUMP_SPEC="${2:-}"
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
    --allow-dirty)
      ALLOW_DIRTY=1
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

if [[ "${ALLOW_DIRTY}" -eq 0 ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree is not clean. Commit/stash changes or use --allow-dirty." >&2
    exit 1
  fi
fi

CURRENT_BRANCH="$(git branch --show-current)"

git fetch "${REMOTE_NAME}" "${TARGET_BRANCH}" || true

if git show-ref --verify --quiet "refs/heads/${TARGET_BRANCH}"; then
  git checkout "${TARGET_BRANCH}"
elif git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/${TARGET_BRANCH}"; then
  git checkout -b "${TARGET_BRANCH}" "${REMOTE_NAME}/${TARGET_BRANCH}"
else
  git checkout -b "${TARGET_BRANCH}"
fi

if git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/${TARGET_BRANCH}"; then
  git pull --ff-only "${REMOTE_NAME}" "${TARGET_BRANCH}"
fi

if [[ "${BUMP_SPEC}" == "patch" ]]; then
  BUMP_SPEC="micro"
fi

CURRENT_VERSION="$(grep -Eo "__version__\s*=\s*['\"][^'\"]+['\"]" src/rolypoly/__init__.py | head -n 1 | sed -E "s/.*['\"]([^'\"]+)['\"]/\1/")"
if [[ -z "${CURRENT_VERSION}" ]]; then
  echo "Could not parse current __version__ from src/rolypoly/__init__.py" >&2
  exit 1
fi

BASE_VERSION="${CURRENT_VERSION%%+*}"
MAJOR_PART="${BASE_VERSION%%.*}"
REST_PART="${BASE_VERSION#*.}"
MINOR_PART="${REST_PART%%.*}"
PATCH_PART="${REST_PART#*.}"

if [[ -z "${MAJOR_PART}" || -z "${MINOR_PART}" || -z "${PATCH_PART}" || ! "${MAJOR_PART}" =~ ^[0-9]+$ || ! "${MINOR_PART}" =~ ^[0-9]+$ || ! "${PATCH_PART}" =~ ^[0-9]+$ ]]; then
  echo "Unsupported current version format: ${CURRENT_VERSION}" >&2
  exit 1
fi

case "${BUMP_SPEC}" in
  major)
    NEW_VERSION="$((MAJOR_PART + 1)).0.0"
    ;;
  minor)
    NEW_VERSION="${MAJOR_PART}.$((MINOR_PART + 1)).0"
    ;;
  micro)
    NEW_VERSION="${MAJOR_PART}.${MINOR_PART}.$((PATCH_PART + 1))"
    ;;
  *)
    if [[ "${BUMP_SPEC}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      NEW_VERSION="${BUMP_SPEC}"
    else
      echo "Unsupported --bump value: ${BUMP_SPEC}. Use major|minor|micro|patch|X.Y.Z" >&2
      exit 1
    fi
    ;;
esac

awk -v new_version="${NEW_VERSION}" '
  BEGIN { updated = 0 }
  {
    if (!updated && $0 ~ /__version__[[:space:]]*=/) {
      print "__version__ = \"" new_version "\""
      updated = 1
      next
    }
    print
  }
  END {
    if (!updated) {
      exit 1
    }
  }
' src/rolypoly/__init__.py > src/rolypoly/__init__.py.tmp
mv src/rolypoly/__init__.py.tmp src/rolypoly/__init__.py

NEW_VERSION="$(grep -Eo "__version__\s*=\s*['\"][^'\"]+['\"]" src/rolypoly/__init__.py | head -n 1 | sed -E "s/.*['\"]([^'\"]+)['\"]/\1/")"
if [[ -z "${NEW_VERSION}" ]]; then
  echo "Could not parse __version__ from src/rolypoly/__init__.py" >&2
  exit 1
fi

MAJOR="${NEW_VERSION%%.*}"
REST_VERSION="${NEW_VERSION#*.}"
MINOR="${REST_VERSION%%.*}"
if [[ -z "${MAJOR}" || -z "${MINOR}" || ! "${MAJOR}" =~ ^[0-9]+$ || ! "${MINOR}" =~ ^[0-9]+$ ]]; then
  echo "Unexpected version format: ${NEW_VERSION}" >&2
  exit 1
fi
NEXT_MINOR=$((MINOR + 1))
ROLYPOLY_PIN="rolypoly-tk>=${NEW_VERSION},<${MAJOR}.${NEXT_MINOR}.0"

TMP_CONDA_EXPORT="$(mktemp)"
TMP_HEADER="$(mktemp)"
TMP_TOP="$(mktemp)"
TMP_PIP="$(mktemp)"
TMP_TOP_OUT="$(mktemp)"
TMP_PIP_OUT="$(mktemp)"
trap 'rm -f "${TMP_CONDA_EXPORT}" "${TMP_HEADER}" "${TMP_TOP}" "${TMP_PIP}" "${TMP_TOP_OUT}" "${TMP_PIP_OUT}"' EXIT
export TMP_HEADER TMP_TOP TMP_PIP

pixi workspace export conda-environment -e complete -n rolypoly-tk "${TMP_CONDA_EXPORT}"

awk '
  BEGIN { mode = "header" }
  {
    if (mode == "header") {
      print > ENVIRON["TMP_HEADER"]
      if ($0 ~ /^dependencies:$/) {
        mode = "top"
      }
      next
    }

    if (mode == "top") {
      if ($0 ~ /^- pip:$/) {
        mode = "pip"
        next
      }
      if ($0 ~ /^- /) {
        dep = $0
        sub(/^- /, "", dep)
        print dep > ENVIRON["TMP_TOP"]
      }
      next
    }

    if (mode == "pip" && $0 ~ /^  - /) {
      dep = $0
      sub(/^  - /, "", dep)
      print dep > ENVIRON["TMP_PIP"]
    }
  }
' "${TMP_CONDA_EXPORT}"

awk '
  function dep_preference(dep) {
    if (dep ~ /^pip[[:space:]]+>=/) return 4
    if (dep ~ /^[^[:space:]]+[[:space:]]+>=/) return 3
    if (dep ~ /^[^[:space:]]+[[:space:]]+~=/) return 2
    return 1
  }

  {
    dep = $0
    name = dep
    sub(/[[:space:]].*$/, "", name)

    if (name == "pip") {
      if (dep ~ /^pip[[:space:]]+>=/) {
        pip_version = dep
      } else {
        pip_plain = 1
      }
      next
    }

    if (!(name in dep_map)) {
      order[++n] = name
      dep_map[name] = dep
      next
    }

    if (dep_preference(dep) >= dep_preference(dep_map[name])) {
      dep_map[name] = dep
    }
  }

  END {
    for (i = 1; i <= n; i++) {
      key = order[i]
      if (key in dep_map) {
        print dep_map[key]
      }
    }
    if (pip_version == "") {
      pip_version = "pip >=25.1.1,<26"
    }
    print pip_version
    print "pip"
  }
' "${TMP_TOP}" > "${TMP_TOP_OUT}"

awk -v rolypoly_pin="${ROLYPOLY_PIN}" '
  {
    dep = $0
    if (dep == "-e .") {
      next
    }

    name = dep
    sub(/[[:space:]].*$/, "", name)

    if (!(name in dep_map)) {
      order[++n] = name
    }
    dep_map[name] = dep
  }

  END {
    if (!("rolypoly-tk" in dep_map)) {
      order[++n] = "rolypoly-tk"
    }
    dep_map["rolypoly-tk"] = rolypoly_pin

    for (i = 1; i <= n; i++) {
      key = order[i]
      if (key in dep_map) {
        print dep_map[key]
      }
    }
  }
' "${TMP_PIP}" > "${TMP_PIP_OUT}"

{
  cat "${TMP_HEADER}"
  while IFS= read -r dep; do
    [[ -n "${dep}" ]] && echo "- ${dep}"
  done < "${TMP_TOP_OUT}"
  echo ""
  echo "- pip:"
  while IFS= read -r dep; do
    [[ -n "${dep}" ]] && echo "  - ${dep}"
  done < "${TMP_PIP_OUT}"
} > src/setup/env_big.yaml

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  mkdir -p testing_folder/outputs
  pytest -q src/tests/test_cli_help_smoke.py
fi

git add src/rolypoly/__init__.py src/setup/env_big.yaml

if git diff --cached --quiet; then
  echo "No version change detected; nothing to commit." >&2
  if [[ "${CURRENT_BRANCH}" != "${TARGET_BRANCH}" ]]; then
    git checkout "${CURRENT_BRANCH}"
  fi
  exit 1
fi

git commit -m "release: bump version to v${NEW_VERSION}"
git push "${REMOTE_NAME}" "${TARGET_BRANCH}"

echo "Pushed release commit for v${NEW_VERSION} to ${REMOTE_NAME}/${TARGET_BRANCH}"
if [[ "${CURRENT_BRANCH}" != "${TARGET_BRANCH}" ]]; then
  git checkout "${CURRENT_BRANCH}"
fi
