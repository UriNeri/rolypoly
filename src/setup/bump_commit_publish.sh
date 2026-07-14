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
  --skip-release                            Push the branch but do not create a GitHub Release
  --allow-dirty                             Allow running with uncommitted changes
  -h, --help                                Show this help

Notes:
  - This command merges the latest origin/main into the deployment branch first (so the
    branch never ships stale code or a stale workflow file), then bumps version locally
    in src/rolypoly/__init__.py, refreshes src/setup/env_big.yaml, commits release files,
    and pushes to the deployment branch.
  - After pushing, it also merges the version-bump commit straight back into main and
    pushes main, so main's __init__.py never drifts from what was actually released.
  - Pushing to the deployment branch only builds/smoke-tests (no publish); it is safe to re-run.
  - Publishing to TestPyPI then PyPI is triggered solely by creating a GitHub Release
    (tag v<version> targeting the deployment branch). This script creates that release
    via `gh release create ... --generate-notes` unless --skip-release is passed, so
    contributors/notes are auto-generated and Bioconda gets a matching tagged source archive.
EOF
}

BUMP_SPEC="micro"
TARGET_BRANCH="release"
REMOTE_NAME="origin"
RUN_SMOKE=1
RUN_RELEASE=1
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
    --skip-release)
      RUN_RELEASE=0
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

# GitHub Actions runs the workflow file as it exists on the ref that triggers
# the event, not whatever is on main. Always promote main into the deployment
# branch first so release/publish behavior (and any code fixes) reflect what
# was reviewed on main - otherwise a stale release branch can silently re-run
# an old workflow and/or ship a stale bug.
git fetch "${REMOTE_NAME}" main || true
if git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/main"; then
  if ! git merge --no-edit "${REMOTE_NAME}/main" -m "Merge branch 'main' into ${TARGET_BRANCH}"; then
    echo "Automatic merge of ${REMOTE_NAME}/main into ${TARGET_BRANCH} failed (conflicts?). Resolve manually, push, and re-run." >&2
    exit 1
  fi
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

awk '
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

# add rolypoly-tk (with the new version) to the bottom (pip deps) of the env_big.yaml file:
echo "  - rolypoly-tk >=${NEW_VERSION},<1" >> src/setup/env_big.yaml

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

# Sync the version bump straight back into main so it never drifts from what
# was actually released (this is what caused main's __init__.py to still say
# an old version after several releases had already gone out from release).
MAIN_BRANCH="main"
if [[ "${TARGET_BRANCH}" != "${MAIN_BRANCH}" ]]; then
  git fetch "${REMOTE_NAME}" "${MAIN_BRANCH}" || true
  if git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/${MAIN_BRANCH}"; then
    if git show-ref --verify --quiet "refs/heads/${MAIN_BRANCH}"; then
      git checkout "${MAIN_BRANCH}"
    else
      git checkout -b "${MAIN_BRANCH}" "${REMOTE_NAME}/${MAIN_BRANCH}"
    fi
    git pull --ff-only "${REMOTE_NAME}" "${MAIN_BRANCH}"
    if git merge --no-edit "${TARGET_BRANCH}" -m "Merge branch '${TARGET_BRANCH}' into ${MAIN_BRANCH}"; then
      git push "${REMOTE_NAME}" "${MAIN_BRANCH}"
      echo "Synced version bump back into ${REMOTE_NAME}/${MAIN_BRANCH}."
    else
      echo "Automatic merge of ${TARGET_BRANCH} into ${MAIN_BRANCH} failed (conflicts?). Resolve manually and push ${MAIN_BRANCH}." >&2
    fi
  fi
fi

# Publishing to TestPyPI/PyPI is only triggered by a GitHub Release (see
# .github/workflows/pypi-release.yml), not by this branch push. Create that
# release here so the process is a single, idempotent step; --generate-notes
# has GitHub auto-draft release notes (commits/PRs/new contributors) since the
# previous tag, which also gives Bioconda a tagged source archive to fetch.
if [[ "${RUN_RELEASE}" -eq 1 ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh CLI not found; skipping GitHub Release creation. Create it manually with:" >&2
    echo "  gh release create v${NEW_VERSION} --target ${TARGET_BRANCH} --title v${NEW_VERSION} --generate-notes" >&2
  elif gh release view "v${NEW_VERSION}" >/dev/null 2>&1; then
    echo "GitHub Release v${NEW_VERSION} already exists; skipping creation." >&2
  else
    gh release create "v${NEW_VERSION}" \
      --target "${TARGET_BRANCH}" \
      --title "v${NEW_VERSION}" \
      --generate-notes
    echo "Created GitHub Release v${NEW_VERSION} targeting ${TARGET_BRANCH}; publish workflow will run now."
  fi
fi

if [[ "${CURRENT_BRANCH}" != "${TARGET_BRANCH}" ]]; then
  git checkout "${CURRENT_BRANCH}"
fi
