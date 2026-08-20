#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  pixi run -e dev bump -- [major|minor|micro|patch|X.Y.Z]
  pixi run -e dev bump -- --bump <major|minor|micro|patch|X.Y.Z>

Options:
  --bump <value>  Version bump type or explicit version (default: micro)
  -h, --help      Show this help

This command only changes local files. It does not commit, push, tag, or publish.
The working tree must be clean before it runs.
EOF
}

BUMP_SPEC="micro"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      ;;
    --bump)
      BUMP_SPEC="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    major|minor|micro|patch|[0-9]*.[0-9]*.[0-9]*)
      BUMP_SPEC="$1"
      shift
      ;;
    *)
      echo "Unknown option or bump value: $1" >&2
      usage
      exit 1
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit or stash changes before bumping." >&2
  exit 1
fi

VERSION_FILE="src/rolypoly/__init__.py"
ENV_FILE="src/setup/env_big.yaml"
CURRENT_VERSION="$(grep -Eo "__version__\s*=\s*['\"][^'\"]+['\"]" "${VERSION_FILE}" | head -n 1 | sed -E "s/.*['\"]([^'\"]+)['\"]/\1/")"
if [[ -z "${CURRENT_VERSION}" ]]; then
  echo "Could not parse current __version__ from ${VERSION_FILE}" >&2
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

if [[ "${BUMP_SPEC}" == "patch" ]]; then
  BUMP_SPEC="micro"
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
      echo "Unsupported bump value: ${BUMP_SPEC}. Use major|minor|micro|patch|X.Y.Z" >&2
      exit 1
    fi
    ;;
esac

if [[ "${NEW_VERSION}" == "${BASE_VERSION}" ]]; then
  echo "Requested version is already current: ${NEW_VERSION}" >&2
  exit 1
fi

TMP_VERSION="$(mktemp)"
TMP_CONDA_EXPORT="$(mktemp)"
TMP_HEADER="$(mktemp)"
TMP_TOP="$(mktemp)"
TMP_PIP="$(mktemp)"
TMP_TOP_OUT="$(mktemp)"
TMP_PIP_OUT="$(mktemp)"
TMP_ENV="$(mktemp)"
trap 'rm -f "${TMP_VERSION}" "${TMP_CONDA_EXPORT}" "${TMP_HEADER}" "${TMP_TOP}" "${TMP_PIP}" "${TMP_TOP_OUT}" "${TMP_PIP_OUT}" "${TMP_ENV}"' EXIT
export TMP_HEADER TMP_TOP TMP_PIP

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
    if (!updated) exit 1
  }
' "${VERSION_FILE}" > "${TMP_VERSION}"

PIXI_BIN="${PIXI_EXE:-$(command -v pixi || true)}"
if [[ -z "${PIXI_BIN}" ]]; then
  echo "Could not find pixi. Run this script through 'pixi run -e dev bump'." >&2
  exit 1
fi

"${PIXI_BIN}" workspace export conda-environment -e complete -n rolypoly-tk "${TMP_CONDA_EXPORT}"

awk '
  BEGIN { mode = "header" }
  {
    if (mode == "header") {
      print > ENVIRON["TMP_HEADER"]
      if ($0 ~ /^dependencies:$/) mode = "top"
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
      if (dep ~ /^pip[[:space:]]+>=/) pip_version = dep
      next
    }
    if (!(name in dep_map)) order[++n] = name
    if (!(name in dep_map) || dep_preference(dep) >= dep_preference(dep_map[name])) dep_map[name] = dep
  }
  END {
    for (i = 1; i <= n; i++) print dep_map[order[i]]
    if (pip_version == "") pip_version = "pip >=25.1.1,<26"
    print pip_version
    print "pip"
  }
' "${TMP_TOP}" > "${TMP_TOP_OUT}"

awk '
  {
    dep = $0
    if (dep == "-e .") next
    name = dep
    sub(/[[:space:]].*$/, "", name)
    if (!(name in dep_map)) order[++n] = name
    dep_map[name] = dep
  }
  END {
    for (i = 1; i <= n; i++) print dep_map[order[i]]
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
  echo "  - rolypoly-tk >=${NEW_VERSION},<1"
} > "${TMP_ENV}"

mv "${TMP_VERSION}" "${VERSION_FILE}"
mv "${TMP_ENV}" "${ENV_FILE}"

echo "Bumped RolyPoly ${CURRENT_VERSION} -> ${NEW_VERSION}."
echo "Updated ${VERSION_FILE} and regenerated ${ENV_FILE}."
echo "No commit, push, tag, or publication was performed."
echo "Review the diff, then run: pixi run -e dev commit-release"
