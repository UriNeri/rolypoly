"""Collect contextual Caveats sections without rewriting their source pages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import quote

INDEX_BEGIN = "<!-- BEGIN GENERATED CAVEATS INDEX -->"
INDEX_END = "<!-- END GENERATED CAVEATS INDEX -->"
DEFAULT_DOCS = Path(__file__).resolve().parents[2] / "docs/mkdocs_docs"


def visible_lines(markdown: str) -> list[str]:
    """Ignore comments and fenced examples when discovering section headings."""
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)
    lines = []
    fence = None
    for line in markdown.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if (match and match[1][0] == fence[0]
                    and len(match[1]) >= len(fence) and not match[2].strip()):
                fence = None
            continue
        if match:
            fence = match[1]
            continue
        lines.append(line)
    return lines


def caveat_entry(markdown: str, relative_path: str) -> str | None:
    lines = visible_lines(markdown)
    title = Path(relative_path).stem.replace("_", " ").title()
    for index, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            break
        if index + 1 < len(lines) and re.fullmatch(r"=+", lines[index + 1]):
            title = line.strip()
            break

    starts = [i for i, line in enumerate(lines) if line.strip() == "## Caveats"]
    if len(starts) > 1:
        raise ValueError(f"{relative_path}: use one ## Caveats section per page")
    if not starts:
        return None
    body = []
    for line in lines[starts[0] + 1:]:
        if re.match(r"^#{1,2}\s", line):
            break
        body.append(line)
    if not any(line.strip() and not line.startswith("#") for line in body):
        return None
    topics = [line[4:].strip() for line in body if line.startswith("### ")]
    description = f" — {'; '.join(topics)}." if topics else ""
    return f"- [{title}]({quote(relative_path)}#caveats){description}"


def collect_caveats(docs_dir: Path) -> str:
    entries = []
    for path in sorted(docs_dir.rglob("*.md")):
        if path == docs_dir / "caveats.md":
            continue
        entry = caveat_entry(path.read_text(encoding="utf-8"), path.relative_to(docs_dir).as_posix())
        if entry:
            entries.append(entry)
    return "\n".join(entries)


def update_index(existing: str, entries: str) -> str:
    if (existing.count(INDEX_BEGIN) != 1 or existing.count(INDEX_END) != 1
            or existing.index(INDEX_BEGIN) > existing.index(INDEX_END)):
        raise ValueError("Malformed caveats index markers; refusing to overwrite")
    start = existing.index(INDEX_BEGIN) + len(INDEX_BEGIN)
    end = existing.index(INDEX_END)
    return existing[:start] + "\n\n" + entries + "\n\n" + existing[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS)
    args = parser.parse_args()
    target = args.docs_dir / "caveats.md"
    existing = target.read_text(encoding="utf-8")
    updated = update_index(existing, collect_caveats(args.docs_dir))
    if updated != existing:
        target.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
