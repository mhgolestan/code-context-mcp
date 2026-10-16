"""The layout rules: every source and test file is small, says what it holds, and is listed in the README.

- No `.py` file under src/ or tests/ is longer than 300 lines (tests/fixtures/ holds parser input and is exempt).
- Each one opens with a module docstring, whose first line states its purpose.
- The README's `## Layout` section lists exactly these files, each with a purpose. Its code block is a tree: a name
  ending in `/` is a folder, and a line indented under it names something inside it.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 300
FILES = sorted(p for folder in ("src", "tests") for p in (ROOT / folder).rglob("*.py")
               if "fixtures" not in p.relative_to(ROOT).parts)
IDS = [p.relative_to(ROOT).as_posix() for p in FILES]


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_file_is_at_most_300_lines(path):
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= LIMIT, f"{lines} lines"


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_file_opens_with_a_docstring(path):
    assert ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))), "no module docstring"


def readme_layout() -> dict[str, str]:
    """The files the README's Layout tree lists, as {path from the project root: purpose}."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(r"^## Layout\n.*?^```text\n(.*?)^```", text, re.M | re.S).group(1)
    listed, folders = {}, []  # folders: (indent, path) of the folders enclosing the current line
    for line in block.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        name, _, purpose = line.strip().partition(" ")
        while folders and folders[-1][0] >= indent:
            folders.pop()
        path = "/".join([f.rstrip("/") for _, f in folders[-1:]] + [name])
        if name.endswith("/"):
            folders.append((indent, path))
        else:
            listed[path] = purpose.strip()
    return listed


def test_readme_lists_every_source_and_test_file():
    listed = readme_layout()
    on_disk = set(IDS)
    named = {path for path in listed if path.endswith(".py")}
    assert sorted(on_disk - named) == [], "on disk but not in the README's Layout"
    assert sorted(named - on_disk) == [], "in the README's Layout but not on disk"
    assert [path for path in named if not listed[path]] == [], "listed without a purpose"
