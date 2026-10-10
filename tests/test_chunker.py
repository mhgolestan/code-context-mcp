"""Chunker: definition-level chunks from tree-sitter, and line windows when a file cannot be parsed."""

from pathlib import Path

import pytest

from golestan_code_mcp.chunker import chunk, detect_language, windows

FIXTURES = Path(__file__).parent / "fixtures"


def outline(language: str, name: str) -> set[tuple[str, str, int, int]]:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    chunks = chunk(language, text)
    assert chunks is not None
    lines = text.splitlines()
    for c in chunks:
        assert c.source == "\n".join(lines[c.start_line - 1 : c.end_line])
    return {(c.name, c.kind, c.start_line, c.end_line) for c in chunks}


def test_python_outline():
    assert outline("python", "sample.py") == {
        ("add", "function", 6, 7),
        ("greet", "function", 10, 14),
        ("Counter", "class", 17, 31),
        ("Counter.__init__", "method", 18, 19),
        ("Counter.value", "method", 21, 23),
        ("Counter.bump", "method", 25, 27),
        ("Counter.Step", "class", 29, 31),
        ("Counter.Step.size", "method", 30, 31),
        ("square", "function", 34, 36),
    }


def test_go_outline():
    assert outline("go", "sample.go") == {
        ("Greet", "function", 6, 8),
        ("Counter", "type", 10, 12),
        ("Counter.Bump", "method", 14, 17),
        ("Greeter", "interface", 19, 21),
        ("Point", "type", 24, 26),
        ("Shape", "interface", 27, 29),
        ("Engine", "type", 32, 34),
        ("Engine.Use", "method", 36, 38),
    }


def test_javascript_outline():
    assert outline("javascript", "sample.js") == {
        ("add", "function", 3, 5),
        ("count", "function", 7, 9),
        ("Counter", "class", 11, 20),
        ("Counter.constructor", "method", 12, 14),
        ("Counter.bump", "method", 16, 19),
        ("square", "function", 22, 24),
        ("main", "function", 26, 28),
    }


def test_crlf_text_matches_lf():
    text = (FIXTURES / "sample.py").read_text(encoding="utf-8")
    assert chunk("python", text.replace("\n", "\r\n")) == chunk("python", text)


def test_parse_error_returns_none():
    assert chunk("python", (FIXTURES / "broken.py").read_text(encoding="utf-8")) is None


def test_unknown_language_returns_none():
    assert chunk("cobol", "IDENTIFICATION DIVISION.\n") is None


def spans(text: str) -> list[tuple[str, str, int, int]]:
    return [(c.name, c.kind, c.start_line, c.end_line) for c in windows("notes.md", text)]


def numbered(n: int) -> str:
    return "".join(f"line {i}\n" for i in range(1, n + 1))


def test_windows_overlap_and_stop_at_last_line():
    assert spans(numbered(130)) == [
        ("notes.md:1-60", "chunk", 1, 60),
        ("notes.md:51-110", "chunk", 51, 110),
        ("notes.md:101-130", "chunk", 101, 130),
    ]


def test_windows_source_is_the_lines():
    last = windows("notes.md", numbered(130))[-1]
    assert last.source == "\n".join(f"line {i}" for i in range(101, 131))


def test_sixty_lines_is_one_window():
    assert spans(numbered(60)) == [("notes.md:1-60", "chunk", 1, 60)]


def test_empty_text_has_no_windows():
    assert windows("notes.md", "") == []


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("pkg/app.py", "python"),
        ("main.go", "go"),
        ("index.js", "javascript"),
        ("lib/mod.MJS", "javascript"),
        ("README.md", "markdown"),
        ("docs/index.rst", "rst"),
        ("data.xyz", "text"),
        ("Makefile", "text"),
    ],
)
def test_detect_language(path, language):
    assert detect_language(path) == language
