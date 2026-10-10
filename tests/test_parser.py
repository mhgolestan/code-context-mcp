"""Parser smoke test: Tree-sitter finds top-level definitions and their line ranges."""

import importlib
from pathlib import Path

import pytest
from tree_sitter import Language, Parser

FIXTURES = Path(__file__).parent / "fixtures"
KINDS = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "go": {"function_declaration": "function", "method_declaration": "method", "type_declaration": "type"},
}


def outline(lang: str, path: Path) -> set[tuple[str, str, int, int]]:
    try:
        grammar = importlib.import_module(f"tree_sitter_{lang}")
    except ImportError:
        pytest.fail(f"missing grammar package: tree-sitter-{lang}")
    tree = Parser(Language(grammar.language())).parse(path.read_bytes())
    found = set()
    for node in tree.root_node.named_children:
        target = node.child_by_field_name("definition") if node.type == "decorated_definition" else node
        kind = KINDS[lang].get(target.type)
        if not kind:
            continue
        named = target.named_children[0] if target.type == "type_declaration" else target
        name = named.child_by_field_name("name").text.decode()
        found.add((name, kind, node.start_point[0] + 1, node.end_point[0] + 1))
    return found


def test_python_outline():
    assert outline("python", FIXTURES / "sample.py") == {
        ("add", "function", 6, 7),
        ("greet", "function", 10, 12),
        ("Counter", "class", 15, 21),
        ("square", "function", 24, 26),
    }


def test_go_outline():
    assert outline("go", FIXTURES / "sample.go") == {
        ("Greet", "function", 6, 8),
        ("Counter", "type", 10, 12),
        ("Bump", "method", 14, 17),
    }


def test_missing_grammar_is_named():
    with pytest.raises(pytest.fail.Exception, match="tree-sitter-cobol"):
        outline("cobol", FIXTURES / "sample.py")
