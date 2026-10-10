"""Chunker: split file text into definition-level chunks with tree-sitter, or into line windows."""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import PurePath

import tree_sitter_go
import tree_sitter_javascript
import tree_sitter_python
from tree_sitter import Language, Node, Parser

GRAMMARS = {"python": tree_sitter_python, "go": tree_sitter_go, "javascript": tree_sitter_javascript}

EXTENSIONS = {
    ".py": "python",
    ".go": "go",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".sh": "shell",
    ".bash": "shell",
    ".txt": "text",
    ".css": "css",
    ".sql": "sql",
    ".tmpl": "text",
    ".proto": "proto",
    ".cfg": "ini",
    ".ini": "ini",
    ".bat": "batch",
}

JS_FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}

# (name, kind, node whose lines form the chunk)
Found = tuple[str, str, Node]


@dataclass(frozen=True)
class Chunk:
    name: str
    kind: str
    start_line: int
    end_line: int
    source: str


def detect_language(path: str) -> str:
    return EXTENSIONS.get(PurePath(path).suffix.lower(), "text")


@cache
def _parser(language: str) -> Parser:
    return Parser(Language(GRAMMARS[language].language()))


def _name(node: Node) -> str:
    return node.child_by_field_name("name").text.decode()


def _python(body: Node, prefix: str = "") -> Iterator[Found]:
    for outer in body.named_children:
        node = outer.child_by_field_name("definition") if outer.type == "decorated_definition" else outer
        if node.type == "function_definition":
            yield prefix + _name(node), "method" if prefix else "function", outer
        elif node.type == "class_definition":
            name = prefix + _name(node)
            yield name, "class", outer
            yield from _python(node.child_by_field_name("body"), name + ".")


def _go_receiver(node: Node) -> str:
    receiver = node.child_by_field_name("receiver").named_children[0].child_by_field_name("type")
    return receiver.text.decode().lstrip("*").split("[")[0].strip()


def _go(root: Node) -> Iterator[Found]:
    for node in root.named_children:
        if node.type == "function_declaration":
            yield _name(node), "function", node
        elif node.type == "method_declaration":
            yield f"{_go_receiver(node)}.{_name(node)}", "method", node
        elif node.type == "type_declaration":
            grouped = any(child.type == "(" for child in node.children)
            for spec in node.named_children:
                if spec.type in ("type_spec", "type_alias"):
                    kind = "interface" if spec.child_by_field_name("type").type == "interface_type" else "type"
                    yield _name(spec), kind, spec if grouped else node


def _javascript(root: Node) -> Iterator[Found]:
    for outer in root.named_children:
        node = outer
        if outer.type == "export_statement":
            node = outer.child_by_field_name("declaration") or next(
                (c for c in outer.named_children if c.type in ("function_declaration", "class_declaration")), None
            )
            if node is None:
                continue
        if node.type in ("function_declaration", "generator_function_declaration"):
            yield _name(node), "function", outer
        elif node.type == "class_declaration":
            name = _name(node)
            yield name, "class", outer
            for method in node.child_by_field_name("body").named_children:
                if method.type == "method_definition":
                    yield f"{name}.{_name(method)}", "method", method
        elif node.type in ("lexical_declaration", "variable_declaration"):
            for declarator in node.named_children:
                value = declarator.child_by_field_name("value")
                if declarator.type == "variable_declarator" and value and value.type in JS_FUNCTION_VALUES:
                    yield _name(declarator), "function", outer


WALKERS = {"python": _python, "go": _go, "javascript": _javascript}


def chunk(language: str, text: str) -> list[Chunk] | None:
    """Definition chunks, or None when there is no grammar or the file does not parse cleanly."""
    if language not in GRAMMARS:
        return None
    tree = _parser(language).parse(text.encode("utf-8"))
    root = tree.root_node
    if root.has_error:
        return None
    lines = text.splitlines()
    chunks = []
    for name, kind, node in WALKERS[language](root):
        start, end = node.start_point.row + 1, node.end_point.row + 1
        chunks.append(Chunk(name, kind, start, end, "\n".join(lines[start - 1 : end])))
    return chunks


def windows(name: str, text: str, size: int = 60, step: int = 50) -> list[Chunk]:
    """Overlapping line windows of kind 'chunk', named '<name>:<start>-<end>'."""
    lines = text.splitlines()
    chunks = []
    for start in range(1, len(lines) + 1, step):
        end = min(start + size - 1, len(lines))
        chunks.append(Chunk(f"{name}:{start}-{end}", "chunk", start, end, "\n".join(lines[start - 1 : end])))
        if end == len(lines):
            break
    return chunks
