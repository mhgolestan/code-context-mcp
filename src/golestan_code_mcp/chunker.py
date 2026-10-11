"""Chunker: split file text into definition-level chunks with tree-sitter, or into line windows.

The same parse also yields the calls and imports each file makes, for the code graph."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import PurePath
from typing import NamedTuple

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


class Call(NamedTuple):
    caller: str | None  # qualified name of the innermost enclosing symbol, None at module level
    name: str  # the called name, last component only
    form: str  # 'bare' for x(...), 'attr' for r.x(...)
    receiver: str | None  # the dotted text of r for an attr call, None when r is not a dotted name
    line: int


class Import(NamedTuple):
    module: str  # dotted Python module (a from-import appends the member), Go import path, or JS relative path
    rel_level: int  # leading dots of a Python relative import
    bound: str | None  # the name the import binds in the file


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


# The call node types with their callee field, and the attribute node type with its (receiver, name) fields.
CALLS = {
    "python": ({"call": "function"}, "attribute", ("object", "attribute")),
    "go": ({"call_expression": "function"}, "selector_expression", ("operand", "field")),
    "javascript": (
        {"call_expression": "function", "new_expression": "constructor"},
        "member_expression",
        ("object", "property"),
    ),
}
DOTTED = re.compile(r"[\w.]+")


def _definition(node: Node) -> Node:
    # Decorators run where the function is defined, so their calls belong to the enclosing scope.
    return node.child_by_field_name("definition") if node.type == "decorated_definition" else node


def _text(node: Node | None) -> str | None:
    return node.text.decode() if node is not None else None


def _calls(language: str, root: Node, scopes: dict[int, str]) -> list[Call]:
    types, attribute, (receiver_field, name_field) = CALLS[language]
    found, stack = [], [(root, None)]
    while stack:
        node, caller = stack.pop()
        caller = scopes.get(node.id, caller)
        callee = node.child_by_field_name(types[node.type]) if node.type in types else None
        if callee is not None and callee.type == "identifier":
            found.append(Call(caller, callee.text.decode(), "bare", None, callee.start_point.row + 1))
        elif callee is not None and callee.type == attribute:
            name, receiver = callee.child_by_field_name(name_field), _text(callee.child_by_field_name(receiver_field))
            dotted = receiver if DOTTED.fullmatch(receiver) else None
            found.append(Call(caller, name.text.decode(), "attr", dotted, name.start_point.row + 1))
        stack.extend((child, caller) for child in reversed(node.named_children))
    return sorted(found, key=lambda call: call.line)


def _python_imports(root: Node) -> Iterator[Import]:
    for node in root.named_children:
        if node.type == "import_statement":
            for item in node.children_by_field_name("name"):
                dotted = item.child_by_field_name("name") if item.type == "aliased_import" else item
                module = dotted.text.decode()
                yield Import(module, 0, _text(item.child_by_field_name("alias")) or module.split(".")[0])
        elif node.type == "import_from_statement":
            source = node.child_by_field_name("module_name")
            level, module = 0, source.text.decode()
            if source.type == "relative_import":
                level = len(source.named_children[0].text)
                module = _text(next((c for c in source.named_children if c.type == "dotted_name"), None)) or ""
            names = node.children_by_field_name("name")
            if not names and module:  # from m import *
                yield Import(module, level, None)
            for item in names:
                dotted = item.child_by_field_name("name") if item.type == "aliased_import" else item
                member = dotted.text.decode()
                # The member may be a submodule, so the module names it; resolution falls back to its parent.
                full = f"{module}.{member}" if module else member
                yield Import(full, level, _text(item.child_by_field_name("alias")) or member)


def _go_imports(root: Node) -> Iterator[Import]:
    for node in root.named_children:
        if node.type != "import_declaration":
            continue
        for child in node.named_children:
            for spec in [child] if child.type == "import_spec" else child.named_children:
                path = spec.child_by_field_name("path").text.decode().strip('"`')
                yield Import(path, 0, _text(spec.child_by_field_name("name")) or path.rsplit("/", 1)[-1])


def _js_bound(clause: Node | None) -> list[str]:
    names = []
    for node in clause.named_children if clause else []:
        if node.type == "identifier":
            names.append(node.text.decode())
        elif node.type == "namespace_import":
            names += [c.text.decode() for c in node.named_children]
        elif node.type == "named_imports":
            names += [_text(s.child_by_field_name("alias") or s.child_by_field_name("name")) for s in node.named_children]
    return names


def _js_imports(root: Node) -> Iterator[Import]:
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.named_children))
        if node.type == "import_statement":
            path = node.child_by_field_name("source").text.decode()[1:-1]
            if path.startswith("."):
                bound = _js_bound(next((c for c in node.named_children if c.type == "import_clause"), None))
                yield from (Import(path, 0, name) for name in bound or [None])
        elif node.type == "call_expression" and _text(node.child_by_field_name("function")) == "require":
            args = node.child_by_field_name("arguments").named_children
            path = args[0].text.decode()[1:-1] if len(args) == 1 and args[0].type == "string" else ""
            if path.startswith("."):
                name = node.parent.child_by_field_name("name") if node.parent.type == "variable_declarator" else None
                yield Import(path, 0, _text(name) if name is not None and name.type == "identifier" else None)


IMPORTS = {"python": _python_imports, "go": _go_imports, "javascript": _js_imports}


def parse(language: str, text: str) -> tuple[list[Chunk], list[Call], list[Import]] | None:
    """Definition chunks, calls and imports, or None when there is no grammar or the file does not parse cleanly."""
    if language not in GRAMMARS:
        return None
    root = _parser(language).parse(text.encode("utf-8")).root_node
    if root.has_error:
        return None
    lines = text.splitlines()
    chunks, scopes = [], {}
    for name, kind, node in WALKERS[language](root):
        start, end = node.start_point.row + 1, node.end_point.row + 1
        chunks.append(Chunk(name, kind, start, end, "\n".join(lines[start - 1 : end])))
        scopes[_definition(node).id] = name
    return chunks, _calls(language, root, scopes), list(IMPORTS[language](root))


def chunk(language: str, text: str) -> list[Chunk] | None:
    """Definition chunks, or None when there is no grammar or the file does not parse cleanly."""
    parsed = parse(language, text)
    return parsed[0] if parsed else None


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
