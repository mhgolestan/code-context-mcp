"""Chunker: split file text into definition-level chunks with tree-sitter, or into line windows.

The same parse also yields the calls and imports each file makes, for the code graph, and the documentation of
each definition and of the file, for the module wiki. Python, Go and JavaScript are defined here; the other languages
(specs/012-language-coverage) live in languages.py and are merged into these tables at the end of this module."""

import importlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import PurePath
from typing import NamedTuple

from tree_sitter import Language, Node, Parser

# language -> (grammar module, function returning its language), imported on the first file of that language
GRAMMARS = {"python": ("tree_sitter_python", "language"), "go": ("tree_sitter_go", "language"),
            "javascript": ("tree_sitter_javascript", "language")}

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

NAMES: dict[str, str] = {}  # file names with a language but no telling extension (Rakefile)

JS_FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}

# (name, kind, node whose lines form the chunk, whether its language counts it as public)
Found = tuple[str, str, Node, bool]


@dataclass(frozen=True)
class Chunk:
    name: str
    kind: str
    start_line: int
    end_line: int
    source: str
    doc: str | None = None  # the first paragraph of its documentation, cleaned (see _clean)
    exported: bool | None = None  # public in its language (research R9); None for a line window


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
    path = PurePath(path)
    return NAMES.get(path.name) or EXTENSIONS.get(path.suffix.lower(), "text")


@cache
def _parser(language: str) -> Parser:
    module, function = GRAMMARS[language]
    return Parser(Language(getattr(importlib.import_module(module), function)()))


def _name(node: Node) -> str:
    return node.child_by_field_name("name").text.decode()


def _public(name: str) -> bool:
    """Python's and JavaScript's rule: a leading underscore marks a name private."""
    return not name.startswith("_")


def _python(body: Node, prefix: str = "") -> Iterator[Found]:
    for outer in body.named_children:
        node = outer.child_by_field_name("definition") if outer.type == "decorated_definition" else outer
        if node.type == "function_definition":
            yield prefix + _name(node), "method" if prefix else "function", outer, _public(_name(node))
        elif node.type == "class_definition":
            name = prefix + _name(node)
            yield name, "class", outer, _public(_name(node))
            yield from _python(node.child_by_field_name("body"), name + ".")


def _go_receiver(node: Node) -> str:
    receiver = node.child_by_field_name("receiver").named_children[0].child_by_field_name("type")
    return receiver.text.decode().lstrip("*").split("[")[0].strip()


def _go_public(name: str) -> bool:
    return name[:1].isupper()


def _go(root: Node) -> Iterator[Found]:
    for node in root.named_children:
        if node.type == "function_declaration":
            yield _name(node), "function", node, _go_public(_name(node))
        elif node.type == "method_declaration":
            yield f"{_go_receiver(node)}.{_name(node)}", "method", node, _go_public(_name(node))
        elif node.type == "type_declaration":
            grouped = any(child.type == "(" for child in node.children)
            for spec in node.named_children:
                if spec.type in ("type_spec", "type_alias"):
                    kind = "interface" if spec.child_by_field_name("type").type == "interface_type" else "type"
                    yield _name(spec), kind, spec if grouped else node, _go_public(_name(spec))


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
            yield _name(node), "function", outer, _public(_name(node))
        elif node.type == "class_declaration":
            name = _name(node)
            yield name, "class", outer, _public(name)
            for method in node.child_by_field_name("body").named_children:
                if method.type == "method_definition":
                    yield f"{name}.{_name(method)}", "method", method, _public(_name(method))
        elif node.type in ("lexical_declaration", "variable_declaration"):
            for declarator in node.named_children:
                value = declarator.child_by_field_name("value")
                if declarator.type == "variable_declarator" and value and value.type in JS_FUNCTION_VALUES:
                    yield _name(declarator), "function", outer, _public(_name(declarator))


WALKERS: dict[str, Callable[[Node], Iterator[Found]]] = {"python": _python, "go": _go, "javascript": _javascript}


# The call node types with their callee field, and the attribute node type with its (receiver, name) fields.
# A language in languages.py gives a function (root, scopes) -> list[Call] instead.
CALLS: dict = {
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

DOC_LIMIT = 1000
UNDERLINE = re.compile(r"^\s*([=\-~^\"'#*+])\1{2,}\s*$")  # an RST heading's underline
QUOTES = re.compile(r"""^[A-Za-z]*(\"\"\"|'''|"|')|(\"\"\"|'''|"|')$""")


def _clean(raw: str | None) -> str | None:
    """The first paragraph of a doc that is not an RST heading, with its whitespace collapsed, at most DOC_LIMIT
    characters; None if empty. `Tagged JSON` over `~~~~` names a module; the paragraph after it says what it does."""
    paragraphs, current = [], []
    for line in [*(raw or "").splitlines(), ""]:
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append(current)
            current = []
    body = next((p for p in paragraphs if not UNDERLINE.match(p[-1])), paragraphs[0] if paragraphs else [])
    return " ".join(" ".join(body).split())[:DOC_LIMIT].rstrip() or None


def _docstring(block: Node | None) -> str | None:
    """The string a Python body or module starts with, without its quotes."""
    first = next((c for c in block.named_children if c.type != "comment"), None) if block else None
    if first is None or first.type != "expression_statement" or first.named_child_count != 1:
        return None
    string = first.named_children[0]
    return QUOTES.sub("", string.text.decode()) if string.type == "string" else None


def _alone(data: bytes, node: Node) -> bool:
    """Whether only whitespace comes before the node on its first line."""
    return not data[data.rfind(b"\n", 0, node.start_byte) + 1 : node.start_byte].strip()


def _comments_above(node: Node, data: bytes) -> list[Node]:
    """The run of comments ending on the line just above the node, each on lines of its own, in source order."""
    run, line, prev = [], node.start_point.row, node.prev_named_sibling
    while prev is not None and prev.type == "comment" and prev.end_point.row == line - 1 and _alone(data, prev):
        run.append(prev)
        line, prev = prev.start_point.row, prev.prev_named_sibling
    return run[::-1]


def _uncomment(comment: str) -> str:
    if comment.startswith("//"):
        return comment[2:]
    body = comment.removeprefix("/*").removesuffix("*/").lstrip("*")  # /* */ and /** */
    return "\n".join(re.sub(r"^\s*\*", "", line) for line in body.splitlines())


def _go_doc(node: Node, data: bytes) -> str | None:
    """The comments just above, without directives such as //go:build."""
    return "\n".join(_uncomment(c.text.decode()) for c in _comments_above(node, data)
                     if not c.text.startswith(b"//go:"))


def _js_doc(node: Node, data: bytes) -> str | None:
    above = _comments_above(node, data)[-1:]
    return _uncomment(above[0].text.decode()) if above and above[0].text.startswith(b"/**") else None


SYMBOL_DOCS = {
    "python": lambda node, data: _docstring(_definition(node).child_by_field_name("body")),
    "go": _go_doc,
    "javascript": _js_doc,
}


def _go_file_doc(root: Node, data: bytes) -> str | None:
    package = next((c for c in root.named_children if c.type == "package_clause"), None)
    return _go_doc(package, data) if package else None


# simplified: no JS file docs, read a leading block comment when a JS repository needs it
FILE_DOCS: dict = {"python": lambda root, data: _docstring(root), "go": _go_file_doc}
NAMESPACES: dict = {}  # language -> root -> the namespaces the file declares (C#)
LEADING: dict[str, set[str]] = {}  # language -> sibling node types above a definition that belong to its chunk


def _start(language: str, node: Node) -> Node:
    """The first of the attribute-like siblings directly above the node, or the node itself."""
    first, kinds = node, LEADING.get(language, ())
    while (prev := first.prev_named_sibling) is not None and prev.type in kinds:
        first = prev
    return first


def parse(language: str, text: str) -> tuple[list[Chunk], list[Call], list[Import], str | None, list[str]] | None:
    """Definition chunks, calls, imports, the file's doc and its declared namespaces, or None when there is no
    grammar or the file does not parse cleanly."""
    if language not in GRAMMARS:
        return None
    data = text.encode("utf-8")
    # a directive on the last line needs its line end (C#'s #pragma, C's #endif); the lines are unchanged by it
    root = _parser(language).parse(data if data.endswith(b"\n") else data + b"\n").root_node
    if root.has_error:
        return None
    lines = text.splitlines()
    chunks, scopes = [], {}
    for name, kind, node, exported in WALKERS[language](root):
        start, end = _start(language, node).start_point.row + 1, node.end_point.row + 1
        doc = _clean(SYMBOL_DOCS[language](node, data))
        chunks.append(Chunk(name, kind, start, end, "\n".join(lines[start - 1 : end]), doc, exported))
        scopes[_definition(node).id] = name
    file_doc = FILE_DOCS.get(language)
    calls = CALLS[language](root, scopes) if callable(CALLS[language]) else _calls(language, root, scopes)
    namespaces = NAMESPACES.get(language)
    return (chunks, calls, list(IMPORTS[language](root)), _clean(file_doc(root, data)) if file_doc else None,
            namespaces(root) if namespaces else [])


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


# The other languages' rules (specs/012-language-coverage). Imported last: languages.py uses the helpers above.
from golestan_code_mcp import languages as _languages  # noqa: E402

for _table, _theirs in [(EXTENSIONS, _languages.EXTENSIONS), (NAMES, _languages.NAMES), (GRAMMARS, _languages.GRAMMARS),
                        (WALKERS, _languages.WALKERS), (CALLS, _languages.CALLS), (IMPORTS, _languages.IMPORTS),
                        (SYMBOL_DOCS, _languages.SYMBOL_DOCS), (FILE_DOCS, _languages.FILE_DOCS),
                        (NAMESPACES, _languages.NAMESPACES), (LEADING, _languages.LEADING)]:
    _table.update(_theirs)
