"""Rules for the languages beyond Python, Go and JavaScript (specs/012-language-coverage).

Each language has the same pieces as the three in chunker.py: a walker yielding (name, kind, node, exported), a call
extractor, an import extractor and doc readers. chunker.py merges these tables into its own at import. A language's
extensions are added together with its rules, never before (tasks.md): a file stored under a language without rules
would never be re-read once the rules arrive.

Names follow research R4 (enclosing types joined with dots, namespaces left out), kinds R5, calls R7, docs R8 and
visibility R9.
"""

import re
from collections.abc import Callable, Iterator

from tree_sitter import Node

from golestan_code_mcp.chunker import CALLS as _BASE_CALLS
from golestan_code_mcp.chunker import (
    DOTTED,
    JS_FUNCTION_VALUES,
    Call,
    Found,
    Import,
    _alone,
    _comments_above,
    _js_doc,
    _js_imports,
    _name,
    _uncomment,
)

JS_CALLS = _BASE_CALLS["javascript"]  # TypeScript calls look like JavaScript's

EXTENSIONS: dict[str, str] = {}
NAMES: dict[str, str] = {}
GRAMMARS: dict[str, tuple[str, str]] = {}
WALKERS: dict = {}
CALLS: dict = {}
IMPORTS: dict = {}
SYMBOL_DOCS: dict = {}
FILE_DOCS: dict = {}
NAMESPACES: dict = {}
LEADING: dict[str, set[str]] = {}

# What a call visitor returns for a call node: (the node of the called name, 'bare' or 'attr', the dotted receiver)
Hit = tuple[Node, str, str | None] | None


def _walk_calls(visit: Callable[[Node], Hit]) -> Callable[[Node, dict[int, str]], list[Call]]:
    """A language's call extractor from a visitor that recognizes its call nodes. Comments and strings hold no call
    nodes, so they never count (FR-010)."""
    def calls(root: Node, scopes: dict[int, str]) -> list[Call]:
        found, stack = [], [(root, None)]
        while stack:
            node, caller = stack.pop()
            caller = scopes.get(node.id, caller)
            if (hit := visit(node)) is not None:
                name, form, receiver = hit
                found.append(Call(caller, name.text.decode(), form, receiver, name.start_point.row + 1))
            stack.extend((child, caller) for child in reversed(node.named_children))
        return sorted(found, key=lambda call: call.line)
    return calls


def _dotted(node: Node | None) -> str | None:
    """The receiver's text when it is a dotted name (`a.b`, `this`), else None."""
    text = node.text.decode() if node is not None else ""
    return text if DOTTED.fullmatch(text) else None


def _child(node: Node, *types: str) -> Node | None:
    return next((c for c in node.named_children if c.type in types), None)


COMMENTS = {"comment", "block_comment", "line_comment"}  # what the grammars call a comment


def _above(node: Node, data: bytes) -> list[Node]:
    """chunker's _comments_above for every grammar's comment types. A Rust line comment ends at the start of the next
    line, so its last line is the one before when it ends in column 0."""
    run, line, prev = [], node.start_point.row, node.prev_named_sibling
    while prev is not None and prev.type in COMMENTS and _alone(data, prev):
        end = prev.end_point.row - (1 if prev.end_point.column == 0 and prev.end_point.row > prev.start_point.row else 0)
        if end != line - 1:
            break
        run.append(prev)
        line, prev = prev.start_point.row, prev.prev_named_sibling
    return run[::-1]


def _modifiers(node: Node) -> set[str]:
    return {c.text.decode() for c in node.children if c.type == "modifier"}


def _register(language: str, extensions: tuple[str, ...], grammar: tuple[str, str], walker, calls, imports, doc,
              file_doc=None, namespaces=None, leading=None) -> None:
    EXTENSIONS.update(dict.fromkeys(extensions, language))
    GRAMMARS[language], WALKERS[language], CALLS[language] = grammar, walker, calls
    IMPORTS[language], SYMBOL_DOCS[language] = imports, doc
    for table, value in ((FILE_DOCS, file_doc), (NAMESPACES, namespaces), (LEADING, leading)):
        if value is not None:
            table[language] = value


# --- C# (US1) ---

CS_TYPES = {"class_declaration": "class", "record_declaration": "class", "interface_declaration": "interface",
            "struct_declaration": "type", "record_struct_declaration": "type", "enum_declaration": "type"}
CS_MEMBERS = {"method_declaration", "constructor_declaration"}
CS_NAMESPACES = {"namespace_declaration", "file_scoped_namespace_declaration"}


def _csharp(body: Node, prefix: str = "") -> Iterator[Found]:
    for node in body.named_children:
        if node.type in CS_NAMESPACES:  # left out of names; a file-scoped one's declarations are its siblings
            yield from _csharp(node.child_by_field_name("body") or node, prefix)
        elif node.type in CS_TYPES:
            name = prefix + _name(node)
            yield name, CS_TYPES[node.type], node, "public" in _modifiers(node)
            members = node.child_by_field_name("body")
            if members is not None and node.type != "enum_declaration":
                yield from _csharp(members, name + ".")
        elif node.type in CS_MEMBERS and prefix and _child(node, "block", "arrow_expression_clause"):
            yield prefix + _name(node), "method", node, "public" in _modifiers(node)  # bodiless members are not


def _identifier(node: Node) -> Node:
    """The name node of a possibly generic (`List<int>`) or qualified (`System.Text.Builder`) name."""
    if node.type == "generic_name":
        return _child(node, "identifier") or node
    if node.type == "qualified_name":
        return _identifier(node.child_by_field_name("name"))
    return node


def _csharp_call(node: Node) -> Hit:
    if node.type == "object_creation_expression":  # new T(...) reaches the class T
        return _identifier(node.child_by_field_name("type")), "bare", None
    if node.type != "invocation_expression":
        return None
    callee = node.child_by_field_name("function")
    if callee.type in ("identifier", "generic_name"):
        return _identifier(callee), "bare", None
    if callee.type == "member_access_expression":
        return _identifier(callee.child_by_field_name("name")), "attr", _dotted(callee.child_by_field_name("expression"))
    if callee.type == "conditional_access_expression":  # x?.M()
        binding = _child(callee, "member_binding_expression")
        if binding is not None:
            return _identifier(binding.child_by_field_name("name")), "attr", _dotted(
                callee.child_by_field_name("condition"))
    return None


def _csharp_scopes(root: Node) -> Iterator[Node]:
    """The compilation unit's and every namespace body's direct children."""
    for node in root.named_children:
        yield node
        if node.type in CS_NAMESPACES and (body := node.child_by_field_name("body")) is not None:
            yield from _csharp_scopes(body)


def _csharp_imports(root: Node) -> Iterator[Import]:
    for node in _csharp_scopes(root):
        if node.type == "using_directive":
            alias = node.child_by_field_name("name")
            target = next((c for c in node.named_children if (alias is None or c.id != alias.id)
                           and c.type in ("qualified_name", "identifier", "generic_name", "alias_qualified_name")), None)
            if target is not None:
                yield Import(target.text.decode(), 0, None)  # a type, not a module: never bound (research R6)


def _csharp_namespaces(root: Node, prefix: str = "") -> list[str]:
    found = []
    for node in root.named_children:
        if node.type in CS_NAMESPACES:
            name = prefix + node.child_by_field_name("name").text.decode()
            found.append(name)
            if (body := node.child_by_field_name("body")) is not None:
                found += _csharp_namespaces(body, name + ".")
    return found


XML_REF = re.compile(r"""<(?:see|seealso|paramref|typeparamref)\s+\w+\s*=\s*["']([^"']*)["']\s*/>""")
XML_TAG = re.compile(r"<[^>]+>")


def _csharp_doc(node: Node, data: bytes) -> str | None:
    """The text of a `///` XML doc's <summary> (or of the whole comment without one), its references as names."""
    lines = [c.text.decode()[3:] for c in _comments_above(node, data) if c.text.startswith(b"///")]
    if not lines:
        return None
    xml = "\n".join(lines)
    summary = re.search(r"<summary>(.*?)</summary>", xml, re.S)
    text = XML_REF.sub(lambda m: m.group(1).rpartition(":")[2], summary.group(1) if summary else xml)
    return XML_TAG.sub("", text)


_register("csharp", (".cs",), ("tree_sitter_c_sharp", "language"), _csharp, _walk_calls(_csharp_call),
          _csharp_imports, _csharp_doc, namespaces=_csharp_namespaces)


# --- TypeScript and TSX (US2): the JavaScript rules, plus TypeScript's declarations ---

TS_TYPES = {"interface_declaration": "interface", "type_alias_declaration": "type", "enum_declaration": "type"}
TS_CLASSES = {"class_declaration", "abstract_class_declaration"}
TS_FUNCTIONS = {"function_declaration", "generator_function_declaration"}


def _named(node: Node | None) -> str | None:
    """A node's name, or None for an anonymous one (`export default function () {}`)."""
    name = node.child_by_field_name("name") if node is not None else None
    return name.text.decode() if name is not None else None


def _typescript(root: Node) -> Iterator[Found]:
    for outer in root.named_children:
        node, exported = outer, False
        if outer.type == "export_statement":
            node, exported = outer.child_by_field_name("declaration") or _child(outer, *TS_CLASSES, *TS_FUNCTIONS), True
        if node is None:
            continue
        if node.type in ("lexical_declaration", "variable_declaration"):
            for declarator in node.named_children:
                value = declarator.child_by_field_name("value")
                if declarator.type == "variable_declarator" and value and value.type in JS_FUNCTION_VALUES:
                    yield _name(declarator), "function", outer, exported
            continue
        if (name := _named(node)) is None:
            continue
        if node.type in TS_FUNCTIONS:
            yield name, "function", outer, exported
        elif node.type in TS_TYPES:
            yield name, TS_TYPES[node.type], outer, exported
        elif node.type in TS_CLASSES:
            yield name, "class", outer, exported
            for method in node.child_by_field_name("body").named_children:
                if method.type == "method_definition" and (short := _named(method)):
                    access = _child(method, "accessibility_modifier")
                    private = short.startswith("#") or (access is not None and access.text == b"private")
                    yield f"{name}.{short}", "method", method, not private


def _until_tag(text: str | None) -> str | None:
    """A doc comment's description: the lines before its first `@param`-style (or `\\param`) tag."""
    if text is None:
        return None
    lines = []
    for line in text.splitlines():
        if line.strip().startswith(("@", "\\")):
            break
        lines.append(line)
    return "\n".join(lines)


def _typescript_imports(root: Node) -> Iterator[Import]:
    yield from _js_imports(root)
    for node in root.named_children:  # export ... from "./x" re-exports what it names
        if node.type == "export_statement" and (source := node.child_by_field_name("source")) is not None:
            path = source.text.decode()[1:-1]
            if path.startswith("."):
                yield Import(path, 0, None)


for _language, _grammar, _extensions in (("typescript", "language_typescript", (".ts", ".mts", ".cts")),
                                         ("tsx", "language_tsx", (".tsx",))):
    _register(_language, _extensions, ("tree_sitter_typescript", _grammar), _typescript, JS_CALLS,
              _typescript_imports, lambda node, data: _until_tag(_js_doc(node, data)))


# --- C and C++ (US2): one walker; C has no classes, namespaces or templates, so their branches never run for it ---

C_TYPES = {"struct_specifier": "type", "union_specifier": "type", "enum_specifier": "type",
           "class_specifier": "class"}
C_PASS_THROUGH = {"preproc_ifdef", "preproc_if", "preproc_else", "preproc_elif", "preproc_elifdef"}
C_SCOPED = {"namespace_definition", "linkage_specification"}  # their declarations are in their body


def _declared(definition: Node) -> tuple[str | None, str | None]:
    """(the last qualifier, the name) a function definition declares: `app::Engine::start` gives (Engine, start)."""
    node = definition.child_by_field_name("declarator")
    while node is not None and node.type != "function_declarator":  # through pointer and reference declarators
        node = node.child_by_field_name("declarator")
    node = node.child_by_field_name("declarator") if node is not None else None
    scopes = []
    while node is not None and node.type == "qualified_identifier":
        if (scope := node.child_by_field_name("scope")) is not None:
            scopes.append(scope.text.decode().split("<")[0])
        node = node.child_by_field_name("name")
    if node is not None and node.type == "template_function":
        node = node.child_by_field_name("name")
    return (scopes[-1] if scopes else None), (node.text.decode() if node is not None else None)


def _static(node: Node) -> bool:
    return any(c.type == "storage_class_specifier" and c.text == b"static" for c in node.children)


def _c_family(body: Node) -> Iterator[Found]:
    for outer in body.named_children:
        node = _child(outer, "function_definition", *C_TYPES) if outer.type == "template_declaration" else outer
        if node is None:
            continue
        if node.type in C_PASS_THROUGH:
            yield from _c_family(node)
        elif node.type in C_SCOPED and (inner := node.child_by_field_name("body")) is not None:
            yield from _c_family(inner)  # namespaces are left out of names
        elif node.type == "function_definition":
            owner, name = _declared(node)
            if name:
                yield (f"{owner}.{name}", "method", outer, True) if owner else (name, "function", outer, not _static(node))
        elif node.type == "type_definition" and (declarator := node.child_by_field_name("declarator")) is not None:
            yield declarator.text.decode(), "type", outer, True
        elif node.type in C_TYPES or node.type == "declaration":
            yield from _c_type(outer, node if node.type in C_TYPES else _child(node, *C_TYPES), "", True)


def _c_type(outer: Node, node: Node | None, prefix: str, exported: bool) -> Iterator[Found]:
    """A struct, union, enum or class with a body, and a class's or struct's members by their access sections."""
    if node is None or node.child_by_field_name("body") is None or (name := _named(node)) is None:
        return
    name = prefix + name
    yield name, C_TYPES[node.type], outer, exported
    if node.type == "enum_specifier":
        return
    public = node.type != "class_specifier"  # a struct's members are public until a section says otherwise
    for member in node.child_by_field_name("body").named_children:
        inner = _child(member, "function_definition", *C_TYPES) if member.type == "template_declaration" else member
        if member.type == "access_specifier":
            public = member.text.decode().strip().rstrip(":") == "public"
        elif inner is not None and inner.type == "function_definition" and (short := _declared(inner)[1]):
            yield f"{name}.{short}", "method", member, public
        elif inner is not None and inner.type in C_TYPES:
            yield from _c_type(member, inner, name + ".", public)
        elif member.type == "field_declaration":
            yield from _c_type(member, _child(member, *C_TYPES), name + ".", public)


def _c_receiver(node: Node | None) -> str | None:
    """A receiver with `->` and `::` written as dots, when it is a dotted name."""
    if node is None:
        return None
    text = node.text.decode().replace("->", ".").replace("::", ".")
    return text if DOTTED.fullmatch(text) else None


def _c_call(node: Node) -> Hit:
    if node.type == "new_expression" and (kind := node.child_by_field_name("type")) is not None:
        return _c_name(kind), "bare", None
    if node.type != "call_expression":
        return None
    callee = node.child_by_field_name("function")
    if callee.type == "identifier":
        return callee, "bare", None
    if callee.type == "template_function":
        return callee.child_by_field_name("name"), "bare", None
    if callee.type == "field_expression":  # e.start(), this->speed()
        return callee.child_by_field_name("field"), "attr", _c_receiver(callee.child_by_field_name("argument"))
    if callee.type == "qualified_identifier":  # Helper::run(), std::max(...)
        name = _c_name(callee)
        receiver = callee.text.decode()[: name.start_byte - callee.start_byte].rstrip(":")
        return name, "attr", _c_receiver_text(receiver)
    return None


def _c_name(node: Node) -> Node:
    """The last name of a possibly qualified or templated name."""
    while node.type in ("qualified_identifier", "template_function", "template_type"):
        node = node.child_by_field_name("name")
    return node


def _c_receiver_text(text: str) -> str | None:
    text = text.replace("::", ".")
    return text if DOTTED.fullmatch(text) else None


def _c_includes(root: Node) -> Iterator[Import]:
    stack = [root]
    while stack:  # includes may sit in include guards and other preprocessor blocks
        node = stack.pop()
        stack.extend(reversed(node.named_children))
        if node.type == "preproc_include" and (path := node.child_by_field_name("path")) is not None:
            if path.type == "system_lib_string":
                yield Import(path.text.decode(), 0, None)  # <vector>: brackets kept, so it never resolves to a file
            elif (content := _child(path, "string_content")) is not None:
                yield Import(content.text.decode(), 0, None)


DOXYGEN = (b"///", b"//!", b"/**", b"/*!")


def _doxygen(node: Node, data: bytes) -> str | None:
    """`///`, `//!`, `/** */` and `/*! */` comments directly above, without their markers and tags."""
    comments = [c for c in _comments_above(node, data) if c.text.startswith(DOXYGEN)]
    if not comments:
        return None
    text = "\n".join(c.text.decode()[3:] if c.text.startswith((b"///", b"//!")) else _uncomment(c.text.decode()[1:])
                     if c.text.startswith(b"/*!") else _uncomment(c.text.decode()) for c in comments)
    return _until_tag(text)


LICENSE = re.compile(r"^\W*(copyright|spdx-license-identifier|license)", re.IGNORECASE)


def _c_file_doc(root: Node, data: bytes) -> str | None:
    """A block comment opening the file, unless it is a license header."""
    first = root.named_children[0] if root.named_child_count else None
    if first is None or first.type != "comment" or not first.text.startswith(b"/*"):
        return None
    text = _uncomment(first.text.decode())
    return None if LICENSE.match(text) else text


for _language, _module, _extensions in (("c", "tree_sitter_c", (".c",)),
                                        ("cpp", "tree_sitter_cpp", (".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh",
                                                                    ".hxx", ".h"))):
    _register(_language, _extensions, (_module, "language"), _c_family, _walk_calls(_c_call), _c_includes,
              _doxygen, file_doc=_c_file_doc)


# --- Java (US4) ---

JAVA_TYPES = {"class_declaration": "class", "record_declaration": "class", "interface_declaration": "interface",
              "annotation_type_declaration": "interface", "enum_declaration": "type"}
JAVA_MEMBERS = {"method_declaration", "constructor_declaration"}


def _java_public(node: Node) -> bool:
    modifiers = _child(node, "modifiers")
    return modifiers is not None and any(c.type == "public" for c in modifiers.children)


def _java(body: Node, prefix: str = "") -> Iterator[Found]:
    for node in body.named_children:
        if node.type in JAVA_TYPES:
            name = prefix + _name(node)
            yield name, JAVA_TYPES[node.type], node, _java_public(node)
            members = node.child_by_field_name("body")
            if members is not None:  # an enum's methods sit in its enum_body_declarations
                inner = _child(members, "enum_body_declarations") if node.type == "enum_declaration" else members
                yield from _java(inner or members, name + ".")
        elif node.type in JAVA_MEMBERS and prefix and node.child_by_field_name("body") is not None:
            yield prefix + _name(node), "method", node, _java_public(node)


def _java_call(node: Node) -> Hit:
    if node.type == "method_invocation":
        receiver = node.child_by_field_name("object")
        if receiver is None:
            return node.child_by_field_name("name"), "bare", None
        return node.child_by_field_name("name"), "attr", _dotted(receiver)
    if node.type == "object_creation_expression":
        kind = node.child_by_field_name("type")
        while kind.type in ("generic_type", "scoped_type_identifier"):  # List<String>, a.B
            kind = kind.named_children[-1] if kind.type == "scoped_type_identifier" else kind.named_children[0]
        return kind, "bare", None
    return None


def _java_imports(root: Node) -> Iterator[Import]:
    for node in root.named_children:
        if node.type == "import_declaration" and (name := _child(node, "scoped_identifier", "identifier")):
            wildcard = any(c.type == "asterisk" for c in node.children)
            yield Import(name.text.decode() + (".*" if wildcard else ""), 0, None)  # types: never bound (R6)


def _block_doc(node: Node, data: bytes) -> str | None:
    """The `/** */` comment directly above, without its markers and tags (Javadoc, KDoc, PHPDoc)."""
    above = _above(node, data)[-1:]
    return _until_tag(_uncomment(above[0].text.decode())) if above and above[0].text.startswith(b"/**") else None


_register("java", (".java",), ("tree_sitter_java", "language"), _java, _walk_calls(_java_call), _java_imports,
          _block_doc)


# --- Kotlin (US4) ---


def _kotlin_public(node: Node) -> bool:
    modifiers = _child(node, "modifiers")
    visibility = _child(modifiers, "visibility_modifier") if modifiers is not None else None
    return visibility is None or visibility.text.decode() not in ("private", "internal")


def _kotlin_kind(node: Node) -> str:
    if any(c.type == "interface" for c in node.children):
        return "interface"
    return "type" if _child(node, "enum_class_body") is not None else "class"


def _kotlin(body: Node, prefix: str = "") -> Iterator[Found]:
    for node in body.named_children:
        if node.type in ("class_declaration", "object_declaration"):
            name = prefix + _name(node)
            yield name, "class" if node.type == "object_declaration" else _kotlin_kind(node), node, _kotlin_public(node)
            if (members := _child(node, "class_body", "enum_class_body")) is not None:
                yield from _kotlin(members, name + ".")
        elif node.type == "companion_object" and (members := _child(node, "class_body")) is not None:
            yield from _kotlin(members, prefix)  # transparent: its members are the class's
        elif node.type == "function_declaration" and _child(node, "function_body") is not None:
            name = node.child_by_field_name("name")
            receiver = next((c for c in node.named_children if c.end_byte <= name.start_byte
                             and c.type in ("user_type", "nullable_type")), None)
            if receiver is not None:  # fun String.slug(): a method of String
                yield f"{receiver.text.decode().rstrip('?').split('<')[0]}.{name.text.decode()}", "method", node, \
                    _kotlin_public(node)
            else:
                yield prefix + name.text.decode(), "method" if prefix else "function", node, _kotlin_public(node)


def _kotlin_call(node: Node) -> Hit:
    if node.type != "call_expression" or not node.named_child_count:
        return None
    callee = node.named_children[0]
    if callee.type == "identifier":
        return callee, "bare", None
    if callee.type == "navigation_expression" and callee.named_children[-1].type == "identifier":
        return callee.named_children[-1], "attr", _dotted(callee.named_children[0])
    return None


def _kotlin_imports(root: Node) -> Iterator[Import]:
    for node in root.named_children:
        if node.type == "import" and (name := _child(node, "qualified_identifier", "identifier")):
            wildcard = any(c.type == "*" for c in node.children)
            yield Import(name.text.decode() + (".*" if wildcard else ""), 0, None)


_register("kotlin", (".kt", ".kts"), ("tree_sitter_kotlin", "language"), _kotlin, _walk_calls(_kotlin_call),
          _kotlin_imports, _block_doc)


# --- Rust (US4) ---

RUST_TYPES = {"struct_item": "type", "enum_item": "type", "union_item": "type", "type_item": "type",
              "trait_item": "interface"}


def _rust_public(node: Node) -> bool:
    return _child(node, "visibility_modifier") is not None


def _rust_type_name(node: Node) -> str:
    """`Stack<T>` is Stack, `a::B` is B."""
    return node.text.decode().split("<")[0].rpartition("::")[2].strip()


def _rust(body: Node) -> Iterator[Found]:
    for node in body.named_children:
        if node.type in RUST_TYPES:
            yield _name(node), RUST_TYPES[node.type], node, _rust_public(node)
        elif node.type == "function_item":
            yield _name(node), "function", node, _rust_public(node)
        elif node.type == "impl_item" and (members := node.child_by_field_name("body")) is not None:
            owner = _rust_type_name(node.child_by_field_name("type"))
            of_trait = node.child_by_field_name("trait") is not None  # a trait's methods are as public as it is
            for member in members.named_children:
                if member.type == "function_item":
                    yield f"{owner}.{_name(member)}", "method", member, of_trait or _rust_public(member)
        elif node.type == "mod_item" and (members := node.child_by_field_name("body")) is not None:
            yield from _rust(members)  # an inline module only groups: left out of names


def _rust_call(node: Node) -> Hit:
    if node.type != "call_expression":  # macro_invocation is not a call (research R7)
        return None
    callee = node.child_by_field_name("function")
    if callee.type == "generic_function":  # f::<T>()
        callee = callee.child_by_field_name("function")
    if callee.type == "identifier":
        return callee, "bare", None
    if callee.type == "field_expression":
        return callee.child_by_field_name("field"), "attr", _dotted(callee.child_by_field_name("value"))
    if callee.type == "scoped_identifier":  # Circle::new, std::mem::take
        path = callee.child_by_field_name("path")
        return callee.child_by_field_name("name"), "attr", _c_receiver(path)
    return None


def _rust_imports(root: Node) -> Iterator[Import]:
    for node in root.named_children:
        if node.type == "mod_item" and node.child_by_field_name("body") is None:
            yield Import(_name(node), 1, None)  # mod x;: a file relative to this module (level 1)
        elif node.type == "use_declaration" and (argument := node.child_by_field_name("argument")) is not None:
            if argument.type == "scoped_use_list":  # use a::b::{C, d}: the module a::b
                argument = argument.child_by_field_name("path") or argument
            path = argument.text.decode().split(" as ")[0].removesuffix("::*")
            last = path.rpartition("::")[2]
            yield Import(path, 0, last if last[:1].islower() and last not in ("self", "super", "crate") else None)


def _rust_docs(comments: list[Node], marker: str) -> str | None:
    lines = [c.child_by_field_name("doc").text.decode() for c in comments
             if _child(c, marker) is not None and c.child_by_field_name("doc") is not None]
    return "\n".join(lines) if lines else None


def _rust_doc(node: Node, data: bytes) -> str | None:
    """`///` comments above the definition, above its attributes."""
    start = node
    while (prev := start.prev_named_sibling) is not None and prev.type == "attribute_item":
        start = prev
    return _rust_docs(_above(start, data), "outer_doc_comment_marker")


def _rust_file_doc(root: Node, data: bytes) -> str | None:
    """The `//!` comments opening the file."""
    opening = []
    for node in root.named_children:
        if node.type not in ("line_comment", "block_comment"):
            break
        opening.append(node)
    return _rust_docs(opening, "inner_doc_comment_marker")


_register("rust", (".rs",), ("tree_sitter_rust", "language"), _rust, _walk_calls(_rust_call), _rust_imports,
          _rust_doc, file_doc=_rust_file_doc, leading={"attribute_item"})


# --- PHP (US5) ---

PHP_TYPES = {"class_declaration": "class", "interface_declaration": "interface", "trait_declaration": "interface",
             "enum_declaration": "type"}
PHP_REQUIRES = {"require_expression", "require_once_expression", "include_expression", "include_once_expression"}


def _php_public(node: Node) -> bool:
    visibility = _child(node, "visibility_modifier")
    return visibility is None or visibility.text == b"public"  # a method without one is public


def _php(body: Node, prefix: str = "") -> Iterator[Found]:
    for node in body.named_children:
        if node.type == "namespace_definition" and (inner := node.child_by_field_name("body")) is not None:
            yield from _php(inner, prefix)  # namespace X { }; a statement namespace's declarations are siblings
        elif node.type in PHP_TYPES and (name := _named(node)):
            yield prefix + name, PHP_TYPES[node.type], node, True
            if (members := node.child_by_field_name("body")) is not None:
                yield from _php(members, prefix + name + ".")
        elif node.type == "method_declaration" and node.child_by_field_name("body") is not None:
            yield prefix + _name(node), "method", node, _php_public(node)
        elif node.type == "function_definition":
            yield prefix + _name(node), "method" if prefix else "function", node, True


def _php_receiver(node: Node | None) -> str | None:
    """`$this` is this, `$a->b` is a.b, `A\\B` is A.B."""
    if node is None:
        return None
    text = node.text.decode().replace("->", ".").replace("::", ".").replace("\\", ".").replace("$", "").strip(".")
    return text if DOTTED.fullmatch(text) else None


def _php_last(node: Node) -> Node:
    """The last name of a possibly qualified name."""
    return _child(node, "name") or node if node.type == "qualified_name" else node


def _php_call(node: Node) -> Hit:
    if node.type == "function_call_expression":
        return _php_last(node.child_by_field_name("function")), "bare", None
    if node.type in ("member_call_expression", "nullsafe_member_call_expression"):
        return node.child_by_field_name("name"), "attr", _php_receiver(node.child_by_field_name("object"))
    if node.type == "scoped_call_expression":  # Helper::get(), self::x(), parent::x()
        return node.child_by_field_name("name"), "attr", _php_receiver(node.child_by_field_name("scope"))
    if node.type == "object_creation_expression" and (kind := _child(node, "name", "qualified_name")) is not None:
        return _php_last(kind), "bare", None
    return None


def _php_string(node: Node | None) -> str | None:
    if node is not None and node.type in ("string", "encapsed_string") and (content := _child(node, "string_content")):
        return content.text.decode()
    return None


def _php_path(argument: Node) -> str | None:
    """A literal path, or `__DIR__ . '/x'` / `dirname(__FILE__) . '/x'`: relative to the file either way."""
    if (literal := _php_string(argument)) is not None:
        return literal
    if argument.type == "binary_expression":
        right = _php_string(argument.child_by_field_name("right"))
        return right.lstrip("/") if right else None
    return None


def _php_imports(root: Node) -> Iterator[Import]:
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.named_children))
        if node.type == "namespace_use_declaration":
            for clause in node.named_children:
                if clause.type == "namespace_use_clause" and (name := _child(clause, "qualified_name", "name")):
                    yield Import(name.text.decode().lstrip("\\"), 0, None)
        elif node.type in PHP_REQUIRES and node.named_child_count and (path := _php_path(node.named_children[0])):
            yield Import(path, 1, None)  # level 1: relative to the requiring file


def _php_file_doc(root: Node, data: bytes) -> str | None:
    """A `/** */` docblock right after `<?php`, before the first declaration."""
    first = next((c for c in root.named_children if c.type != "php_tag"), None)
    if first is None or first.type != "comment" or not first.text.startswith(b"/**"):
        return None
    following = first.next_named_sibling
    if following is not None and following.type in (*PHP_TYPES, "function_definition"):
        return None  # it documents that declaration, not the file
    return _until_tag(_uncomment(first.text.decode()))


_register("php", (".php",), ("tree_sitter_php", "language_php"), _php, _walk_calls(_php_call), _php_imports,
          _block_doc, file_doc=_php_file_doc)


# --- Ruby (US5) ---

RUBY_REQUIRES = {"require": 0, "require_relative": 1}  # imports, not calls; require_relative is relative (level 1)


def _ruby_private_names(body: Node) -> set[str]:
    """Methods named by `private :a, :b` (or `private def ...`) in this body."""
    named = set()
    for node in body.named_children:
        method = node.child_by_field_name("method") if node.type == "call" else None
        if method is not None and method.text in (b"private", b"protected"):
            arguments = node.child_by_field_name("arguments")
            for argument in arguments.named_children if arguments is not None else ():
                if argument.type in ("simple_symbol", "symbol"):
                    named.add(argument.text.decode().lstrip(":"))
                elif argument.type == "method":
                    named.add(_name(argument))
    return named


def _ruby(body: Node, prefix: str = "") -> Iterator[Found]:
    private, hidden = False, _ruby_private_names(body)
    for node in body.named_children:
        if node.type == "identifier" and node.text in (b"private", b"protected", b"public"):
            private = node.text != b"public"  # a bare `private` hides the methods after it
        elif node.type in ("module", "class") and node.child_by_field_name("name") is not None:
            name = prefix + node.child_by_field_name("name").text.decode().replace("::", ".")
            yield name, "class", node, True
            if (members := node.child_by_field_name("body")) is not None:
                yield from _ruby(members, name + ".")
        elif node.type == "singleton_class" and (members := node.child_by_field_name("body")) is not None:
            yield from _ruby(members, prefix)  # class << self: class methods
        elif node.type in ("method", "singleton_method"):
            short = _name(node)
            exported = not (private and node.type == "method") and short not in hidden
            yield prefix + short, "method" if prefix else "function", node, exported
        elif node.type == "call" and (method := node.child_by_field_name("method")) is not None \
                and method.text in (b"private", b"protected"):
            for argument in (node.child_by_field_name("arguments") or node).named_children:
                if argument.type == "method":  # private def x ... end
                    yield prefix + _name(argument), "method" if prefix else "function", argument, False


def _ruby_call(node: Node) -> Hit:
    if node.type != "call" or (method := node.child_by_field_name("method")) is None:
        return None
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        if method.text.decode() in RUBY_REQUIRES or method.text in (b"private", b"protected", b"public"):
            return None
        return method, "bare", None
    return method, "attr", _dotted(receiver).replace("::", ".") if _dotted(receiver) else _c_receiver(receiver)


def _ruby_imports(root: Node) -> Iterator[Import]:
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.named_children))
        method = node.child_by_field_name("method") if node.type == "call" else None
        if method is not None and node.child_by_field_name("receiver") is None \
                and (level := RUBY_REQUIRES.get(method.text.decode())) is not None:
            arguments = node.child_by_field_name("arguments")
            first = arguments.named_children[0] if arguments is not None and arguments.named_child_count else None
            if first is not None and first.type == "string" and (content := _child(first, "string_content")):
                yield Import(content.text.decode(), level, None)


def _ruby_doc(node: Node, data: bytes) -> str | None:
    """The `#` lines directly above, read from the source: the grammar may attach them to the enclosing body."""
    lines = data.decode("utf-8", "replace").splitlines()[:node.start_point.row]
    comments = []
    while lines and lines[-1].strip().startswith("#") and not lines[-1].strip().startswith("#!"):
        comments.append(lines.pop().strip()[1:])
    return "\n".join(reversed(comments)) or None


_register("ruby", (".rb", ".rake"), ("tree_sitter_ruby", "language"), _ruby, _walk_calls(_ruby_call), _ruby_imports,
          _ruby_doc)
NAMES.update({"Rakefile": "ruby", "Gemfile": "ruby"})


# --- Swift (US5) ---

SWIFT_KINDS = {"class": "class", "actor": "class", "struct": "type", "enum": "type"}


def _swift_public(node: Node) -> bool:
    modifiers = _child(node, "modifiers")
    visibility = _child(modifiers, "visibility_modifier") if modifiers is not None else None
    return visibility is None or visibility.text.decode().split("(")[0] not in ("private", "fileprivate", "internal")


def _swift(body: Node, prefix: str = "") -> Iterator[Found]:
    for node in body.named_children:
        if node.type == "class_declaration":
            kind = node.child_by_field_name("declaration_kind").text.decode()
            name = node.child_by_field_name("name").text.decode().split("<")[0]
            if kind != "extension":  # an extension is not a definition; its members are the type's
                yield prefix + name, SWIFT_KINDS.get(kind, "class"), node, _swift_public(node)
            if (members := node.child_by_field_name("body")) is not None:
                yield from _swift(members, (prefix if kind != "extension" else "") + name + ".")
        elif node.type == "protocol_declaration":
            yield prefix + _name(node), "interface", node, _swift_public(node)
        elif node.type == "function_declaration" and node.child_by_field_name("body") is not None:
            yield prefix + _name(node), "method" if prefix else "function", node, _swift_public(node)
        elif node.type == "init_declaration" and prefix:
            yield prefix + "init", "method", node, _swift_public(node)


def _swift_call(node: Node) -> Hit:
    if node.type != "call_expression" or not node.named_child_count:
        return None
    callee = node.named_children[0]
    if callee.type == "simple_identifier":
        return callee, "bare", None
    if callee.type == "navigation_expression":
        suffix = callee.child_by_field_name("suffix")
        name = suffix.child_by_field_name("suffix") if suffix is not None else None
        if name is not None and name.type == "simple_identifier":
            return name, "attr", _dotted(callee.child_by_field_name("target"))
    return None


def _swift_imports(root: Node) -> Iterator[Import]:
    for node in root.named_children:
        if node.type == "import_declaration" and (module := _child(node, "identifier")) is not None:
            yield Import(module.text.decode(), 0, None)  # a module, never a file (FR-014)


def _swift_doc(node: Node, data: bytes) -> str | None:
    lines = [c.text.decode()[3:] for c in _above(node, data) if c.text.startswith(b"///")]
    return "\n".join(lines) if lines else _block_doc(node, data)


_register("swift", (".swift",), ("tree_sitter_swift", "language"), _swift, _walk_calls(_swift_call),
          _swift_imports, _swift_doc)
