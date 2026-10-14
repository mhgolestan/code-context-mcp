"""Rules for the languages beyond Python, Go and JavaScript (specs/012-language-coverage), on the fixture files under
tests/fixtures/languages/<language>/. No database."""

from pathlib import Path

import pytest

from golestan_code_mcp.chunker import Call, Import, detect_language, parse

FIXTURES = Path(__file__).parent / "fixtures" / "languages"


def parsed(language: str, rel: str):
    return parse(language, (FIXTURES / rel).read_text(encoding="utf-8"))


def definitions(language: str, rel: str) -> dict[str, tuple]:
    """name -> (kind, start, end, exported)"""
    return {c.name: (c.kind, c.start_line, c.end_line, c.exported) for c in parsed(language, rel)[0]}


def docs(language: str, rel: str) -> dict[str, str]:
    return {c.name: c.doc for c in parsed(language, rel)[0] if c.doc}


def calls(language: str, rel: str) -> set[Call]:
    return set(parsed(language, rel)[1])


# --- C# (US1) ---


@pytest.mark.parametrize("path", ["a.cs", "A/B/Thing.CS"])
def test_csharp_files(path):
    assert detect_language(path) == "csharp"


def test_csharp_definitions_kinds_and_visibility():
    assert definitions("csharp", "csharp/App/Driver/DriverFixture.cs") == {
        "DriverFixture": ("class", 7, 44, True),
        "DriverFixture.DriverFixture": ("method", 11, 14, True),
        "DriverFixture.Start": ("method", 19, 24, True),  # its [Obsolete] line is in the chunk
        "DriverFixture.GetWebDriver": ("method", 26, 31, False),
        "DriverFixture.Dispose": ("method", 33, 35, True),
        "DriverFixture.Options": ("class", 37, 43, True),
        "DriverFixture.Options.Timeout": ("method", 39, 42, True),
    }  # no namespace in names; the property and the field are not definitions
    assert definitions("csharp", "csharp/App/Driver/DriverFixture.Logging.cs") == {
        "DriverFixture": ("class", 3, 10, True), "DriverFixture.Log": ("method", 5, 9, False)}  # a partial part
    assert definitions("csharp", "csharp/App/Driver/BrowserDriver.cs") == {
        "BrowserDriver": ("class", 3, 15, True),
        "BrowserDriver.Configure": ("method", 7, 10, True),
        "BrowserDriver.Open": ("method", 12, 14, True),
        "IDriverFixture": ("interface", 17, 20, True),  # its bodiless member is not a definition
        "Point": ("class", 22, 22, True),
        "Size": ("type", 24, 27, True),
        "Browser": ("type", 29, 33, True),
    }
    assert definitions("csharp", "csharp/App/Steps/ProductSteps.cs") == {
        "ProductSteps": ("class", 6, 24, True),
        "ProductSteps.ProductSteps": ("method", 10, 13, True),
        "ProductSteps.OpenProduct": ("method", 15, 19, True),
        "ProductSteps.Helper": ("method", 21, 23, False),
    }


def test_csharp_xml_docs():
    assert docs("csharp", "csharp/App/Driver/DriverFixture.cs") == {
        "DriverFixture": "Holds the browser for a test run. Uses BrowserDriver.",
        "DriverFixture.Start": "Starts the browser.",
    }


def test_csharp_calls():
    assert calls("csharp", "csharp/App/Driver/DriverFixture.cs") == {
        Call("DriverFixture.Start", "GetWebDriver", "bare", None, 22),
        Call("DriverFixture.Start", "Open", "attr", "driver", 23),
        Call("DriverFixture.GetWebDriver", "BrowserDriver", "bare", None, 30),  # new BrowserDriver()
        Call("DriverFixture.GetWebDriver", "Configure", "attr", None, 30),
    }  # nothing from the comment or the string
    assert calls("csharp", "csharp/App/Driver/DriverFixture.Logging.cs") == {
        Call("DriverFixture.Log", "WriteLine", "attr", "Console", 7),
        Call("DriverFixture.Log", "Start", "attr", "this", 8),
    }
    assert calls("csharp", "csharp/App/Steps/ProductSteps.cs") == {
        Call("ProductSteps.OpenProduct", "Start", "attr", "fixture", 17),
        Call("ProductSteps.OpenProduct", "Helper", "bare", None, 18),
    }


def test_csharp_call_forms():
    source = ("class C\n{\n    void M()\n    {\n        d?.Open();\n        Get<int>();\n        a.b.Run();\n"
              "        base.M();\n        new List<int>();\n    }\n}\n")
    assert set(parse("csharp", source)[1]) == {
        Call("C.M", "Open", "attr", "d", 5), Call("C.M", "Get", "bare", None, 6), Call("C.M", "Run", "attr", "a.b", 7),
        Call("C.M", "M", "attr", "base", 8), Call("C.M", "List", "bare", None, 9)}


def test_csharp_usings_and_namespaces():
    _, _, imports, _, namespaces = parsed("csharp", "csharp/App/Steps/ProductSteps.cs")
    assert imports == [Import("App.Driver", 0, None), Import("System.Linq", 0, None)] and namespaces == ["App.Steps"]
    _, _, imports, _, namespaces = parsed("csharp", "csharp/App/Driver/DriverFixture.cs")
    assert imports == [Import("System", 0, None)] and namespaces == ["App.Driver"]
    source = ("global using static System.Math;\nusing Alias = A.B;\nnamespace Outer\n{\n    namespace Inner\n    {\n"
              "        class C\n        {\n        }\n    }\n}\n")
    _, _, imports, _, namespaces = parse("csharp", source)
    assert imports == [Import("System.Math", 0, None), Import("A.B", 0, None)]
    assert namespaces == ["Outer", "Outer.Inner"]


def test_csharp_file_that_does_not_parse():
    assert parsed("csharp", "csharp/Broken.cs") is None


# --- TypeScript, TSX, C and C++ (US2) ---


@pytest.mark.parametrize(("path", "language"), [
    ("a.ts", "typescript"), ("a.mts", "typescript"), ("a.cts", "typescript"), ("App.tsx", "tsx"), ("a.c", "c"),
    ("a.h", "cpp"), ("a.cpp", "cpp"), ("a.cc", "cpp"), ("a.cxx", "cpp"), ("a.c++", "cpp"), ("a.hpp", "cpp"),
    ("a.hh", "cpp"), ("a.hxx", "cpp"),
])
def test_us2_files(path, language):
    assert detect_language(path) == language


def test_typescript_definitions_kinds_and_visibility():
    assert definitions("typescript", "typescript/src/cart.ts") == {
        "add": ("function", 9, 11, True),
        "helper": ("function", 13, 15, False),
        "twice": ("function", 17, 17, True),
        "Shape": ("interface", 19, 21, True),
        "Pair": ("type", 23, 23, True),
        "Color": ("type", 25, 28, True),
        "Cart": ("class", 30, 44, True),
        "Cart.constructor": ("method", 31, 31, True),
        "Cart.area": ("method", 33, 35, True),
        "Cart.total": ("method", 37, 39, False),
        "Cart.#secret": ("method", 41, 43, False),
    }
    assert docs("typescript", "typescript/src/cart.ts") == {"add": "Adds two numbers."}  # stops at @param


def test_typescript_calls_and_imports():
    assert calls("typescript", "typescript/src/cart.ts") == {
        Call("add", "helper", "bare", None, 10),
        Call("twice", "add", "bare", None, 17),
        Call("Cart.area", "total", "attr", "this", 34),
        Call("Cart.area", "count", "attr", "api", 34),
        Call("Cart.total", "get", "bare", None, 38),
        Call("Cart.total", "max", "attr", "Math", 38),
        Call("Cart.#secret", "Cart", "bare", None, 42),
        Call("Cart.#secret", "area", "attr", None, 42),
    }
    assert parsed("typescript", "typescript/src/cart.ts")[2] == [
        Import("./api/client", 0, "get"), Import("./api", 0, "api"), Import("./types", 0, "Id")]
    reexport = 'export { get } from "./api/client";\nexport * from "./types";\n'
    assert parse("typescript", reexport)[2] == [Import("./api/client", 0, None), Import("./types", 0, None)]


def test_tsx_components():
    assert definitions("tsx", "tsx/App.tsx") == {"App": ("function", 3, 5, True)}
    assert calls("tsx", "tsx/App.tsx") == {Call("App", "format", "bare", None, 4)}
    assert parsed("tsx", "tsx/App.tsx")[2] == [Import("./format", 0, "format")]


def test_c_definitions_calls_and_includes():
    assert definitions("c", "c/parse.c") == {
        "read_line": ("function", 4, 6, False),  # static
        "parse_config": ("function", 9, 11, True),
        "load": ("function", 13, 15, True),
        "reload": ("function", 17, 20, True),
    }
    assert docs("c", "c/parse.c") == {"parse_config": "Parses the file at path."}
    assert calls("c", "c/parse.c") == {
        Call("read_line", "printf", "bare", None, 5), Call("parse_config", "read_line", "bare", None, 10),
        Call("load", "parse_config", "bare", None, 14), Call("reload", "parse_config", "bare", None, 19)}
    assert parsed("c", "c/parse.c")[2] == [Import("config.h", 0, None), Import("<stdio.h>", 0, None)]


def test_a_header_is_read_as_cpp():
    chunks, _, _, file_doc, _ = parsed("cpp", "c/config.h")
    assert {c.name: (c.kind, c.start_line, c.end_line) for c in chunks} == {
        "config": ("type", 5, 7), "config_t": ("type", 9, 9)}  # inside the include guard; the prototype is not
    assert file_doc == "Configuration loading."


def test_cpp_classes_out_of_class_definitions_and_templates():
    assert definitions("cpp", "cpp/engine.h") == {
        "Engine": ("class", 6, 13, True),
        "Engine.speed": ("method", 9, 9, True),
        "Engine.warm": ("method", 12, 12, False),  # under private:
        "Point": ("type", 15, 17, True),
        "Mode": ("type", 19, 19, True),
    }  # the namespace is left out, and the bodiless `start` is not a definition here
    assert docs("cpp", "cpp/engine.h") == {"Engine": "An engine."}
    assert definitions("cpp", "cpp/engine.cpp") == {
        "Engine.start": ("method", 6, 10, True),  # app::Engine::start, after its last qualifier
        "maxof": ("function", 12, 15, True),  # the template line is in the chunk
        "main": ("function", 17, 21, True),
    }
    assert docs("cpp", "cpp/engine.cpp") == {"Engine.start": "Starts the engine."}


def test_cpp_calls_and_includes():
    assert calls("cpp", "cpp/engine.cpp") == {
        Call("Engine.start", "warm", "bare", None, 7), Call("Engine.start", "speed", "attr", "this", 8),
        Call("Engine.start", "run", "attr", "Helper", 9), Call("main", "start", "attr", "e", 19),
        Call("main", "maxof", "bare", None, 20)}
    assert parsed("cpp", "cpp/engine.h")[2] == [Import("<vector>", 0, None)]
    assert parsed("cpp", "cpp/engine.cpp")[2] == [Import("engine.h", 0, None)]


@pytest.mark.parametrize(("language", "rel"), [
    ("typescript", "typescript/src/broken.ts"), ("tsx", "tsx/Bad.tsx"), ("c", "c/broken.c"), ("cpp", "cpp/broken.cpp")])
def test_us2_files_that_do_not_parse(language, rel):
    assert parsed(language, rel) is None


# --- Java, Kotlin and Rust (US4) ---

ORDER_JAVA = "java/src/main/java/com/acme/Order.java"


@pytest.mark.parametrize(("path", "language"), [
    ("Order.java", "java"), ("Order.kt", "kotlin"), ("build.gradle.kts", "kotlin"), ("lib.rs", "rust")])
def test_us4_files(path, language):
    assert detect_language(path) == language


def test_java_definitions_docs_calls_and_imports():
    assert definitions("java", ORDER_JAVA) == {
        "Order": ("class", 11, 38, True),
        "Order.Order": ("method", 14, 16, True),
        "Order.total": ("method", 18, 22, True),  # its @Override line is in the chunk
        "Order.sum": ("method", 24, 26, False),
        "Order.Line": ("interface", 28, 30, False),  # its bodiless member is not a definition
        "Order.Item": ("class", 32, 32, False),
        "Order.State": ("type", 34, 37, False),
    }
    assert docs("java", ORDER_JAVA) == {"Order": "An order of items."}  # stops at @author
    assert calls("java", ORDER_JAVA) == {
        Call("Order.total", "Money", "bare", None, 20), Call("Order.total", "add", "attr", "m", 21),
        Call("Order.total", "sum", "bare", None, 21), Call("Order.total", "max", "attr", "Math", 21),
        Call("Order.sum", "size", "attr", "this.items", 25)}
    assert parsed("java", ORDER_JAVA)[2] == [
        Import("com.acme.util.Money", 0, None), Import("com.acme.util.*", 0, None), Import("java.util.List", 0, None)]


def test_kotlin_definitions_docs_calls_and_imports():
    assert definitions("kotlin", "kotlin/com/acme/Order.kt") == {
        "String.slug": ("method", 6, 8, True),  # an extension function is a method of its receiver type
        "top": ("function", 10, 12, True),
        "Order": ("class", 14, 28, True),
        "Order.total": ("method", 15, 17, True),
        "Order.sum": ("method", 19, 21, False),
        "Order.create": ("method", 24, 26, True),  # the companion object is transparent
        "Registry": ("class", 30, 34, True),
        "Registry.all": ("method", 31, 33, True),
        "Shape": ("interface", 36, 38, True),
        "Point": ("class", 40, 40, True),
        "Color": ("type", 42, 45, True),
    }
    assert docs("kotlin", "kotlin/com/acme/Order.kt") == {"String.slug": "A slug of the text."}
    assert calls("kotlin", "kotlin/com/acme/Order.kt") == {
        Call("String.slug", "lowercase", "bare", None, 7), Call("String.slug", "replace", "attr", None, 7),
        Call("Order.total", "sum", "bare", None, 16), Call("Order.total", "Money", "bare", None, 16),
        Call("Order.create", "Order", "bare", None, 25), Call("Registry.all", "listOf", "bare", None, 32),
        Call("Registry.all", "create", "attr", "Order", 32)}
    assert parsed("kotlin", "kotlin/com/acme/Order.kt")[2] == [Import("com.acme.Money", 0, None)]


def test_rust_definitions_docs_calls_and_imports():
    chunks, file_calls, imports, file_doc, _ = parsed("rust", "rust/src/lib.rs")
    assert {c.name: (c.kind, c.start_line, c.end_line, c.exported) for c in chunks} == {
        "make": ("function", 9, 11, True), "helper": ("function", 13, 15, False)}
    assert file_doc == "Geometry for the demo." and docs("rust", "rust/src/lib.rs") == {"make": "Makes a circle."}
    assert set(file_calls) == {Call("make", "new", "attr", "Circle", 10), Call("make", "scale", "attr", None, 10),
                               Call("helper", "Ok", "bare", None, 14)}
    assert imports == [Import("geometry", 1, None), Import("crate::geometry::Circle", 0, None),
                       Import("std::fmt", 0, "fmt")]  # mod is relative; a module, not a type, is bound
    assert definitions("rust", "rust/src/geometry.rs") == {
        "Circle": ("type", 2, 5, True),  # the #[derive] line is in the chunk, the doc above it is not
        "Shape": ("interface", 7, 9, True),
        "Circle.area": ("method", 12, 14, True),  # impl Shape for Circle
        "Circle.new": ("method", 18, 20, True),
        "Circle.scale": ("method", 22, 25, True),
        "Kind": ("type", 28, 30, True),
        "square": ("function", 32, 34, False),
    }
    assert docs("rust", "rust/src/geometry.rs") == {"Circle": "A circle."}
    assert calls("rust", "rust/src/geometry.rs") == {Call("Circle.area", "square", "bare", None, 13)}  # no macros


def test_rust_impl_names_drop_generics():
    source = "struct Stack<T> {\n    items: Vec<T>,\n}\n\nimpl<T> Stack<T> {\n    fn push(&mut self, x: T) {\n    }\n}\n"
    assert {c.name for c in parse("rust", source)[0]} == {"Stack", "Stack.push"}


@pytest.mark.parametrize(("language", "rel"), [
    ("java", "java/Broken.java"), ("kotlin", "kotlin/OneLine.kt"), ("rust", "rust/src/broken.rs")])
def test_us4_files_that_do_not_parse(language, rel):
    assert parsed(language, rel) is None  # OneLine.kt: the grammar's known one-line-body limit (research R1)


# --- PHP, Ruby and Swift (US5) ---


@pytest.mark.parametrize(("path", "language"), [
    ("Repo.php", "php"), ("parser.rb", "ruby"), ("tasks.rake", "ruby"), ("Rakefile", "ruby"), ("app/Gemfile", "ruby"),
    ("Shapes.swift", "swift")])
def test_us5_files(path, language):
    assert detect_language(path) == language


def test_php_definitions_docs_calls_and_imports():
    chunks, file_calls, imports, file_doc, _ = parsed("php", "php/Repo.php")
    assert {c.name: (c.kind, c.start_line, c.end_line, c.exported) for c in chunks} == {
        "Repo": ("class", 16, 39, True),
        "Repo.find": ("method", 18, 22, True),
        "Repo.count": ("method", 24, 27, True),  # no modifier: public
        "Repo.log": ("method", 29, 33, False),
        "Repo.make": ("method", 35, 38, True),
        "IRepo": ("interface", 41, 44, True),  # its bodiless member is not a definition
        "Logs": ("interface", 46, 51, True),
        "Logs.trace": ("method", 48, 50, True),
        "top": ("function", 53, 56, True),
    }  # no namespace in names
    assert file_doc == "Data access for the demo." and docs("php", "php/Repo.php") == {"Repo": "A repository of rows."}
    assert set(file_calls) == {
        Call("Repo.find", "log", "attr", "this", 20), Call("Repo.find", "get", "attr", "Helper", 21),
        Call("Repo.find", "query", "bare", None, 21), Call("Repo.log", "write", "attr", "Logger", 32),
        Call("Repo.make", "Repo", "bare", None, 37)}  # nothing from the comment
    assert imports == [Import("App\\Util\\Logger", 0, None), Import("lib/db.php", 1, None)]


def test_ruby_definitions_docs_calls_and_imports():
    assert definitions("ruby", "ruby/lib/parser.rb") == {
        "Acme": ("class", 4, 23, True),
        "Acme.Parser": ("class", 7, 22, True),
        "Acme.Parser.parse": ("method", 8, 11, True),
        "Acme.Parser.create": ("method", 13, 15, True),  # def self.create
        "Acme.Parser.lex": ("method", 19, 21, False),  # after a bare `private`
    }
    assert docs("ruby", "ruby/lib/parser.rb") == {"Acme.Parser": "Parses text into tokens. Second line of the comment."}
    assert calls("ruby", "ruby/lib/parser.rb") == {
        Call("Acme.Parser.parse", "lex", "bare", None, 9), Call("Acme.Parser.parse", "build", "attr", "self", 10),
        Call("Acme.Parser.parse", "finish", "attr", None, 10), Call("Acme.Parser.lex", "new", "attr", "Lexer", 20),
        Call("Acme.Parser.lex", "split", "attr", None, 20)}  # `new` with no arguments is not a call; nor requires
    assert parsed("ruby", "ruby/lib/parser.rb")[2] == [Import("lexer", 1, None), Import("json", 0, None)]
    assert parsed("ruby", "ruby/Rakefile") is not None


def test_ruby_private_by_name():
    source = "class A\n  def a\n  end\n\n  def b\n  end\n  private :b\nend\n"
    assert {c.name: c.exported for c in parse("ruby", source)[0]} == {"A": True, "A.a": True, "A.b": False}


def test_swift_definitions_docs_calls_and_imports():
    assert definitions("swift", "swift/Shapes.swift") == {
        "Circle": ("type", 4, 6, True),
        "Shape": ("interface", 8, 10, True),
        "Circle.area": ("method", 13, 16, True),  # in an extension of Circle
        "Circle.scale": ("method", 18, 20, False),
        "Engine": ("class", 23, 31, True),
        "Engine.init": ("method", 24, 26, True),
        "Engine.start": ("method", 28, 30, True),
        "Kind": ("type", 33, 35, True),
        "square": ("function", 37, 39, True),
    }
    assert docs("swift", "swift/Shapes.swift") == {"Circle": "A circle."}
    assert calls("swift", "swift/Shapes.swift") == {
        Call("Circle.area", "scale", "attr", "self", 14), Call("Circle.area", "square", "bare", None, 15),
        Call("Engine.init", "start", "bare", None, 25), Call("Engine.start", "Circle", "bare", None, 29),
        Call("Engine.start", "area", "attr", None, 29)}
    assert parsed("swift", "swift/Shapes.swift")[2] == [Import("Foundation", 0, None)]


@pytest.mark.parametrize(("language", "rel"), [
    ("php", "php/broken.php"), ("ruby", "ruby/broken.rb"), ("swift", "swift/Broken.swift")])
def test_us5_files_that_do_not_parse(language, rel):
    assert parsed(language, rel) is None


@pytest.mark.parametrize(("language", "source"), [
    ("csharp", "class A\n{\n}\n#pragma warning restore 3016"),  # a generated NSwag client ends like this
    ("cpp", "#ifndef A_H\n#define A_H\nstruct A {\n    int x;\n};\n#endif"),
])
def test_a_directive_on_the_last_line_without_a_newline_parses(language, source):
    chunks = parse(language, source)[0]
    assert [(c.name, c.end_line) for c in chunks] == [("A", {"csharp": 3, "cpp": 5}[language])]
