"""ccctx command line (contracts/cli.md). Database cases skip when it is unreachable."""

import time
import uuid

import pytest

from golestan_code_mcp import cli, db, registry
from golestan_code_mcp.config import load

FAKE_PW = "Not-A-Real-Pw-42"


@pytest.fixture(scope="module")
def live():
    try:
        conn = db.connect(load())
    except Exception as e:  # config missing or database down: skip, never fail
        pytest.skip(f"database unreachable: {type(e).__name__}")
    yield
    with conn.cursor() as cur:
        cur.execute("delete from ccctx_repos where name like 't-%'")
    conn.commit()
    conn.close()


def run(capsys, *argv):
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_add_then_list(live, capsys, tmp_path):
    n = f"t-{uuid.uuid4().hex[:8]}"
    code, out, _ = run(capsys, "repo", "add", str(tmp_path), "--name", n)
    assert code == 0
    assert out.strip() == f"registered {n} {tmp_path.resolve()}"

    code, out, _ = run(capsys, "repo", "list")
    assert code == 0
    header, *rows = out.splitlines()
    assert header.split() == ["NAME", "PATH", "FILES", "SYMBOLS"]
    mine = [r for r in rows if r.split()[0] == n]
    assert len(mine) == 1 and mine[0].split()[-2:] == ["0", "0"]
    assert len({len(line.rstrip()) for line in [header, *rows]}) == 1  # right-aligned counts end in one column


@pytest.mark.parametrize(("path", "name", "message"), [
    ("{tmp}/missing", "t-ok", "does not exist"),
    ("{tmp}/file.txt", "t-ok", "is not a directory"),
    ("{tmp}", "Bad", "invalid name"),
])
def test_refusals_without_database(capsys, tmp_path, monkeypatch, path, name, message):
    monkeypatch.setenv("DB_PORT", "1")  # proves these never reach the database
    (tmp_path / "file.txt").write_text("x")
    code, out, err = run(capsys, "repo", "add", path.format(tmp=tmp_path), "--name", name)
    assert code == 1 and out == ""
    assert err.startswith("error: ") and message in err


def test_duplicate_name_exits_1(live, capsys, tmp_path):
    n = f"t-{uuid.uuid4().hex[:8]}"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert run(capsys, "repo", "add", str(tmp_path / "a"), "--name", n)[0] == 0
    code, _, err = run(capsys, "repo", "add", str(tmp_path / "b"), "--name", n)
    assert code == 1 and "already registered" in err


def test_empty_registry(capsys, monkeypatch, live):
    monkeypatch.setattr(registry, "list_repos", lambda conn: [])
    code, out, _ = run(capsys, "repo", "list")
    assert code == 0 and out.strip() == "no repositories registered"


def test_format_table():
    rows = [registry.Repo("flask", "/r/flask", 0, 0), registry.Repo("gin", "/r/gin", 12, 3456)]
    assert cli.format_table(rows).splitlines() == [
        "NAME   PATH      FILES  SYMBOLS",
        "flask  /r/flask      0        0",
        "gin    /r/gin       12     3456",
    ]


def test_migrate(live, capsys):
    code, out, _ = run(capsys, "db", "migrate")
    assert code == 0
    assert out.startswith(("applied ", "schema up to date (version "))


@pytest.mark.parametrize("argv", [["repo", "list"], ["db", "migrate"], ["repo", "add", ".", "--name", "t-x"]])
def test_unreachable_exits_2_fast_without_secret(capsys, monkeypatch, argv):
    monkeypatch.setenv("DB_PORT", "1")
    monkeypatch.setenv("DB_PASSWORD", FAKE_PW)
    started = time.monotonic()
    code, out, err = run(capsys, *argv)
    assert time.monotonic() - started < 5
    assert code == 2 and err.startswith("error: ")
    assert FAKE_PW not in out + err


def test_missing_setting_exits_1(capsys, monkeypatch):
    monkeypatch.setenv("DB_PASSWORD", "")
    code, _, err = run(capsys, "repo", "list")
    assert code == 1 and "DB_PASSWORD" in err
