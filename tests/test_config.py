import pytest

from golestan_code_mcp.config import PROJECT_ROOT, ConfigError, load, max_file_bytes, repos_dir, vendor_dirs

SECRET = "s3cr3t-Pa55"
BASE = {"DB_USER": "ccctx", "DB_PASSWORD": SECRET}


def test_defaults_applied():
    cfg = load(BASE, dotenv_path=None)
    assert (cfg.db_host, cfg.db_port, cfg.db_service) == ("localhost", 1521, "FREEPDB1")
    assert cfg.dsn == "localhost:1521/FREEPDB1"
    assert repos_dir({}) == (PROJECT_ROOT / "../ccctx-repos").resolve()


def test_all_missing_and_blank_named_together():
    with pytest.raises(ConfigError) as e:
        load({"DB_PASSWORD": "   "}, dotenv_path=None)
    assert "DB_USER" in str(e.value) and "DB_PASSWORD" in str(e.value)


@pytest.mark.parametrize("port", ["abc", "0", "70000"])
def test_bad_port_named(port):
    with pytest.raises(ConfigError, match="DB_PORT"):
        load({**BASE, "DB_PORT": port}, dotenv_path=None)


def test_password_never_shown():
    cfg = load(BASE, dotenv_path=None)
    assert SECRET not in repr(cfg) and SECRET not in str(cfg)
    with pytest.raises(ConfigError) as e:
        load({**BASE, "DB_PORT": "x"}, dotenv_path=None)
    assert SECRET not in str(e.value)


def test_absolute_repos_dir_kept(tmp_path):
    assert repos_dir({"CCCTX_REPOS_DIR": str(tmp_path / "my repos")}) == (tmp_path / "my repos").resolve()


def test_environment_wins_over_dotenv(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("DB_USER=from_file\nDB_PASSWORD=file_pw\n")
    monkeypatch.setenv("DB_USER", "from_env")
    cfg = load(dotenv_path=dotenv)
    assert cfg.db_user == "from_env"
    assert cfg.db_password  # filled from the file or the environment


def test_ingest_limits_defaults_and_overrides():
    assert max_file_bytes({}) == 1048576
    assert max_file_bytes({"CCCTX_MAX_FILE_BYTES": "2048"}) == 2048
    assert vendor_dirs({}) == {"vendor", "node_modules", "third_party"}
    assert vendor_dirs({"CCCTX_VENDOR_DIRS": " deps , ,lib "}) == {"deps", "lib"}
    assert vendor_dirs({"CCCTX_VENDOR_DIRS": ""}) == set()


@pytest.mark.parametrize("bad", ["0", "-5", "1MB"])
def test_bad_max_file_bytes_named(bad):
    with pytest.raises(ConfigError, match="CCCTX_MAX_FILE_BYTES"):
        max_file_bytes({"CCCTX_MAX_FILE_BYTES": bad})
