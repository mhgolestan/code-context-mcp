"""Settings from the environment, with .env at the project root as a fallback."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    db_user: str
    db_password: str = field(repr=False)
    db_host: str
    db_port: int
    db_service: str

    @property
    def dsn(self) -> str:
        return f"{self.db_host}:{self.db_port}/{self.db_service}"


def environment(dotenv_path: Path | None = PROJECT_ROOT / ".env") -> dict[str, str]:
    """The .env file overlaid by the process environment, which wins."""
    file_values = dotenv_values(dotenv_path) if dotenv_path and Path(dotenv_path).is_file() else {}
    return {**{k: v for k, v in file_values.items() if v is not None}, **os.environ}


def repos_dir(env: Mapping[str, str]) -> Path:
    path = Path((env.get("CCCTX_REPOS_DIR") or "").strip() or "../ccctx-repos")
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load(env: Mapping[str, str] | None = None, dotenv_path: Path | None = PROJECT_ROOT / ".env") -> Config:
    """Build a Config. Errors name settings, never values of secrets."""
    if env is None:
        env = environment(dotenv_path)

    def get(name: str, default: str = "") -> str:
        return (env.get(name) or "").strip() or default

    errors = [f"missing setting: {name}" for name in ("DB_USER", "DB_PASSWORD") if not get(name)]
    port_text = get("DB_PORT", "1521")
    port = int(port_text) if port_text.isdigit() else 0
    if not 1 <= port <= 65535:
        errors.append(f"invalid setting: DB_PORT={port_text!r} (expected 1-65535)")
    if errors:
        raise ConfigError("; ".join(errors))

    return Config(
        db_user=get("DB_USER"),
        db_password=get("DB_PASSWORD"),
        db_host=get("DB_HOST", "localhost"),
        db_port=port,
        db_service=get("DB_SERVICE", "FREEPDB1"),
    )
