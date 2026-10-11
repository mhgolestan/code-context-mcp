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


MODEL_URL = (
    "https://adwc4pm.objectstorage.us-ashburn-1.oci.customer-oci.com/p/"
    "iPX9W0MZeRkwJKWdFmdJCemmN-iKAl_bFvNGYLW7YqIrw4kKsukL24J2q93Beb9S/n/adwc4pm/b/OML-ai-models/o/"
    "all_MiniLM_L12_v2.onnx"
)


def model_url(env: Mapping[str, str]) -> str:
    return (env.get("CCCTX_MODEL_URL") or "").strip() or MODEL_URL


def model_cache(env: Mapping[str, str]) -> Path:
    path = Path((env.get("CCCTX_MODEL_CACHE") or "").strip() or ".cache/models")
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def search_candidates(env: Mapping[str, str]) -> int:
    """Rows asked of the hybrid index before repository filtering, the definition boost and the cap."""
    text = (env.get("CCCTX_SEARCH_CANDIDATES") or "").strip() or "100"
    if not text.isdigit() or int(text) < 1:
        raise ConfigError(f"invalid setting: CCCTX_SEARCH_CANDIDATES={text!r} (expected a positive integer)")
    return int(text)


def _weight(env: Mapping[str, str], name: str) -> float:
    text = (env.get(name) or "").strip() or "1"
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not value > 0:
        raise ConfigError(f"invalid setting: {name}={text!r} (expected a positive number)")
    return value


def text_weight(env: Mapping[str, str]) -> float:
    """Keyword side's weight in the fusion."""
    return _weight(env, "CCCTX_TEXT_WEIGHT")


def vector_weight(env: Mapping[str, str]) -> float:
    """Semantic side's weight in the fusion."""
    return _weight(env, "CCCTX_VECTOR_WEIGHT")


def max_file_bytes(env: Mapping[str, str]) -> int:
    text = (env.get("CCCTX_MAX_FILE_BYTES") or "").strip() or "1048576"
    if not text.isdigit() or int(text) < 1:
        raise ConfigError(f"invalid setting: CCCTX_MAX_FILE_BYTES={text!r} (expected a positive integer)")
    return int(text)


def vendor_dirs(env: Mapping[str, str]) -> frozenset[str]:
    text = env.get("CCCTX_VENDOR_DIRS")
    if text is None:
        text = "vendor,node_modules,third_party"
    return frozenset(part.strip() for part in text.split(",") if part.strip())


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
