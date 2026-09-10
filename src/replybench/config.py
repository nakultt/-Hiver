"""Runtime configuration, loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
CACHE_DIR = ROOT / ".cache"

load_dotenv(ROOT / ".env")


def _budget_env_name(model: str) -> str:
    return "BUDGET_" + model.upper().replace("-", "_").replace(".", "_")


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    gen_model: str
    judge_model: str
    data_model: str
    strong_model: str
    concurrency: int
    backend: str          # "api" | "mock"
    cache: bool

    @property
    def live(self) -> bool:
        return self.backend == "api"

    def budget_for(self, model: str) -> int:
        """Daily request cap for a model. 0 means unlimited."""
        return int(os.getenv(_budget_env_name(model), "0") or "0")


def load_settings() -> Settings:
    base = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    if not base.endswith("/"):
        base += "/"
    key = os.getenv("LLM_API_KEY", "").strip()
    backend = os.getenv("LLM_BACKEND", "").strip() or ("api" if key and "paste-your" not in key else "mock")
    return Settings(
        api_key=key,
        base_url=base,
        gen_model=os.getenv("GEN_MODEL", "gemini-2.5-flash-lite"),
        judge_model=os.getenv("JUDGE_MODEL", "gemini-2.5-flash-lite"),
        data_model=os.getenv("DATA_MODEL", "gemini-2.5-flash-lite"),
        strong_model=os.getenv("STRONG_MODEL", "gemini-2.5-flash"),
        concurrency=int(os.getenv("LLM_CONCURRENCY", "8")),
        backend=backend,
        cache=os.getenv("LLM_CACHE", "1") not in ("0", "false", "False"),
    )


SETTINGS = load_settings()
