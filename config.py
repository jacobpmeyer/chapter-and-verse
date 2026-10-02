"""Settings loaded from .env (see .env.example)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    library_path: Path = Path(os.getenv("LIBRARY_PATH", "~/Calibre Library")).expanduser()
    database_url: str = os.getenv("DATABASE_URL", "")

    # Chunking (estimated tokens)
    chunk_min_tokens: int = _int("CHUNK_MIN_TOKENS", 500)
    chunk_max_tokens: int = _int("CHUNK_MAX_TOKENS", 800)
    chunk_overlap_tokens: int = _int("CHUNK_OVERLAP_TOKENS", 80)
    chunk_overlap_cap: int = _int("CHUNK_OVERLAP_CAP", 150)
    chunk_min_tail_tokens: int = _int("CHUNK_MIN_TAIL_TOKENS", 150)

    # Models
    summary_model: str = os.getenv("SUMMARY_MODEL", "claude-sonnet-5-5")
    summary_effort: str = os.getenv("SUMMARY_EFFORT", "medium")
    summary_context_limit: int = _int("SUMMARY_CONTEXT_LIMIT", 800_000)
    agent_model: str = os.getenv("AGENT_MODEL", "claude-opus-5-5")
    agent_effort: str = os.getenv("AGENT_EFFORT", "high")
    agent_max_turns: int = _int("AGENT_MAX_TURNS", 12)
    # summarized = show a summary of the agent's reasoning (dimmed) before it acts
    # updates    = only short progress notes between tool calls (beta; in testing,
    #              Opus 5.5 rarely wrote any, so the terminal stayed silent)
    # omitted    = show nothing
    agent_thinking_display: str = os.getenv("AGENT_THINKING_DISPLAY", "summarized")

    # Embeddings
    embed_provider: str = os.getenv("EMBED_PROVIDER", "voyage")
    embed_model: str = os.getenv("EMBED_MODEL", "voyage-4-large")
    embed_dim: int = _int("EMBED_DIM", 1024)
    embed_batch_size: int = _int("EMBED_BATCH_SIZE", 64)
    # Cap per request, in *locally estimated* tokens. The real per-request limit
    # is 120K for voyage-4-large (320K voyage-4); the local estimate can
    # undercount by ~30%, so stay well below.
    embed_batch_max_tokens: int = _int("EMBED_BATCH_MAX_TOKENS", 60_000)

    # Agent tool limits
    full_book_token_limit: int = _int("FULL_BOOK_TOKEN_LIMIT", 150_000)
    # Cosine similarity below which search results are flagged as weak. Measured on
    # voyage-4-large: correct hits 0.26-0.66 depending on wording, nonsense ~0.1.
    weak_match_threshold: float = _float("WEAK_MATCH_THRESHOLD", 0.2)

    # HTTP API
    api_keys: tuple = tuple(k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip())
    auth_disabled: bool = os.getenv("AUTH_DISABLED", "") == "1"  # local development only
    cors_origins: tuple = tuple(o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip())
    daily_budget_usd: float = _float("DAILY_BUDGET_USD", 5.0)
    db_pool_max: int = _int("DB_POOL_MAX", 10)  # per API process; Cloud SQL's smallest tier allows ~25 in total
    # thread    = a background thread in the API process (local use)
    # cloud_run = a Cloud Run Job execution per indexing job; needs CLOUD_RUN_JOB
    job_runner: str = os.getenv("JOB_RUNNER", "thread")
    cloud_run_job: str = os.getenv("CLOUD_RUN_JOB", "")  # projects/<project>/locations/<region>/jobs/<job>
    # Cloudflare Access: when both are set, every request except /health must carry a
    # valid Access JWT (in addition to an API key). Unset locally.
    cf_access_team_domain: str = os.getenv("CF_ACCESS_TEAM_DOMAIN", "")  # <team>.cloudflareaccess.com
    cf_access_aud: str = os.getenv("CF_ACCESS_AUD", "")  # the Access application's AUD tag


settings = Settings()

# USD per million tokens (input, output). Thinking tokens bill as output.
# Prompt caching: cache writes bill at 1.25x input, cache reads at 0.05x input.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.05

# Embedding prices, USD per million tokens. Voyage's voyage-4 models also
# include 200M free tokens per account.
EMBED_PRICES: dict[str, float] = {
    "voyage-4-large": 0.12,
    "voyage-4": 0.06,
    "voyage-4-lite": 0.02,
    "voyage-3.5": 0.06,
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
