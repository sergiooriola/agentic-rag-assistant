"""Centralized project configuration.

All configuration lives in a single ``Settings`` object (pydantic-settings)
that reads environment variables and, locally, a ``.env`` file. The rest of
the codebase never touches ``os.environ`` directly: it imports or receives
``Settings``, which keeps configuration typed, validated, and easy to
override in tests.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration parameters, overridable via environment variables.

    Example: setting ``CHUNK_SIZE=800`` in the environment (or in ``.env``)
    overrides ``chunk_size``. The field-name -> ENV_VAR mapping is automatic
    and case-insensitive.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    # Empty by default on purpose: tests and tooling that never call the LLM
    # should not require a key. Components that do need it validate at the
    # point of use.
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Paths ---
    data_dir: Path = Path("data")
    chroma_dir: Path = Path(".chroma")
    collection_name: str = "documents"
    checkpoint_db: Path = Path("checkpoints.sqlite")

    # --- Chunking (phase 2) ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Agent (phase 4) ---
    # The intent classifier needs to know what the corpus covers to decide
    # whether a question can be answered from it. Update when data/ changes.
    corpus_description: str = (
        "technical documentation covering HTTP fundamentals, Git branching, "
        "and REST API design guidelines"
    )

    # --- Hybrid retrieval (phase 3) ---
    top_k_semantic: int = 8   # candidates from vector search
    top_k_lexical: int = 8    # candidates from BM25
    top_k_final: int = 4      # chunks that reach the prompt after RRF fusion
    rrf_k: int = 60           # reciprocal rank fusion smoothing constant


@lru_cache
def get_settings() -> Settings:
    """Return the configuration as a singleton.

    ``lru_cache`` guarantees a single read of the environment per process
    and lets tests inject/reset configuration via ``cache_clear()``.
    """
    return Settings()
