import os
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.globals import get_llm_cache, set_llm_cache


def configure_llm_cache() -> bool:
    enabled = os.getenv("LLM_CACHE_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        set_llm_cache(None)
        return False
    if get_llm_cache() is None:
        cache_path = Path(os.getenv("LLM_CACHE_PATH", "./data/llm_cache.db"))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(cache_path)))
    return True