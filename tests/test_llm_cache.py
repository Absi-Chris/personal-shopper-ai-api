import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation

from app.llm_cache import configure_llm_cache


class LlmCacheTests(unittest.TestCase):
    def tearDown(self):
        set_llm_cache(None)

    def test_sqlite_cache_persists_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache" / "llm.db"
            with patch.dict(os.environ, {
                "LLM_CACHE_ENABLED": "true",
                "LLM_CACHE_PATH": str(cache_path),
            }):
                self.assertTrue(configure_llm_cache())
                cache = get_llm_cache()
                cache.update("prompt", "model", [Generation(text="cached answer")])
                set_llm_cache(None)

                self.assertTrue(configure_llm_cache())
                cached = get_llm_cache().lookup("prompt", "model")

            self.assertEqual(cached[0].text, "cached answer")

    def test_cache_can_be_disabled(self):
        with patch.dict(os.environ, {"LLM_CACHE_ENABLED": "false"}):
            self.assertFalse(configure_llm_cache())
        self.assertIsNone(get_llm_cache())


if __name__ == "__main__":
    unittest.main()