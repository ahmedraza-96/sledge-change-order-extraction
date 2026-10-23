"""Real API calls. Run with: uv run pytest -m live

Each provider is skipped unless its key is present, so this exercises whichever
credentials are configured.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from co_extract import pipeline
from co_extract.evaluate import field_correct, load_truths
from co_extract.extract import ClaudeExtractor
from co_extract.gemini import GeminiExtractor
from co_extract.openai_extractor import OpenAIExtractor

ROOT = Path(__file__).parent.parent
pytestmark = pytest.mark.live

PROVIDERS = {
    "claude": ("ANTHROPIC_API_KEY", ClaudeExtractor),
    "gemini": ("GEMINI_API_KEY", GeminiExtractor),
    "openai": ("OPENAI_API_KEY", OpenAIExtractor),
}


@pytest.mark.parametrize("provider", list(PROVIDERS))
@pytest.mark.parametrize("stem,suffix", [("01_aia_clean", ".pdf"), ("03_credit", ".pdf")])
def test_live_extraction(provider, stem, suffix):
    load_dotenv(ROOT / ".env")
    env_var, factory = PROVIDERS[provider]
    if not os.getenv(env_var):
        pytest.skip(f"{env_var} not set")
    truth = load_truths(ROOT / "data" / "truth")[stem]
    r = pipeline.run(ROOT / "data" / "samples" / f"{stem}{suffix}", factory())
    assert r["meta"]["provider"] == provider
    for name in ("change_order_number", "total_change_amount", "revised_contract_sum"):
        assert field_correct(name, r["change_order"][name], truth[name]), name
