"""Real API calls. Run with: uv run pytest -m live"""

from pathlib import Path

import pytest
from dotenv import load_dotenv

from co_extract import pipeline
from co_extract.evaluate import field_correct, load_truths
from co_extract.extract import ClaudeExtractor

ROOT = Path(__file__).parent.parent
pytestmark = pytest.mark.live


@pytest.mark.parametrize("stem,suffix", [("01_aia_clean", ".pdf"), ("03_credit", ".pdf")])
def test_live_extraction(stem, suffix):
    load_dotenv(ROOT / ".env")
    truth = load_truths(ROOT / "data" / "truth")[stem]
    r = pipeline.run(ROOT / "data" / "samples" / f"{stem}{suffix}", ClaudeExtractor())
    for name in ("change_order_number", "total_change_amount", "revised_contract_sum"):
        assert field_correct(name, r["change_order"][name], truth[name]), name
