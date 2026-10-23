"""Build an ``Extractor`` by provider name — shared by the CLI and the web app."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

from .extract import MODEL, ClaudeExtractor, Extractor, FixtureExtractor
from .gemini import GEMINI_MODEL, GeminiExtractor
from .openai_extractor import OPENAI_MODEL, OpenAIExtractor
from .openrouter import OPENROUTER_MODEL, OpenRouterExtractor


class Provider(StrEnum):
    claude = "claude"
    gemini = "gemini"
    openai = "openai"
    openrouter = "openrouter"


DEFAULT_MODELS = {Provider.claude: MODEL, Provider.gemini: GEMINI_MODEL, Provider.openai: OPENAI_MODEL,
                  Provider.openrouter: OPENROUTER_MODEL}
API_KEY_VARS = {Provider.claude: ("ANTHROPIC_API_KEY",), Provider.gemini: ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                Provider.openai: ("OPENAI_API_KEY",), Provider.openrouter: ("OPENROUTER_API_KEY",)}


def make_extractor(provider: Provider, model: str | None = None, fixtures: Path | None = None,
                   text_only: bool = False) -> Extractor:
    if fixtures:
        return FixtureExtractor(fixtures)
    load_dotenv()
    model = model or DEFAULT_MODELS[provider]
    if provider is Provider.gemini:
        return GeminiExtractor(model=model, send_pdf=not text_only)
    if provider is Provider.openai:
        return OpenAIExtractor(model=model, send_pdf=not text_only)
    if provider is Provider.openrouter:
        return OpenRouterExtractor(model=model, send_pdf=not text_only)
    return ClaudeExtractor(model=model, send_pdf=not text_only)


def has_key(provider: Provider) -> bool:
    load_dotenv()
    return any(os.environ.get(v) for v in API_KEY_VARS[provider])
