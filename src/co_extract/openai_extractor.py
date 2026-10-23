"""LLM extraction step (OpenAI Responses API, structured outputs).

Mirrors :class:`~co_extract.extract.ClaudeExtractor` and
:class:`~co_extract.gemini.GeminiExtractor`: same ``SYSTEM_PROMPT``, same user-turn
text, same ``LLMResult`` and ``ExtractionError``. ``pipeline.run`` cannot tell them apart.

Every ``RawExtraction`` field is required, so the schema passes OpenAI's strict mode
as-is and ``responses.parse`` hands back a validated model.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path

import openai

from .extract import (
    SYSTEM_PROMPT,
    ExtractionError,
    LLMResult,
    document_prompt,
    repair_prompt,
    wants_pdf,
)
from .ingest import Document
from .schema import Issue, RawExtraction

OPENAI_MODEL = "gpt-5-mini"
MAX_OUTPUT_TOKENS = 16000


def build_input(doc: Document, send_pdf: bool,
                repair: tuple[RawExtraction, list[Issue]] | None) -> list[dict]:
    """Wrap the shared prompt text in a Responses API user message (cf. ``extract.build_content``)."""
    content: list[dict] = []
    if wants_pdf(doc, send_pdf):
        content.append({
            "type": "input_file",
            "filename": Path(doc.source).name,
            "file_data": "data:application/pdf;base64," + base64.b64encode(doc.pdf_bytes).decode(),
        })
    content.append({"type": "input_text", "text": document_prompt(doc)})
    if repair:
        content.append({"type": "input_text", "text": repair_prompt(*repair)})
    return [{"role": "user", "content": content}]


class OpenAIExtractor:
    def __init__(self, client: openai.OpenAI | None = None, model: str = OPENAI_MODEL, send_pdf: bool = True):
        # OpenAI() reads OPENAI_API_KEY from the environment.
        self.client = client or openai.OpenAI()
        self.model = model
        self.send_pdf = send_pdf

    def extract(self, doc: Document, repair: tuple[RawExtraction, list[Issue]] | None = None,
                progress: Callable[..., None] | None = None) -> LLMResult:
        try:
            resp = self.client.responses.parse(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=build_input(doc, self.send_pdf, repair),
                text_format=RawExtraction,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        except openai.RateLimitError as e:
            # Surfaced verbatim: "insufficient_quota" means the account has no credit.
            raise ExtractionError(f"rate limited / quota exhausted: {e.message}") from e
        except openai.APIStatusError as e:
            raise ExtractionError(f"API error {e.status_code}: {e.message}") from e
        except openai.APIConnectionError as e:
            raise ExtractionError(f"connection error: {e}") from e

        if resp.status == "incomplete":
            reason = resp.incomplete_details.reason if resp.incomplete_details else "unknown"
            raise ExtractionError(f"output incomplete: {reason}")
        for item in resp.output:
            for part in getattr(item, "content", None) or []:
                if part.type == "refusal":
                    raise ExtractionError(f"model refused: {part.refusal}")
        if resp.output_parsed is None:
            raise ExtractionError("no structured output returned")

        usage = resp.usage
        return LLMResult(
            extraction=resp.output_parsed,
            model=resp.model or self.model,
            provider="openai",
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        )
