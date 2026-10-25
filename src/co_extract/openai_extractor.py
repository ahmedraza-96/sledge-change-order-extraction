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
from .openrouter import page_images
from .schema import Issue, RawExtraction

OPENAI_MODEL = "gpt-5-mini"
# GPT-6 Luna reads text and images but not PDFs, so it gets page images instead. Its reasoning
# defaults to "medium" and reasoning tokens bill as output; "low" keeps a document near $0.003.
LUNA_MODEL = "gpt-6-luna"
LUNA_EFFORT = "low"
MAX_OUTPUT_TOKENS = 16000


def build_input(doc: Document, send_pdf: bool,
                repair: tuple[RawExtraction, list[Issue]] | None, images: bool = False) -> list[dict]:
    """Wrap the shared prompt text in a Responses API user message (cf. ``extract.build_content``).

    ``images`` sends the PDF's pages as PNGs instead of the PDF, for models without file input."""
    content: list[dict] = []
    if images and wants_pdf(doc, send_pdf):
        content += [{"type": "input_image", "image_url": url} for url in page_images(doc)]
    elif wants_pdf(doc, send_pdf):
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
    def __init__(self, client: openai.OpenAI | None = None, model: str = OPENAI_MODEL, send_pdf: bool = True,
                 send_images: bool = False, reasoning_effort: str | None = None, provider: str = "openai"):
        # OpenAI() reads OPENAI_API_KEY from the environment.
        self.client = client or openai.OpenAI()
        self.model = model
        self.send_pdf = send_pdf
        self.send_images = send_images
        self.reasoning_effort = reasoning_effort
        self.provider = provider  # "luna" keeps its eval cache apart from gpt-5-mini's

    def extract(self, doc: Document, repair: tuple[RawExtraction, list[Issue]] | None = None,
                progress: Callable[..., None] | None = None) -> LLMResult:
        try:
            resp = self.client.responses.parse(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=build_input(doc, self.send_pdf, repair, images=self.send_images),
                text_format=RawExtraction,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                **({"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else {}),
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
            provider=self.provider,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        )
