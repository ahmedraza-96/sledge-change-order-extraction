"""LLM extraction step (Gemini, structured outputs).

Mirrors :class:`~co_extract.extract.ClaudeExtractor`: it sends the same
``SYSTEM_PROMPT`` and the same user-turn text (both from ``extract.py``), returns
the same ``LLMResult``, and raises the same ``ExtractionError`` on every failure
path. ``pipeline.run`` cannot tell the two apart.

Gemini's ``response_schema`` accepts the ``RawExtraction`` pydantic model
directly; the SDK inlines the ``$ref``s pydantic emits for the 21 repeated
``ExtractedField`` properties and turns ``str | None`` into ``nullable``.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from google import genai
from google.genai import errors, types

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

# gemini-3-flash-preview is equally verified; both handle the full schema and PDF
# input. Pro models need billing enabled (free tier reports quota `limit: 0`).
GEMINI_MODEL = "gemini-3.5-flash"
MAX_OUTPUT_TOKENS = 16000

# The free tier enforces two separate ceilings: ~5 requests/minute and ~20/day, and
# it also sheds load with 503s under contention. A batch run hits all three.
#
# Budget the backoff for the per-minute window (~60s total) and no further. The daily
# quota resets in *hours* — the 429 for it says "Please retry in 15h14m" — so retrying
# it is pure dead wall-clock. Better to fail the document fast and surface the message,
# which names which ceiling was hit.
RETRY = types.HttpRetryOptions(
    attempts=3,
    initial_delay=10.0,
    max_delay=30.0,
    exp_base=2.0,
    jitter=0.3,
    http_status_codes=[429, 500, 502, 503, 504],
)

# A finish_reason meaning the model declined or bailed, rather than completed.
_DECLINED = frozenset({
    types.FinishReason.SAFETY,
    types.FinishReason.RECITATION,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.SPII,
    types.FinishReason.LANGUAGE,
    types.FinishReason.OTHER,
})


def build_parts(doc: Document, send_pdf: bool,
                repair: tuple[RawExtraction, list[Issue]] | None) -> list[types.Part]:
    """Wrap the shared prompt text in Gemini parts (cf. ``extract.build_content``)."""
    parts: list[types.Part] = []
    if wants_pdf(doc, send_pdf):
        parts.append(types.Part.from_bytes(data=doc.pdf_bytes, mime_type="application/pdf"))
    parts.append(types.Part.from_text(text=document_prompt(doc)))
    if repair:
        parts.append(types.Part.from_text(text=repair_prompt(*repair)))
    return parts


class GeminiExtractor:
    def __init__(self, client: genai.Client | None = None, model: str = GEMINI_MODEL,
                 thinking_level: str | None = None, send_pdf: bool = True):
        # Client() reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
        self.client = client or genai.Client(http_options=types.HttpOptions(retry_options=RETRY))
        self.model = model
        self.thinking_level = thinking_level
        self.send_pdf = send_pdf

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RawExtraction,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=(types.ThinkingConfig(thinking_level=self.thinking_level)
                             if self.thinking_level else None),
        )

    def extract(self, doc: Document, repair: tuple[RawExtraction, list[Issue]] | None = None,
                progress: Callable[..., None] | None = None) -> LLMResult:
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=build_parts(doc, self.send_pdf, repair))],
                config=self._config(),
            )
        # ClientError/ServerError subclass APIError, so they must be caught first.
        except errors.ClientError as e:
            if e.code == 429:
                # Surfaced verbatim: on the free tier this carries the quota that ran out.
                raise ExtractionError(f"rate limited / quota exhausted: {e.message}") from e
            raise ExtractionError(f"API error {e.code}: {e.message}") from e
        except errors.ServerError as e:
            raise ExtractionError(f"server error {e.code}: {e.message}") from e
        except errors.APIError as e:
            raise ExtractionError(f"API error: {e}") from e
        except httpx.HTTPError as e:
            raise ExtractionError(f"connection error: {e}") from e

        feedback = getattr(resp, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise ExtractionError(f"prompt blocked: {feedback.block_reason}")
        if not resp.candidates:
            raise ExtractionError("no candidates returned")

        finish = resp.candidates[0].finish_reason
        if finish == types.FinishReason.MAX_TOKENS:
            raise ExtractionError("output truncated at max_output_tokens")
        if finish in _DECLINED:
            raise ExtractionError(f"model declined: {finish}")

        parsed = resp.parsed
        if parsed is None:
            raise ExtractionError("no structured output returned")
        if not isinstance(parsed, RawExtraction):  # older SDKs hand back a dict
            parsed = RawExtraction.model_validate(parsed)

        usage = resp.usage_metadata
        return LLMResult(
            extraction=parsed,
            model=resp.model_version or self.model,
            provider="gemini",
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )
