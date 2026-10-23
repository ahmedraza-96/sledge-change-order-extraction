"""LLM extraction step (OpenRouter, free models, structured outputs).

OpenRouter speaks the OpenAI Chat Completions API, so this reuses the ``openai`` SDK
pointed at ``https://openrouter.ai/api/v1``. Like the other backends it sends the
shared ``SYSTEM_PROMPT`` / ``document_prompt()`` / ``repair_prompt()`` text, returns
``LLMResult`` and raises ``ExtractionError``; ``pipeline.run`` cannot tell them apart.

Two OpenRouter-specific choices:

- **Fallbacks.** Free models are rate-limited and sometimes overloaded, so the request
  names a ``models`` list; OpenRouter moves to the next one when a model errors.
  ``LLMResult.model`` records which one actually answered.
- **Pages as images, not a PDF.** OpenRouter's PDF input may route through a paid OCR
  engine for models without native PDF support. Rendering pages to PNG here keeps every
  call free and works with any vision model.
"""

from __future__ import annotations

import base64
import os
import re
import time
from collections.abc import Callable

import openai
import pymupdf
from pydantic import ValidationError

from .extract import (
    SYSTEM_PROMPT,
    ExtractionError,
    LLMResult,
    document_prompt,
    repair_prompt,
    wants_pdf,
)
from .ingest import Document
from .schema import FIELD_SPECS, Issue, RawExtraction

BASE_URL = "https://openrouter.ai/api/v1"
# Chosen from the free models that accept images *and* enforce a JSON schema.
# Nemotron 3.5 Super VL is the one with published document/OCR results; the others
# take over when it is rate-limited or down. The last one is text-only.
OPENROUTER_MODEL = "nvidia/nemotron-3.5-super-vl:free"
OPENROUTER_FALLBACKS = ("dots-studio/dots-3-note-preview:free", "nvidia/nemotron-3-super-120b-a12b:free")
MAX_OUTPUT_TOKENS = 16000
PAGE_DPI = 150
MAX_PAGE_IMAGES = 10  # change orders are short; this bounds the request size


def page_images(doc: Document) -> list[str]:
    """Render the document's PDF pages as PNG data URIs."""
    out = []
    with pymupdf.open(stream=doc.pdf_bytes, filetype="pdf") as pdf:
        for page in list(pdf)[:MAX_PAGE_IMAGES]:
            png = page.get_pixmap(dpi=PAGE_DPI).tobytes("png")
            out.append("data:image/png;base64," + base64.b64encode(png).decode())
    return out


def build_messages(doc: Document, send_pdf: bool,
                   repair: tuple[RawExtraction, list[Issue]] | None) -> list[dict]:
    """Wrap the shared prompt text in Chat Completions messages (cf. ``extract.build_content``)."""
    content: list[dict] = []
    if wants_pdf(doc, send_pdf):
        content += [{"type": "image_url", "image_url": {"url": url}} for url in page_images(doc)]
    content.append({"type": "text", "text": document_prompt(doc)})
    if repair:
        content.append({"type": "text", "text": repair_prompt(*repair)})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


class OpenRouterExtractor:
    def __init__(self, client: openai.OpenAI | None = None, model: str = OPENROUTER_MODEL,
                 fallbacks: tuple[str, ...] = OPENROUTER_FALLBACKS, send_pdf: bool = True):
        # Reads OPENROUTER_API_KEY. A couple of SDK retries ride out a brief 429/503.
        self.client = client or openai.OpenAI(
            base_url=BASE_URL, api_key=_api_key(), max_retries=2,
            default_headers={"X-Title": "co-extract"},
        )
        self.model = model
        self.fallbacks = [m for m in fallbacks if m != model]
        self.send_pdf = send_pdf

    def extract(self, doc: Document, repair: tuple[RawExtraction, list[Issue]] | None = None,
                progress: Callable[..., None] | None = None) -> LLMResult:
        # Streamed so a caller can show real progress: the model's reasoning arrives first,
        # then the JSON, whose field keys we count as they appear.
        tracker = _Progress(progress)
        try:
            with self.client.chat.completions.stream(
                model=self.model,
                messages=build_messages(doc, self.send_pdf, repair),
                response_format=RawExtraction,
                max_tokens=MAX_OUTPUT_TOKENS,
                stream_options={"include_usage": True},
                extra_body={
                    "models": [self.model, *self.fallbacks],
                    # Only route to providers that honour the JSON schema, never one that ignores it.
                    "provider": {"require_parameters": True},
                },
            ) as stream:
                for event in stream:
                    if event.type == "chunk" and event.chunk.choices:
                        tracker.thinking(getattr(event.chunk.choices[0].delta, "reasoning", None))
                    elif event.type == "content.delta":
                        tracker.writing(event.snapshot)
                resp = stream.get_final_completion()
        except openai.RateLimitError as e:
            raise ExtractionError(f"rate limited (free models: 20/min, 50/day without credits): {e.message}") from e
        except openai.APIStatusError as e:
            hint = ""
            if e.status_code == 404:
                # Free endpoints are often run by providers that log prompts; a strict privacy
                # setting filters them all out and OpenRouter answers 404.
                hint = " (if no endpoint matched, allow free-model providers at https://openrouter.ai/settings/privacy)"
            raise ExtractionError(f"API error {e.status_code}: {e.message}{hint}") from e
        except openai.APIConnectionError as e:
            raise ExtractionError(f"connection error: {e}") from e
        except openai.APIError as e:
            # An error OpenRouter sends inside an already-open stream (e.g. the upstream model failed).
            raise ExtractionError(f"API error during the response: {e.message}") from e
        # The SDK raises these while parsing, before the response reaches us.
        except openai.LengthFinishReasonError as e:
            raise ExtractionError("output truncated at max_tokens") from e
        except openai.ContentFilterFinishReasonError as e:
            raise ExtractionError("model declined: content filter") from e
        except (ValidationError, ValueError) as e:
            # A model that drifts from the schema; the SDK fails while parsing its JSON.
            raise ExtractionError(f"model returned output that does not match the schema: {e}") from e

        if not resp.choices:
            raise ExtractionError("no choices returned")
        choice = resp.choices[0]
        if choice.message.refusal:
            raise ExtractionError(f"model refused: {choice.message.refusal}")
        if choice.message.parsed is None:
            raise ExtractionError("no structured output returned")

        usage = resp.usage
        return LLMResult(
            extraction=choice.message.parsed,
            model=resp.model or self.model,
            provider="openrouter",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


_FIELD_KEYS = (*FIELD_SPECS, "line_items")
_KEY_RE = re.compile(r'"(' + "|".join(_FIELD_KEYS) + r')"\s*:')


class _Progress:
    """Turns stream events into throttled ``progress(...)`` calls.

    phase="thinking" carries ``words`` (reasoning so far); phase="writing" carries ``found``
    and ``total``: how many of the output's field keys have appeared in the JSON so far.
    """

    INTERVAL = 0.4  # seconds between thinking updates; field counts report on every change

    def __init__(self, callback: Callable[..., None] | None):
        self.callback, self.words, self.found, self.last = callback, 0, 0, 0.0

    def thinking(self, text: str | None) -> None:
        if not self.callback or not text:
            return
        self.words += len(text.split())
        now = time.monotonic()
        if now - self.last >= self.INTERVAL:
            self.last = now
            self.callback(phase="thinking", words=self.words)

    def writing(self, snapshot: str) -> None:
        if not self.callback:
            return
        found = len(set(_KEY_RE.findall(snapshot)))
        if found != self.found:
            self.found = found
            self.callback(phase="writing", found=found, total=len(_FIELD_KEYS))


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ExtractionError("OPENROUTER_API_KEY is not set (add it to .env)")
    return key
