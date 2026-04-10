from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError, RateLimitError
import os
import json
import time
from typing import Any

app = FastAPI()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


OPENAI_TIMEOUT_SECONDS = _env_float("OPENAI_TIMEOUT_SECONDS", 45.0)
OPENAI_CLIENT_RETRIES = _env_int("OPENAI_CLIENT_RETRIES", 2)
TOOL_RETRY_COUNT = _env_int("TOOL_RETRY_COUNT", 1)

client_kwargs: dict[str, Any] = {
    "api_key": os.environ["OPENAI_API_KEY"],
    "timeout": OPENAI_TIMEOUT_SECONDS,
    "max_retries": OPENAI_CLIENT_RETRIES,
}
if os.environ.get("OPENAI_PROJECT"):
    client_kwargs["project"] = os.environ["OPENAI_PROJECT"]
if os.environ.get("OPENAI_ORGANIZATION"):
    client_kwargs["organization"] = os.environ["OPENAI_ORGANIZATION"]

client = OpenAI(**client_kwargs)
VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4.1")


class Query(BaseModel):
    question: str
    max_results: int = Field(default=8, ge=1, le=100)  # snabbare/stabilare default
    year: int | None = None
    issue: int | None = None
    issue_from: int | None = None
    issue_to: int | None = None
    page: int | None = None
    page_from: int | None = None
    page_to: int | None = None
    include_search_results: bool = False


def build_attribute_filters(query: Query) -> dict[str, Any] | None:
    filters: list[dict[str, Any]] = []

    if query.issue is not None:
        filters.append({"type": "lte", "key": "issue_start", "value": query.issue})
        filters.append({"type": "gte", "key": "issue_end", "value": query.issue})
    else:
        if query.issue_from is not None:
            filters.append(
                {"type": "gte", "key": "issue_end", "value": query.issue_from}
            )
        if query.issue_to is not None:
            filters.append({"type": "lte", "key": "issue_start", "value": query.issue_to})

    if query.year is not None:
        filters.append({"type": "eq", "key": "year", "value": query.year})

    if query.page is not None:
        filters.append({"type": "eq", "key": "page_number", "value": query.page})
    else:
        if query.page_from is not None:
            filters.append({"type": "gte", "key": "page_number", "value": query.page_from})
        if query.page_to is not None:
            filters.append({"type": "lte", "key": "page_number", "value": query.page_to})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"type": "and", "filters": filters}


def _is_retryable_status(exc: APIStatusError) -> bool:
    return getattr(exc, "status_code", None) in {408, 409, 429, 500, 502, 503, 504}


def _create_response_with_retry(
    query: Query,
    tool_config: dict[str, Any],
    response_kwargs: dict[str, Any],
    prompt: str,
):
    attempts = 1 + max(0, TOOL_RETRY_COUNT)
    last_exc: Exception | None = None
    current_max_results = int(tool_config.get("max_num_results", query.max_results))

    for attempt in range(attempts):
        attempt_tool_config = dict(tool_config)
        attempt_tool_config["max_num_results"] = max(4, current_max_results)

        try:
            return client.responses.create(
                model=MODEL_NAME,
                input=prompt,
                tools=[attempt_tool_config],
                **response_kwargs,
            )
        except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
            last_exc = exc
        except APIStatusError as exc:
            if _is_retryable_status(exc):
                last_exc = exc
            else:
                raise

        if attempt < attempts - 1:
            current_max_results = max(4, current_max_results // 2)
            time.sleep(0.8 * (attempt + 1))

    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown tool call error")


def run_archive_search(query: Query) -> dict:
    prompt = f"""
Du är en arkivassistent för Kyrkans Tidning.

Använd endast material som hittas via file_search i vector store.

Uppgift:
1. Besvara användarens fråga kort, sakligt och tydligt på svenska.
2. För varje relevant källa, extrahera om möjligt:
   - issue: tidningens nummer/utgåva
   - year: årtal
   - page: sida
   - excerpt: ett kort relevant utdrag eller sammanfattning
3. Uppgifterna issue, year och page får bara anges om de faktiskt framgår av texten.
4. Nämn inte interna filnamn, tekniska id:n eller vector store-information.
5. Om underlaget är osäkert eller ofullständigt, säg det tydligt.
6. Returnera svaret som JSON med exakt denna struktur:

{{
  "answer": "kort svar till användaren",
  "sources": [
    {{
      "issue": "string eller null",
      "year": "string eller null",
      "page": "string eller null",
      "excerpt": "string"
    }}
  ]
}}

Användarens fråga:
{query.question}
""".strip()

    tool_config: dict[str, Any] = {
        "type": "file_search",
        "vector_store_ids": [VECTOR_STORE_ID],
        "max_num_results": query.max_results,
    }

    attribute_filters = build_attribute_filters(query)
    if attribute_filters:
        tool_config["filters"] = attribute_filters

    response_kwargs: dict[str, Any] = {}
    if query.include_search_results:
        response_kwargs["include"] = ["file_search_call.results"]

    try:
        response = _create_response_with_retry(query, tool_config, response_kwargs, prompt)
    except Exception:
        return {
            "answer": "Arkivverktyget svarade inte just nu. Försök igen om en stund.",
            "sources": [],
            "applied_filters": attribute_filters,
            "search_results_count": None,
            "tool_error": True,
        }

    raw_text = response.output_text
    search_results_count = 0

    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "file_search_call":
            result_items = getattr(item, "results", None) or getattr(
                item, "search_results", None
            )
            if result_items:
                search_results_count += len(result_items)

    try:
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            raise ValueError("Model response was not a JSON object")

        answer = data.get("answer", "")
        sources = data.get("sources", [])

        if not isinstance(answer, str):
            answer = str(answer)

        if not isinstance(sources, list):
            sources = []

        cleaned_sources = []
        for src in sources[:8]:
            if not isinstance(src, dict):
                continue
            cleaned_sources.append({
                "issue": src.get("issue"),
                "year": src.get("year"),
                "page": src.get("page"),
                "excerpt": src.get("excerpt", "")
            })

        return {
            "answer": answer,
            "sources": cleaned_sources,
            "applied_filters": attribute_filters,
            "search_results_count": (
                search_results_count if query.include_search_results else None
            ),
        }

    except Exception:
        return {
            "answer": raw_text,
            "sources": [],
            "applied_filters": attribute_filters,
            "search_results_count": (
                search_results_count if query.include_search_results else None
            ),
        }


@app.get("/health")
def health():
    return {
        "ok": True,
        "vector_store": VECTOR_STORE_ID,
        "project": os.environ.get("OPENAI_PROJECT"),
        "organization": os.environ.get("OPENAI_ORGANIZATION"),
        "model": MODEL_NAME,
        "openai_timeout_seconds": OPENAI_TIMEOUT_SECONDS,
        "openai_client_retries": OPENAI_CLIENT_RETRIES,
        "tool_retry_count": TOOL_RETRY_COUNT,
    }


@app.post("/search")
def search(query: Query):
    return run_archive_search(query)


@app.post("/api/chat")
def api_chat(query: Query):
    return run_archive_search(query)


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html>
<html lang="sv">
<head>
  <meta charset="utf-8">
  <title>KT Arkivchat API</title>
</head>
<body>
  <h1>KT Arkivchat API</h1>
  <p>API är igång.</p>
</body>
</html>
"""
