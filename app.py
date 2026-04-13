from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError, RateLimitError
import os
import json
import time
import re
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


OPENAI_TIMEOUT_SECONDS = _env_float("OPENAI_TIMEOUT_SECONDS", 30.0)
OPENAI_CLIENT_RETRIES = _env_int("OPENAI_CLIENT_RETRIES", 1)
TOOL_RETRY_COUNT = _env_int("TOOL_RETRY_COUNT", 0)
MAX_RESULTS_HARD_CAP = _env_int("MAX_RESULTS_HARD_CAP", 12)
AUTO_BROADEN_SEARCH = _env_bool("AUTO_BROADEN_SEARCH", False)
AUTO_BROADEN_TARGET = _env_int("AUTO_BROADEN_TARGET", 12)
MIN_SOURCES_FOR_SINGLE_PASS = _env_int("MIN_SOURCES_FOR_SINGLE_PASS", 1)
ALWAYS_INCLUDE_RESULTS = _env_bool("ALWAYS_INCLUDE_RESULTS", False)

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
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


class Query(BaseModel):
    question: str
    max_results: int = Field(default=6, ge=1, le=100)  # snabbare/stabilare default
    year: int | None = None
    issue: int | None = None
    issue_from: int | None = None
    issue_to: int | None = None
    page: int | None = None
    page_from: int | None = None
    page_to: int | None = None
    include_search_results: bool = False


def _normalize_meta_field(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if int(value) <= 0:
            return None
        return str(int(value))

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    invalid_values = {
        "0",
        "nr 0",
        "nr. 0",
        "issue 0",
        "okänt",
        "okand",
        "unknown",
        "saknas",
        "n/a",
        "null",
        "none",
    }
    if lowered in invalid_values:
        return None

    return text


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            dumped = obj.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {}


def _parse_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        iv = int(value)
        return iv if iv > 0 else None
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    iv = int(match.group(0))
    return iv if iv > 0 else None


def _issue_from_attrs(attrs: dict[str, Any]) -> str | None:
    issue_text = _normalize_meta_field(attrs.get("issue_number_text"))
    if issue_text and issue_text != "0":
        return issue_text.replace("/", "-")

    start = _parse_positive_int(attrs.get("issue_start"))
    end = _parse_positive_int(attrs.get("issue_end"))
    if start and end:
        return str(start) if start == end else f"{start}-{end}"
    return None


def _extract_text_blob(result_dict: dict[str, Any]) -> str:
    for key in ("text", "content", "snippet", "excerpt"):
        value = result_dict.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    maybe_text = item.get("text") or item.get("content")
                    if isinstance(maybe_text, str):
                        parts.append(maybe_text)
            merged = "\n".join(p for p in parts if p)
            if merged.strip():
                return merged
    return ""


def _issue_from_text(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"Kyrkans Tidning\s+nr\.?\s*([0-9]{1,2}(?:[/-][0-9]{1,2})?)",
        r"\bNr\.?\s*([0-9]{1,2}(?:[/-][0-9]{1,2})?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        token = match.group(1).replace("/", "-").strip()
        if token in {"0", "00"}:
            return None
        return token
    return None


def _page_from_text(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"(?:---\s*)?sida\s+(\d{1,3})\s+av\s+\d{1,4}", text, flags=re.IGNORECASE)
    if not match:
        return None
    page = _parse_positive_int(match.group(1))
    return str(page) if page else None


def _extract_result_meta_candidates(response: Any) -> list[dict[str, str | None]]:
    candidates: list[dict[str, str | None]] = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "file_search_call":
            continue
        result_items = getattr(item, "results", None) or getattr(item, "search_results", None)
        if not result_items:
            continue
        for result in result_items:
            rd = _to_plain_dict(result)
            attrs = rd.get("attributes")
            attrs = attrs if isinstance(attrs, dict) else {}
            text_blob = _extract_text_blob(rd)

            issue = _issue_from_attrs(attrs) or _issue_from_text(text_blob)
            year = _normalize_meta_field(attrs.get("year"))
            page = _normalize_meta_field(attrs.get("page_number")) or _page_from_text(text_blob)

            candidates.append({
                "issue": issue,
                "year": year,
                "page": page,
                "text": text_blob,
            })
    return candidates


def _tokenize(text: str) -> set[str]:
    return {
        w.lower()
        for w in re.findall(r"[A-Za-zÅÄÖåäö0-9]{4,}", text or "")
    }


def _best_candidate_for_excerpt(
    excerpt: str,
    candidates: list[dict[str, str | None]],
) -> dict[str, str | None] | None:
    if not candidates:
        return None

    excerpt_tokens = _tokenize(excerpt)
    best: dict[str, str | None] | None = None
    best_score = 0

    for cand in candidates:
        cand_tokens = _tokenize(str(cand.get("text") or ""))
        if not excerpt_tokens or not cand_tokens:
            score = 0
        else:
            score = len(excerpt_tokens.intersection(cand_tokens))
        if score > best_score:
            best = cand
            best_score = score

    if best and best_score >= 2:
        return best

    complete = [c for c in candidates if c.get("issue") and c.get("page")]
    if len(complete) == 1:
        return complete[0]
    return None


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


def _narrow_filters_present(query: Query) -> bool:
    return any(
        value is not None
        for value in (
            query.issue,
            query.issue_from,
            query.issue_to,
            query.page,
            query.page_from,
            query.page_to,
        )
    )


def _source_metadata_score(sources: list[dict[str, Any]]) -> int:
    score = 0
    for src in sources:
        if src.get("issue"):
            score += 1
        if src.get("year"):
            score += 1
        if src.get("page"):
            score += 1
    return score


def _payload_score(payload: dict[str, Any]) -> tuple[int, int, int]:
    sources = payload.get("sources")
    if not isinstance(sources, list):
        sources = []
    answer = payload.get("answer", "")
    return (
        len(sources),
        _source_metadata_score(sources),  # prefer richer source metadata
        len(str(answer)),
    )


def _build_payload_from_response(
    response: Any,
    query: Query,
    attribute_filters: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_text = response.output_text
    search_results_count = 0
    result_meta_candidates = _extract_result_meta_candidates(response)

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
            cleaned = {
                "issue": _normalize_meta_field(src.get("issue")),
                "year": _normalize_meta_field(src.get("year")),
                "page": _normalize_meta_field(src.get("page")),
                "excerpt": str(src.get("excerpt", "")).strip()
            }

            if not (cleaned["issue"] and cleaned["year"] and cleaned["page"]):
                best = _best_candidate_for_excerpt(cleaned["excerpt"], result_meta_candidates)
                if best:
                    if not cleaned["issue"]:
                        cleaned["issue"] = _normalize_meta_field(best.get("issue"))
                    if not cleaned["year"]:
                        cleaned["year"] = _normalize_meta_field(best.get("year"))
                    if not cleaned["page"]:
                        cleaned["page"] = _normalize_meta_field(best.get("page"))

            if not cleaned["year"] and query.year is not None:
                cleaned["year"] = str(query.year)

            cleaned_sources.append(cleaned)

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
   Om värdet saknas eller blir 0 ska du returnera null.
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

    safe_max_results = max(1, min(int(query.max_results), max(1, MAX_RESULTS_HARD_CAP)))

    tool_config: dict[str, Any] = {
        "type": "file_search",
        "vector_store_ids": [VECTOR_STORE_ID],
        "max_num_results": safe_max_results,
    }

    attribute_filters = build_attribute_filters(query)
    if attribute_filters:
        tool_config["filters"] = attribute_filters

    response_kwargs: dict[str, Any] = {}
    if query.include_search_results or ALWAYS_INCLUDE_RESULTS:
        response_kwargs["include"] = ["file_search_call.results"]

    try:
        first_response = _create_response_with_retry(query, tool_config, response_kwargs, prompt)
    except Exception:
        return {
            "answer": "Arkivverktyget svarade inte just nu. Försök igen om en stund.",
            "sources": [],
            "applied_filters": attribute_filters,
            "search_results_count": None,
            "tool_error": True,
        }

    best_payload = _build_payload_from_response(first_response, query, attribute_filters)

    can_broaden = (
        AUTO_BROADEN_SEARCH
        and not _narrow_filters_present(query)
        and safe_max_results < min(AUTO_BROADEN_TARGET, MAX_RESULTS_HARD_CAP)
    )

    if can_broaden:
        first_sources = best_payload.get("sources", [])
        first_count = len(first_sources) if isinstance(first_sources, list) else 0
        if first_count < max(1, MIN_SOURCES_FOR_SINGLE_PASS):
            broadened_max = max(
                safe_max_results,
                min(AUTO_BROADEN_TARGET, MAX_RESULTS_HARD_CAP),
            )
            broadened_tool_config = dict(tool_config)
            broadened_tool_config["max_num_results"] = broadened_max

            try:
                second_response = _create_response_with_retry(
                    query,
                    broadened_tool_config,
                    response_kwargs,
                    prompt,
                )
                second_payload = _build_payload_from_response(
                    second_response,
                    query,
                    attribute_filters,
                )
                if _payload_score(second_payload) > _payload_score(best_payload):
                    best_payload = second_payload
            except Exception:
                # Keep first successful payload if broadened pass fails.
                pass

    return best_payload


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
        "max_results_hard_cap": MAX_RESULTS_HARD_CAP,
        "auto_broaden_search": AUTO_BROADEN_SEARCH,
        "auto_broaden_target": AUTO_BROADEN_TARGET,
        "min_sources_for_single_pass": MIN_SOURCES_FOR_SINGLE_PASS,
        "always_include_results": ALWAYS_INCLUDE_RESULTS,
    }


@app.post("/search", operation_id="searchSearchPost", summary="Search")
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
