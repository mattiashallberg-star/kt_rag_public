from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import json
from typing import Any

app = FastAPI()

client_kwargs: dict[str, Any] = {"api_key": os.environ["OPENAI_API_KEY"]}
if os.environ.get("OPENAI_PROJECT"):
    client_kwargs["project"] = os.environ["OPENAI_PROJECT"]
if os.environ.get("OPENAI_ORGANIZATION"):
    client_kwargs["organization"] = os.environ["OPENAI_ORGANIZATION"]

client = OpenAI(**client_kwargs)
VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4.1")


class Query(BaseModel):
    question: str
    max_results: int = Field(default=20, ge=1, le=100)
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

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        tools=[tool_config],
        **response_kwargs,
    )

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
        # Fallback om modellen inte returnerar giltig JSON
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
    }


# Behåll denna för Custom GPT Action
@app.post("/search")
def search(query: Query):
    return run_archive_search(query)


# Extra endpoint för vanlig frontend
@app.post("/api/chat")
def api_chat(query: Query):
    return run_archive_search(query)


# Minimal testsida
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html lang="sv">
<head>
  <meta charset="utf-8">
  <title>KT Arkivchat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 16px; }
    textarea, button { font: inherit; }
    textarea { width: 100%; min-height: 100px; margin-bottom: 12px; }
    button { padding: 10px 16px; cursor: pointer; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-top: 20px; }
    .source { margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee; }
    .meta { color: #555; font-size: 0.95em; margin-bottom: 6px; }
    .answer { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>KT Arkivchat</h1>
  <p>Enkel testsida för backend och RAG-sökning.</p>

  <textarea id="question" placeholder="Ställ en fråga om Kyrkans Tidnings arkiv..."></textarea>
  <br>
  <button id="askBtn">Fråga</button>

  <div id="result" class="card" style="display:none;">
    <h2>Svar</h2>
    <div id="answer" class="answer"></div>

    <h3>Källor</h3>
    <div id="sources"></div>
  </div>

  <script>
    const btn = document.getElementById("askBtn");
    const questionEl = document.getElementById("question");
    const resultEl = document.getElementById("result");
    const answerEl = document.getElementById("answer");
    const sourcesEl = document.getElementById("sources");

    btn.addEventListener("click", async () => {
      const question = questionEl.value.trim();
      if (!question) return;

      btn.disabled = true;
      btn.textContent = "Söker...";

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question })
        });

        const data = await res.json();

        resultEl.style.display = "block";
        answerEl.textContent = data.answer || "";
        sourcesEl.innerHTML = "";

        (data.sources || []).forEach((src) => {
          const div = document.createElement("div");
          div.className = "source";

          const meta = document.createElement("div");
          meta.className = "meta";
          meta.textContent =
            `Nummer: ${src.issue || "okänt"} | År: ${src.year || "okänt"} | Sida: ${src.page || "okänd"}`;

          const excerpt = document.createElement("div");
          excerpt.textContent = src.excerpt || "";

          div.appendChild(meta);
          div.appendChild(excerpt);
          sourcesEl.appendChild(div);
        });
      } catch (err) {
        resultEl.style.display = "block";
        answerEl.textContent = "Något gick fel.";
        sourcesEl.innerHTML = "";
      } finally {
        btn.disabled = false;
        btn.textContent = "Fråga";
      }
    });
  </script>
</body>
</html>
"""
