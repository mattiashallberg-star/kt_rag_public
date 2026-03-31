from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import json

app = FastAPI()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4.1")


class Query(BaseModel):
    question: str


def run_archive_search(question: str) -> dict:
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
{question}
""".strip()

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID]
        }]
    )

    raw_text = response.output_text

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
            "sources": cleaned_sources
        }

    except Exception:
        # Fallback om modellen inte returnerar giltig JSON
        return {
            "answer": raw_text,
            "sources": []
        }


@app.get("/health")
def health():
    return {"ok": True}


# Behåll denna för Custom GPT Action
@app.post("/search")
def search(query: Query):
    return run_archive_search(query.question)


# Extra endpoint för vanlig frontend
@app.post("/api/chat")
def api_chat(query: Query):
    return run_archive_search(query.question)


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
