from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import json

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/search")
def search(query: Query):
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

    response = client.responses.create(
        model="gpt-4.1",
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
            raise ValueError("Svar var inte ett objekt")

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
        return {
            "answer": raw_text,
            "sources": []
        }
