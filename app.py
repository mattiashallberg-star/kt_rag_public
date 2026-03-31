from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

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
    hits = client.vector_stores.search(
        vector_store_id=VECTOR_STORE_ID,
        query=query.question,
        max_num_results=5,
    )

    sources = []
    for item in getattr(hits, "data", []):
        text = getattr(item, "text", "") or ""
        filename = getattr(item, "filename", None)
        score = getattr(item, "score", None)
        attributes = getattr(item, "attributes", {}) or {}

        sources.append({
            "filename": filename,
            "score": score,
            "attributes": attributes,
            "excerpt": text[:1500],
        })

    context_blocks = []
    for i, src in enumerate(sources, start=1):
        context_blocks.append(
            f"KÄLLA {i}\n"
            f"Filnamn: {src.get('filename')}\n"
            f"Metadata: {src.get('attributes')}\n"
            f"Utdrag:\n{src.get('excerpt')}"
        )

    context = "\n\n".join(context_blocks)

    prompt = (
        "Du är en arkivassistent för Kyrkans Tidning.\n"
        "Svara endast utifrån källutdragen nedan.\n"
        "Om underlaget inte räcker, säg det tydligt.\n"
        "Nämn inte interna tekniska id:n eller vector store-information.\n"
        "Om datum, nummer/utgåva eller sida framgår av underlaget, ange det.\n"
        "Om det inte framgår, säg inte att du vet det.\n\n"
        f"Fråga: {query.question}\n\n"
        f"{context}"
    )

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
    )

    return {
        "answer": response.output_text,
        "sources": sources,
    }
