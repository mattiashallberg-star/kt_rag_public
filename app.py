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
        max_num_results=3,
    )

    rows = []
    for item in getattr(hits, "data", []):
        rows.append({
            "filename": getattr(item, "filename", None),
            "attributes": getattr(item, "attributes", None),
            "score": getattr(item, "score", None),
            "text": (getattr(item, "text", "") or "")[:500],
        })

    return {"results": rows}
