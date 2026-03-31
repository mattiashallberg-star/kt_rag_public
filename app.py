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
    response = client.responses.create(
        model="gpt-4.1",
        input=query.question,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID]
        }]
    )
    return {"answer": response.output_text}
