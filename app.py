from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class Query(BaseModel):
    question: str

@app.post("/search")
def search(query: Query):
    response = client.responses.create(
        model="gpt-5.3",
        input=query.question,
        tools=[{
            "type": "file_search",
            "vector_store_ids": ["vs_69c50c8c22d08191a741bcbe025605a7"]
        }]
    )

    return {
        "answer": response.output_text
    }
