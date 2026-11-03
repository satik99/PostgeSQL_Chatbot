from fastapi import FastAPI
from pydantic import BaseModel

from generate import answer

app = FastAPI(title="PostgreSQL Docs Assistant")


class Ask(BaseModel):
    question: str
    mode: str = "rerank"   # dense | hybrid | rerank


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Ask):
    r = answer(body.question, mode=body.mode)
    return {"answer": r["answer"], "sources": r["sources"]}