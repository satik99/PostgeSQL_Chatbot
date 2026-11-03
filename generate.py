from llm import chat
from retrieve import SEARCH

SYSTEM = (
    "You answer questions about PostgreSQL using ONLY the numbered context passages. "
    "Cite sources inline like [1] or [2][3]. "
    "If the context does not contain the answer, reply exactly: "
    "\"I couldn't find this in the documentation.\" Do not use outside knowledge."
)


def answer(question: str, k: int = 5, mode: str = "dense"):
    hits = SEARCH[mode](question, k=k)
    context = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    text = chat(SYSTEM, f"Context:\n{context}\n\nQuestion: {question}")
    return {
        "question": question,
        "answer": text,
        "contexts": [h["text"] for h in hits],
        "sources": [{"n": i + 1, "url": h["url"]} for i, h in enumerate(hits)],
    }


if __name__ == "__main__":
    for q in [
        "What is the difference between VACUUM and VACUUM FULL?",
        "How do I make coffee with PostgreSQL?",
    ]:
        r = answer(q, mode="hybrid")
        print(f"\nQ: {q}\nA: {r['answer']}")