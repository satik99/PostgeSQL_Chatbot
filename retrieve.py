import json
import re
from pathlib import Path

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

COLLECTION = "postgres_docs"
MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # fast on CPU
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = SentenceTransformer(MODEL)
_client = QdrantClient(path="data/qdrant")
_reranker = None

# BM25 index. Line order matches the point ids used in index.py.
_chunks = [json.loads(l) for l in Path("data/chunks_pg.jsonl").open()]


def tokenize(text: str):
    return re.findall(r"[a-z0-9_]+", text.lower())


_bm25 = BM25Okapi([tokenize(c["text"]) for c in _chunks])


def dense_search(query: str, k: int = 5):
    vec = _model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    hits = _client.query_points(COLLECTION, query=vec, limit=k).points
    return [{"score": h.score, "id": h.id, **h.payload} for h in hits]


def bm25_search(query: str, k: int = 30):
    scores = _bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"score": float(scores[i]), "id": i, **_chunks[i]} for i in top]


def hybrid_search(query: str, k: int = 5, pool: int = 30, rrf_k: int = 60):
    """Reciprocal Rank Fusion of dense and BM25 results."""
    fused, by_id = {}, {}
    for results in (dense_search(query, pool), bm25_search(query, pool)):
        for rank, r in enumerate(results, 1):
            fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (rrf_k + rank)
            by_id[r["id"]] = r
    ranked = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**by_id[i], "score": fused[i]} for i in ranked]


def rerank_search(query: str, k: int = 5, pool: int = 30):
    """Hybrid top-`pool`, then rescore with a cross-encoder and keep the top k."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    cands = hybrid_search(query, k=pool, pool=pool)
    scores = _reranker.predict([(query, c["text"]) for c in cands])
    ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)[:k]
    return [{**c, "score": float(s)} for s, c in ranked]


SEARCH = {"dense": dense_search, "hybrid": hybrid_search, "rerank": rerank_search}

if __name__ == "__main__":
    for q in ["What does error code 23505 mean?", "What does shared_buffers control?"]:
        for name, fn in SEARCH.items():
            print(f"\n=== {name}: {q}")
            for r in fn(q):
                print(f"{r['score']:.3f}  {r['url']}")