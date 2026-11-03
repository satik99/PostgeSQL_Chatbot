import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

CHUNKS = Path("data/chunks_pg.jsonl")
COLLECTION = "postgres_docs"
MODEL = "BAAI/bge-small-en-v1.5"

chunks = [json.loads(line) for line in CHUNKS.open()]
print(f"Embedding {len(chunks)} chunks...")

model = SentenceTransformer(MODEL)
vectors = model.encode(
    [c["text"] for c in chunks],
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)

client = QdrantClient(path="data/qdrant")
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
)
client.upsert(
    collection_name=COLLECTION,
    points=[
        PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
        for i in range(len(chunks))
    ],
)
print("Done.")