import os, hashlib, time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder

app = FastAPI(title="STELLA Embedding Service")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant.railway.internal:6333")

embed_model = None
reranker = None
qdrant = None

COLLECTIONS = {
    "knowledge": "stella_knowledge",
    "products": "stella_products",
    "decisions": "stella_decisions",
    "lessons": "stella_lessons",
}

@app.on_event("startup")
async def startup():
    global embed_model, reranker, qdrant
    embed_model = SentenceTransformer("BAAI/bge-m3")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    qdrant = QdrantClient(url=QDRANT_URL)
    for name, collection in COLLECTIONS.items():
        existing = [c.name for c in qdrant.get_collections().collections]
        if collection not in existing:
            qdrant.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )

class IndexRequest(BaseModel):
    text: str
    collection: str = "knowledge"
    source: str = ""
    project: str = ""
    metadata: dict = {}

class SearchRequest(BaseModel):
    query: str
    collection: str = "knowledge"
    project: str = ""
    top_k: int = 10
    rerank_top: int = 5

class MultiSearchRequest(BaseModel):
    query: str
    collections: List[str] = ["knowledge", "products", "decisions"]
    project: str = ""
    top_k_per_collection: int = 5
    rerank_top: int = 5

@app.get("/health")
async def health():
    try:
        collections = {name: qdrant.get_collection(col).points_count
                       for name, col in COLLECTIONS.items()}
        return {"status": "ok", "collections": collections}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/index")
async def index_document(req: IndexRequest):
    collection = COLLECTIONS.get(req.collection, COLLECTIONS["knowledge"])
    doc_id = hashlib.md5(req.text.encode()).hexdigest()
    chunks = chunk_text(req.text, max_words=400)
    points = []
    for i, chunk in enumerate(chunks):
        embedding = embed_model.encode(chunk).tolist()
        points.append(PointStruct(
            id=abs(hash(f"{doc_id}_{i}")) % (2**63),
            vector=embedding,
            payload={
                "text": chunk,
                "source": req.source,
                "project": req.project,
                "doc_id": doc_id,
                "chunk_index": i,
                "indexed_at": time.time(),
                **req.metadata
            }
        ))
    qdrant.upsert(collection_name=collection, points=points)
    return {"indexed_chunks": len(points), "collection": req.collection}

@app.post("/search")
async def search(req: SearchRequest):
    collection = COLLECTIONS.get(req.collection, COLLECTIONS["knowledge"])
    query_embedding = embed_model.encode(req.query).tolist()
    query_filter = None
    if req.project:
        query_filter = Filter(must=[
            FieldCondition(key="project", match=MatchValue(value=req.project))
        ])
    results = qdrant.search(
        collection_name=collection,
        query_vector=query_embedding,
        query_filter=query_filter,
        limit=req.top_k
    )
    if not results:
        return {"results": []}
    passages = [r.payload["text"] for r in results]
    pairs = [[req.query, p] for p in passages]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return {
        "results": [
            {"text": r.payload["text"], "source": r.payload.get("source", ""),
             "project": r.payload.get("project", ""), "score": float(s)}
            for r, s in ranked[:req.rerank_top]
        ]
    }

@app.post("/search_all")
async def search_all(req: MultiSearchRequest):
    all_results = []
    query_embedding = embed_model.encode(req.query).tolist()
    for col_name in req.collections:
        collection = COLLECTIONS.get(col_name)
        if not collection:
            continue
        query_filter = None
        if req.project:
            query_filter = Filter(must=[
                FieldCondition(key="project", match=MatchValue(value=req.project))
            ])
        results = qdrant.search(
            collection_name=collection,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=req.top_k_per_collection
        )
        for r in results:
            all_results.append({
                "text": r.payload["text"],
                "source": r.payload.get("source", ""),
                "collection": col_name,
                "project": r.payload.get("project", ""),
                "score": float(r.score)
            })
    if all_results:
        pairs = [[req.query, r["text"]] for r in all_results]
        scores = reranker.predict(pairs)
        for r, s in zip(all_results, scores):
            r["rerank_score"] = float(s)
        all_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return {"results": all_results[:req.rerank_top]}

def chunk_text(text, max_words=400):
    sections = text.split("\n## ")
    chunks = []
    for section in sections:
        words = section.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                if chunk.strip():
                    chunks.append(chunk.strip())
        elif section.strip():
            chunks.append(section.strip())
    return chunks if chunks else [text]
