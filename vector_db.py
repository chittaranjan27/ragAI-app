from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=768):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dim = dim

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=str(ids[i]), vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
            if vectors[i] and len(vectors[i]) == self.dim
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
        )
        contexts, sources = [], set()
        for r in results:
            payload = r.payload or {}
            if "text" in payload:
                contexts.append(payload["text"])
                sources.add(payload.get("source", ""))
        return {"contexts": contexts, "sources": list(sources)}
