from typing import Mapping

from rank_bm25 import BM25Okapi


def tokenize(text: str):
    return [token.lower() for token in text.split()]


def filter_chunks(chunks, filters: Mapping[str, str] | None = None):
    """Keep chunks matching every requested metadata filter."""
    if not filters:
        return list(chunks)
    return [
        chunk
        for chunk in chunks
        if all(str(chunk.get(key)) == str(value) for key, value in filters.items())
    ]


class HybridRetriever:
    def __init__(self, chunks, vectorstore):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus) if chunks else None

    def sparse(self, query: str, k: int = 5):
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k]
        return [
            {
                "chunk": self.chunks[index],
                "score": float(score),
                "source": "bm25",
                "rank": rank + 1,
            }
            for rank, (index, score) in enumerate(ranked)
        ]

    def dense(self, query: str, k: int = 5):
        if not self.chunks:
            return []
        output = self.vectorstore.query(query, k=k)
        ids = output.get("ids", [[]])[0]
        distances = output.get("distances", [[]])[0]
        results = []
        for index, chunk_id in enumerate(ids):
            chunk = next(
                (chunk for chunk in self.chunks if chunk["id"] == chunk_id),
                None,
            )
            if chunk:
                results.append(
                    {
                        "chunk": chunk,
                        "score": float(distances[index]) if index < len(distances) else 0.0,
                        "source": "dense",
                        "rank": index + 1,
                    }
                )
        return results

    def fuse(self, dense_results, sparse_results, k: int = 5):
        scores = {}
        by_id = {chunk["id"]: chunk for chunk in self.chunks}
        for rank, item in enumerate(dense_results, start=1):
            chunk_id = item["chunk"]["id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (60 + rank)
        for rank, item in enumerate(sparse_results, start=1):
            chunk_id = item["chunk"]["id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (60 + rank)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [{"chunk": by_id[chunk_id], "score": score} for chunk_id, score in ranked]
