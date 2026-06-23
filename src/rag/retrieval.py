from rank_bm25 import BM25Okapi

def tokenize(text: str):
    return [t.lower() for t in text.split()]

class HybridRetriever:
    def __init__(self, chunks, vectorstore):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.corpus = [tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus) if chunks else None

    def sparse(self, query: str, k: int = 5):
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [{'chunk': self.chunks[i], 'score': float(score), 'source': 'bm25', 'rank': r+1} for r, (i, score) in enumerate(ranked)]

    def dense(self, query: str, k: int = 5):
        out = self.vectorstore.query(query, k=k)
        ids = out.get('ids', [[]])[0]
        scores = out.get('distances', [[]])[0]
        results = []
        for idx, cid in enumerate(ids):
            chunk = next((c for c in self.chunks if c['id'] == cid), None)
            if chunk:
                results.append({'chunk': chunk, 'score': float(scores[idx]) if idx < len(scores) else 0.0, 'source': 'dense', 'rank': idx+1})
        return results

    def fuse(self, dense_results, sparse_results, k: int = 5):
        scores = {}
        by_id = {c['id']: c for c in self.chunks}
        for rank, item in enumerate(dense_results, start=1):
            key = item['chunk']['id']
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        for rank, item in enumerate(sparse_results, start=1):
            key = item['chunk']['id']
            scores[key] = scores.get(key, 0) + 1.0 / (60 + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{'chunk': by_id[cid], 'score': score} for cid, score in ranked]
