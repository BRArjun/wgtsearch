# greentext/indexing.py
import faiss
import numpy as np
import json
from typing import List, Dict

class FAISSIndex:
    def __init__(self, dim: int, index_path: str = None, metadata_path: str = None):
        """
        Uses inner product (IP) with normalized vectors -> cosine similarity.
        """
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

        if index_path and metadata_path:
            self.load(index_path, metadata_path)

    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        embeddings: (N, D) float32, should be normalized already.
        metadata: list of dicts with same length.
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_emb: np.ndarray, top_k: int = 5):
        """
        query_emb: (1, D) float32, should be normalized.
        returns list of dicts {"score": float, "text": ..., "image_path": ...}
        """
        if query_emb.dtype != np.float32:
            query_emb = query_emb.astype("float32")
        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            results.append({
                "score": float(score),  # cosine similarity
                "text": item.get("text", ""),
                "image_path": item.get("image_path", "")
            })
        return results

    def save(self, index_path: str, metadata_path: str):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        # try to set dim from index
        try:
            self.dim = self.index.d
        except:
            pass

