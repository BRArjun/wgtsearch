# greentext/search.py
from .embedding import CLIPEmbedder
from .indexing import FAISSIndex
import numpy as np
from pathlib import Path
import os

class GreentextCLIPSearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32",
                 index_path: str = None, metadata_path: str = None, device: str = None):
        self.embedder = CLIPEmbedder(model_name=model_name, device=device)
        self.index = None
        self.index_path = index_path
        self.metadata_path = metadata_path

        if index_path and metadata_path and Path(index_path).exists() and Path(metadata_path).exists():
            # load existing index
            self.index = FAISSIndex(dim=self.embedder.model.config.projection_dim,
                                     index_path=index_path, metadata_path=metadata_path)

    def load_index(self, index_path: str, metadata_path: str):
        self.index = FAISSIndex(dim=self.embedder.model.config.projection_dim,
                                 index_path=index_path, metadata_path=metadata_path)
        self.index_path = index_path
        self.metadata_path = metadata_path

    def search(self, query: str, top_k: int = 6):
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() or initialize with existing index_path")
        q_emb = self.embedder.embed_text([query])  # (1, D), normalized
        return self.index.search(q_emb.astype("float32"), top_k=top_k)

