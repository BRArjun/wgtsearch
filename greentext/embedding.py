# greentext/embedding.py
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from typing import List
from PIL import Image
import math

class CLIPEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_text(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of texts; returns (N, D) numpy array normalized to unit length."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)  # (B, D)
            emb = emb.cpu().numpy()
            all_embs.append(emb)
        all_embs = np.vstack(all_embs) if all_embs else np.zeros((0, self.model.config.projection_dim))
        return self._normalize(all_embs)

    def embed_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Embed a list of PIL images; returns (N, D) numpy array normalized to unit length."""
        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            emb = emb.cpu().numpy()
            all_embs.append(emb)
        all_embs = np.vstack(all_embs) if all_embs else np.zeros((0, self.model.config.projection_dim))
        return self._normalize(all_embs)

