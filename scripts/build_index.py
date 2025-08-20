# scripts/build_index.py
import os
from pathlib import Path
import json
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from greentext.embedding import CLIPEmbedder
from greentext.indexing import FAISSIndex
import numpy as np

# Paths (relative)
INDEX_PATH = Path("app/faiss_index.bin")
METADATA_PATH = Path("app/metadata.json")
IMAGES_DIR = Path("app/images")

DATASET_ID = "KrakenAspecto/greentxt_ml"  # your HF dataset id
SPLIT = "train"  # change if dataset uses different split

def ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (Path("app")).mkdir(parents=True, exist_ok=True)

def save_pil_image(img, out_path: Path):
    # Ensure RGB and save as PNG
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out_path, format="PNG", optimize=True)
    except Exception as e:
        raise

def build_index(batch_size: int = 256):
    """
    Loads dataset from Hugging Face, saves local images under app/images/,
    computes embeddings (text + image), builds FAISS index and dumps assets.
    """
    ensure_dirs()
    print(f"⬇️ Loading dataset {DATASET_ID} (split={SPLIT}) from Hugging Face...")
    dataset = load_dataset(DATASET_ID, split=SPLIT)

    embedder = CLIPEmbedder()
    dim = embedder.model.config.projection_dim
    index = FAISSIndex(dim=dim)

    texts = []
    images = []
    metadata = []

    print("📥 Iterating dataset and saving images locally...")
    for i, item in enumerate(tqdm(dataset, desc="saving images")):
        # item['image'] sometimes is PIL Image, sometimes features; handle robustly
        try:
            img = item["image"]
            if not isinstance(img, Image.Image):
                # datasets.Image may expose .convert('RGB') via to_pil?
                try:
                    img = Image.fromarray(img)
                except Exception:
                    # fallback: continue
                    continue
            # Choose an id or fallback to index
            img_id = item.get("post_id") or item.get("id") or f"{i}"
            out_name = f"{img_id}.png"
            out_path = IMAGES_DIR / out_name
            save_pil_image(img, out_path)

            text = item.get("text") or item.get("title") or ""
            texts.append(text)
            images.append(img)
            metadata.append({
                "text": text,
                "image_path": str(out_path)
            })
        except Exception as e:
            # skip problematic items but continue
            print(f"⚠️ Skipped item {i}: {e}")
            continue

    if len(texts) == 0:
        raise RuntimeError("No images/texts were loaded from the dataset.")

    # Embed dataset in batches to avoid OOM
    print("🔎 Creating text embeddings...")
    text_emb = embedder.embed_text(texts, batch_size=64)
    print("🔎 Creating image embeddings...")
    img_emb = embedder.embed_images(images, batch_size=32)

    # Combine embeddings (weighted average)
    print("⚖️ Combining embeddings (text 0.3, image 0.7)...")
    text_weight = 0.3
    image_weight = 0.7
    # Ensure shapes align
    if text_emb.shape[0] != img_emb.shape[0]:
        raise RuntimeError("Text/image embedding length mismatch")
    combined = text_weight * text_emb + image_weight * img_emb
    # normalize combined vectors
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined = combined / norms

    print("➕ Adding to FAISS index...")
    index.add(combined.astype("float32"), metadata)

    print(f"💾 Saving FAISS index -> {INDEX_PATH}")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(INDEX_PATH), str(METADATA_PATH))
    print(f"✅ Index saved. {len(metadata)} items indexed.")
    return str(INDEX_PATH), str(METADATA_PATH)

if __name__ == "__main__":
    build_index()

