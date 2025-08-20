# app/app.py
import gradio as gr
from greentext.search import GreentextCLIPSearchEngine
from pathlib import Path
import scripts.build_index as build_idx_module
import time

INDEX_PATH = Path("app/faiss_index.bin")
METADATA_PATH = Path("app/metadata.json")

def ensure_index():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        # build index (this will download dataset from HF and save images locally)
        print("⚠️ FAISS index or metadata not found. Building index now (this may take a while)...")
        build_idx_module.build_index()
        # small wait to ensure files are flushed
        time.sleep(1)

# Ensure index exists on startup (optional)
ensure_index()

# Initialize engine (this may take a little time for loading model)
engine = GreentextCLIPSearchEngine(index_path=str(INDEX_PATH), metadata_path=str(METADATA_PATH))

def search_fn(query: str, k: int = 5):
    if not query or query.strip() == "":
        return []
    results = engine.search(query, top_k=int(k))
    # Build gallery items as (image_path_or_PIL, caption)
    gallery = []
    for r in results:
        score = f"{r['score']:.4f}"
        text_snip = (r['text'][:200] + "...") if len(r['text']) > 200 else r['text']
        caption = f"Score: {score}\n{text_snip}"
        gallery.append((r['image_path'], caption))
    return gallery

with gr.Blocks(title="Greentext Search") as demo:
    gr.Markdown("# 🍀 Greentext Search (CLIP + FAISS)")
    gr.Markdown("Search greentext images semantically. Results show similarity score and a text snippet.")

    with gr.Row():
        query = gr.Textbox(placeholder="e.g. 'wholesome mom story'...", label="Query", lines=1)
        k = gr.Slider(minimum=1, maximum=12, value=6, step=1, label="Top K")
        search_btn = gr.Button("Search")

    output_gallery = gr.Gallery(label="Results").style(grid=[2], height="auto")

    search_btn.click(fn=search_fn, inputs=[query, k], outputs=output_gallery)
    # Also allow pressing Enter in the textbox to submit
    query.submit(fn=search_fn, inputs=[query, k], outputs=output_gallery)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

