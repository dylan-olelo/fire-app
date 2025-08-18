import os, pickle, hashlib
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

IMAGES_ROOT = "images"
IMAGE_META_FILE = "image_meta.pkl"
IMAGE_EMB_FILE = "image_embeddings.npy"

# Tuning knobs
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "clip-ViT-B-32")
MIN_W = int(os.environ.get("IMG_MIN_W", "120"))
MIN_H = int(os.environ.get("IMG_MIN_H", "120"))
COMMON_HASH_FREQ = int(os.environ.get("IMG_COMMON_HASH_FREQ", "6"))

def sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()

def iter_images() -> List[Dict]:
    records = []
    for model in os.listdir(IMAGES_ROOT):
        model_dir = os.path.join(IMAGES_ROOT, model)
        if not os.path.isdir(model_dir):
            continue
        for pdf in os.listdir(model_dir):
            pdf_dir = os.path.join(model_dir, pdf)
            if not os.path.isdir(pdf_dir):
                continue
            for page in os.listdir(pdf_dir):
                page_dir = os.path.join(pdf_dir, page)
                if not os.path.isdir(page_dir):
                    continue
                page_num = int(page.replace("page_", "")) if page.startswith("page_") else None
                for fname in os.listdir(page_dir):
                    if not (fname.lower().endswith((".png", ".jpg", ".jpeg"))):
                        continue
                    path = os.path.join(page_dir, fname)
                    try:
                        with open(path, "rb") as f:
                            raw = f.read()
                        h = sha1_bytes(raw)
                        img = Image.open(path).convert("RGB")
                        w, hpx = img.size
                    except Exception:
                        continue
                    # Keep basic metadata; store POSIX-style path for serving
                    records.append({
                        "path": path.replace("\\", "/"),
                        "model": model,
                        "pdf": pdf,
                        "page": page_num,
                        "width": w,
                        "height": hpx,
                        "sha1": sha1_bytes(raw)
                    })
    return records

def main():
    print("Scanning images...")
    meta = iter_images()
    if not meta:
        print("No images found.")
        return

    # Compute hash frequencies (helps flag very common icons)
    freq = {}
    for r in meta:
        freq[r["sha1"]] = freq.get(r["sha1"], 0) + 1
    for r in meta:
        r["is_common"] = freq[r["sha1"]] >= COMMON_HASH_FREQ

    # Filter out tiny images early
    meta = [r for r in meta if r["width"] >= MIN_W and r["height"] >= MIN_H]

    print(f"Kept {len(meta)} images after filtering. Embedding with CLIP...")
    clip = SentenceTransformer(CLIP_MODEL_NAME)

    embs = []
    for i, r in enumerate(meta, 1):
        try:
            img = Image.open(r["path"]).convert("RGB")
            vec = clip.encode([img], convert_to_tensor=False, normalize_embeddings=True)[0]
        except Exception:
            vec = np.zeros((512,), dtype=np.float32)  # fallback shape; CLIP-B/32 is 512-D
        embs.append(vec)
        if i % 200 == 0:
            print(f"Embedded {i}/{len(meta)}")

    embs = np.array(embs, dtype=np.float32)

    print(f"Saving {IMAGE_META_FILE} and {IMAGE_EMB_FILE}...")
    with open(IMAGE_META_FILE, "wb") as f:
        pickle.dump(meta, f)
    np.save(IMAGE_EMB_FILE, embs)
    print("Done.")

if __name__ == "__main__":
    main()