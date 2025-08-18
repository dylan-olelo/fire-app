import faiss
import pickle
import os
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

# --- Configuration (remains the same) ---
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "text_chunks.pkl"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDINGS_BACKEND = os.environ.get("EMBEDDINGS_BACKEND", "openai")  # local or openai
LOCAL_EMBEDDINGS_MODEL = os.environ.get("LOCAL_EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
OPENAI_EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Image filtering config for relevant visuals
IMAGE_FILTER_BACKEND = os.environ.get("IMAGE_FILTER_BACKEND", "clip")  # clip | none
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "clip-ViT-B-32")
IMAGE_TOP_K = int(os.environ.get("IMAGE_TOP_K", "3"))
IMAGE_MIN_SCORE = float(os.environ.get("IMAGE_MIN_SCORE", "0.28"))
PAGE_WINDOW = int(os.environ.get("PAGE_WINDOW", "1"))
MAX_PAGES_FROM_TEXT = int(os.environ.get("MAX_PAGES_FROM_TEXT", "8"))
IMAGE_META_FILE = "image_meta.pkl"
IMAGE_EMB_FILE = "image_embeddings.npy"

# Performance/behavior toggles
USE_HYDE = os.environ.get("USE_HYDE", "1") not in {"0", "false", "False"}
ENABLE_SENTENCE_LINKS = os.environ.get("ENABLE_SENTENCE_LINKS", "0") in {"1", "true", "True"}
ENABLE_CURATED_KB = os.environ.get("ENABLE_CURATED_KB", "1") not in {"0", "false", "False"}
CURATED_KB_PATH = os.environ.get("CURATED_KB_PATH", "kb/curated_kb.json")

# --- 1. Initialize Flask and Models ---
app = Flask(__name__)

print("Loading knowledge base...")
index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, 'rb') as f:
    chunks = pickle.load(f)

# Load image index (metadata + embeddings) built by build_image_index.py
image_meta = []
image_emb = None
if os.path.exists(IMAGE_META_FILE) and os.path.exists(IMAGE_EMB_FILE):
    print("Loading image index...")
    with open(IMAGE_META_FILE, "rb") as f:
        image_meta = pickle.load(f)
    image_emb = np.load(IMAGE_EMB_FILE).astype("float32")
    print(f"Loaded {len(image_meta)} images with embeddings {image_emb.shape}.")
else:
    print("No image index found; image ranking will be limited.")

print("Initializing OpenAI client for generation (HyDE and final answer)...")
openai_client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
print("Initialization complete. Ready to answer questions.")

# Initialize local retriever for offline embedding if configured
if EMBEDDINGS_BACKEND == "local":
    print(f"Initializing local embedding model for retrieval: {LOCAL_EMBEDDINGS_MODEL}")
    local_embedder = SentenceTransformer(LOCAL_EMBEDDINGS_MODEL)

# Initialize CLIP model for image filtering if enabled
clip_model = None
if IMAGE_FILTER_BACKEND == "clip":
    try:
        clip_model = SentenceTransformer(CLIP_MODEL_NAME)
        print(f"Initialized CLIP image filter: {CLIP_MODEL_NAME}")
    except Exception as e:
        print(f"Failed to initialize CLIP ({CLIP_MODEL_NAME}); disabling image filter. Error: {e}")
        IMAGE_FILTER_BACKEND = "none"


# ---- Lightweight caches & curated KB ----
embedding_cache: dict[str, np.ndarray] = {}
hyde_cache: dict[str, str] = {}
answer_cache: dict[tuple[str, str], dict] = {}

curated_kb = {}
if ENABLE_CURATED_KB and os.path.exists(CURATED_KB_PATH):
    try:
        import json
        with open(CURATED_KB_PATH, "r", encoding="utf-8") as f:
            curated_kb = json.load(f)
        print(f"Loaded curated KB from {CURATED_KB_PATH}")
    except Exception as e:
        print(f"Failed to load curated KB: {e}")

def generate_text_via_openai(system_message: str, user_prompt: str, max_tokens: int) -> str:
    """Call OpenAI Chat Completions API to generate text deterministically."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    else:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({"role": "user", "content": user_prompt})

    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


def embed_query(text: str) -> np.ndarray:
    # Simple in-memory caching for speed across repeated queries
    if text in embedding_cache:
        return embedding_cache[text]
    api_key = os.environ.get("OPENAI_API_KEY")
    if EMBEDDINGS_BACKEND == "local":
        vec = local_embedder.encode([text])
        arr = np.array(vec, dtype=np.float32)
    else:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        resp = openai_client.embeddings.create(model=OPENAI_EMBEDDINGS_MODEL, input=[text])
        arr = np.array([resp.data[0].embedding], dtype=np.float32)
    embedding_cache[text] = arr
    return arr
    
def parse_source(source_str: str) -> tuple[str, int]:
    try:
        file_part, page_part = source_str.split(", page ")
        return file_part.strip(), int(page_part.strip())
    except Exception:
        return source_str.split(",")[0].strip(), 1
    
def get_pdf_page(ch: dict) -> tuple[str, int]:
    pdf = ch.get("pdf")
    page = ch.get("page")
    if pdf and page:
        return pdf, int(page)
    return parse_source(ch["source"])

def gather_top_pages(retrieved_idx: list[int], target_model: str, max_pages: int) -> list[tuple[str, int]]:
    seen = []
    seen_set = set()
    for i in retrieved_idx:
        ch = chunks[i]
        if ch.get("model") != target_model:
            continue
        pdf, page = get_pdf_page(ch)
        t = (pdf, page)
        if t not in seen_set:
            seen_set.add(t)
            seen.append(t)
        if len(seen) >= max_pages:
            break
    return seen  # ordered unique

def expand_pages(pages: list[tuple[str,int]], window: int) -> set[tuple[str,int]]:
    out = set()
    for pdf, page in pages:
        for p in range(page - window, page + window + 1):
            if p >= 1:
                out.add((pdf, p))
    return out

def rank_images_for_text(text: str, candidate_rows: list[int]) -> list[int]:
    if clip_model is None or image_emb is None or not candidate_rows:
        return candidate_rows[:IMAGE_TOP_K]
    q = clip_model.encode([text], convert_to_tensor=False, normalize_embeddings=True)[0]
    q = np.asarray(q, dtype=np.float32)
    cand = image_emb[candidate_rows]          # (K, 512)
    scores = cand @ q                         # cosine similarity (embeddings are normalized)
    order = np.argsort(-scores)
    ranked = [candidate_rows[i] for i in order if scores[i] >= IMAGE_MIN_SCORE]
    return ranked[:IMAGE_TOP_K]

def rank_images_for_text_with_scores(text: str, candidate_rows: list[int]) -> list[tuple[int, float]]:
    if clip_model is None or image_emb is None or not candidate_rows:
        return [(r, 0.0) for r in candidate_rows[:IMAGE_TOP_K]]
    q = clip_model.encode([text], convert_to_tensor=False, normalize_embeddings=True)[0]
    q = np.asarray(q, dtype=np.float32)
    cand = image_emb[candidate_rows]          # (K, 512)
    scores = cand @ q                         # cosine similarity (normalized)
    order = np.argsort(-scores)
    return [(candidate_rows[i], float(scores[i])) for i in order]

def split_into_sentences(text: str) -> list[str]:
    import re
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def pdf_base(name: str) -> str:
    # 'foo.pdf' -> 'foo'; 'foo' -> 'foo'
    return os.path.splitext(name)[0]

def build_doc_link(model: str, pdf_name: str, page: int) -> str:
    # Ensure .pdf extension in link
    pdf_with_ext = pdf_name if pdf_name.lower().endswith(".pdf") else f"{pdf_name}.pdf"
    return f"/docs/{model}/{pdf_with_ext}#page={page}"



# --- 2. Create the API Endpoint ---
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'question' not in data or 'model' not in data:
        return jsonify({"error": "Request must include 'question' and 'model'."}), 400

    question = data['question']
    target_model = data['model']
    print(f"Received question for model '{target_model}': {question}")

    # Quick full-response cache
    cache_key = (target_model, question.strip())
    if cache_key in answer_cache:
        return jsonify(answer_cache[cache_key])

    # --- STEP A: Optional HyDE for stronger recall ---
    if USE_HYDE:
        print("Generating hypothetical document for search via OpenAI API...")
        if question in hyde_cache:
            hypothetical_doc = hyde_cache[question]
        else:
            hyde_user_prompt = (
                f"Generate a concise, factual paragraph that answers the following question about the {target_model} "
                f"as if it were from the vehicle's Emergency Response Guide.\n\n"
                f"Question: {question}"
            )
            try:
                hypothetical_doc = generate_text_via_openai(
                    system_message="You are a helpful assistant for fire fighters.",
                    user_prompt=hyde_user_prompt,
                    max_tokens=180,
                )
                hyde_cache[question] = hypothetical_doc
            except Exception as e:
                return jsonify({"error": f"HyDE generation failed: {str(e)}"}), 500
        print(f"HyDE Doc: {hypothetical_doc}")
        retrieval_text = hypothetical_doc
    else:
        retrieval_text = question

    # --- STEP B: RETRIEVE - Search using the HyDE document ---
    print("Searching index...")
    search_embedding = embed_query(retrieval_text)
    
    # Ensure embedding dimensionality matches index
    if search_embedding.shape[1] != index.d:
        return jsonify({
            "error": (
                f"Embedding dimension {search_embedding.shape[1]} does not match index dimension {index.d}. "
                "Recreate the index with create_index.py using the same EMBEDDINGS_MODEL configured on the server."
            )
        }), 500

    k_search = 10
    distances, indices = index.search(search_embedding, k_search)

    retrieved_idx = list(indices[0])

    # Pages we care about (from retrieved chunks), with ± window
    top_pages = gather_top_pages(retrieved_idx, target_model, MAX_PAGES_FROM_TEXT)
    expanded = expand_pages(top_pages, PAGE_WINDOW)

    # Normalize to base names for image matching
    expanded_base = set((pdf_base(pdf), page) for (pdf, page) in expanded)

    candidate_rows = []
    if image_meta and image_emb is not None:
        for row, rec in enumerate(image_meta):
            if rec["model"] != target_model:
                continue
            # rec["pdf"] is a base name from image index; compare against expanded_base
            if (rec["pdf"], rec["page"]) not in expanded_base:
                continue
            if rec.get("is_common"):
                continue
            if rec.get("width", 1) < 120 or rec.get("height", 1) < 120:
                continue
            candidate_rows.append(row)

    # Build context for synthesis only
    context_text = ""
    for i in retrieved_idx:
        ch = chunks[i]
        if ch.get('model') == target_model:
            context_text += ch['text'] + "\n\n"

    print(f"Context Text: {context_text}")
    
    if not context_text:
        return jsonify({"answer": "I could not find any relevant information for that model, even with an expanded search. Please try rephrasing your question.", "sources": [], "images": []})

    # --- STEP C: Fast curated KB path (optional) ---
    selected_images: list[str] = []
    curated_hit = None
    if ENABLE_CURATED_KB and curated_kb:
        def norm(s: str) -> str:
            return ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in s).strip()
        qn = norm(question)
        model_kb = curated_kb.get(target_model, {})
        intents = (model_kb or {}).get("intents", {})
        for intent_name, entry in intents.items():
            patterns = [p.lower() for p in entry.get("patterns", [])]
            if any(p in qn for p in patterns):
                curated_hit = entry
                selected_images = [p for p in entry.get("images", [])]
                break

    # --- STEP D: SYNTHESIZE - Generate the final answer ---
    if curated_hit and curated_hit.get("answer"):
        final_answer = curated_hit["answer"].strip()
    else:
        print("Generating final answer via OpenAI API...")
        system_prompt = """You are an expert assistant for fire fighters. Your task is to answer the fire fighter's original question based ONLY on the provided text from the vehicle's Emergency Response Guide. Be clear, concise, and prioritize immediate safety actions. If the information isn't in the provided text, say "The provided guide does not contain that information." Do not make anything up. Structure your answer with clear steps if possible."""

        final_user_prompt = (
            f"Original Question: {question}\n\n"
            f"Emergency Guide Text:\n{context_text}\n\n"
            f"Provide the best possible answer now."
        )
        try:
            final_answer = generate_text_via_openai(
                system_message=system_prompt,
                user_prompt=final_user_prompt,
                max_tokens=600,
            )
        except Exception as e:
            return jsonify({"error": f"Answer generation failed: {str(e)}"}), 500
    
    sentences = split_into_sentences(final_answer)
    inline = []
    used_rows = set()

    # If we have curated images, use them directly (fast, deterministic)
    if selected_images:
        for img in selected_images[:IMAGE_TOP_K]:
            inline.append({"sentence": "relevant image", "image": img, "image_score": 1.0, "source_link": None})
    else:
        # Optionally compute per-sentence links (can be slow); default disabled
        if ENABLE_SENTENCE_LINKS:
            for sent in sentences:
                best_link = None
                try:
                    s_vec = embed_query(sent)
                    s_dist, s_idx = index.search(s_vec, 50)
                    for j in s_idx[0]:
                        ch = chunks[j]
                        if ch.get("model") != target_model:
                            continue
                        pdf, page = get_pdf_page(ch)
                        if not expanded or (pdf, page) in expanded:
                            best_link = build_doc_link(target_model, pdf, page)
                            break
                except Exception:
                    best_link = None

                best_img = None
                best_score = None
                if candidate_rows:
                    remain = [r for r in candidate_rows if r not in used_rows]
                    ranked_pairs = rank_images_for_text_with_scores(sent, remain)
                    for row, score in ranked_pairs:
                        if score >= IMAGE_MIN_SCORE:
                            best_img = image_meta[row]["path"]
                            best_score = round(float(score), 3)
                            used_rows.add(row)
                            break

                inline.append({"sentence": sent, "image": best_img, "image_score": best_score, "source_link": best_link})
        else:
            # Single pass, score full answer against images found on nearby pages
            remain = [r for r in candidate_rows if r not in used_rows]
            ranked_pairs = rank_images_for_text_with_scores(final_answer, remain)
            for row, score in ranked_pairs:
                if score >= IMAGE_MIN_SCORE and len(inline) < IMAGE_TOP_K:
                    inline.append({
                        "sentence": "relevant image",
                        "image": image_meta[row]["path"],
                        "image_score": round(float(score), 3),
                        "source_link": None
                    })
                    used_rows.add(row)
            if not inline and remain:
                # As a last resort, just return the first candidate image(s)
                for r in remain[:IMAGE_TOP_K]:
                    inline.append({"sentence": "image", "image": image_meta[r]["path"], "image_score": None, "source_link": None})

        # Build a compact debug preview to help tuning
    candidate_preview = []
    if candidate_rows:
        global_rank = rank_images_for_text_with_scores(final_answer, candidate_rows)[:10]
        for row, score in global_rank:
            rec = image_meta[row]
            candidate_preview.append({
                "path": rec["path"],
                "pdf": rec["pdf"],
                "page": rec["page"],
                "width": rec.get("width"),
                "height": rec.get("height"),
                "is_common": bool(rec.get("is_common", False)),
                "score": round(float(score), 3),
            })

    debug_info = {
        "config": {
            "IMAGE_TOP_K": IMAGE_TOP_K,
            "IMAGE_MIN_SCORE": IMAGE_MIN_SCORE,
            "PAGE_WINDOW": PAGE_WINDOW,
            "MAX_PAGES_FROM_TEXT": MAX_PAGES_FROM_TEXT,
        },
        "pages": {
            "top_pages": top_pages,                          # ordered unique from text retrieval
            "expanded_pages": sorted(list(expanded)),
            "expanded_base": sorted(list(expanded_base)),
        },
        "candidates_count": len(candidate_rows),
        "top_candidate_images": candidate_preview,
    }

    # If curated entry provides explicit source pages, prefer them
    if curated_hit and curated_hit.get("pdf") and curated_hit.get("pages"):
        pdf_name = curated_hit.get("pdf")
        curated_pages = [(pdf_name, int(p)) for p in curated_hit.get("pages", []) if int(p) >= 1]
        top_sources = curated_pages[:3]
    else:
        top_sources = [ (pdf, page) for (pdf, page) in top_pages ][:3]
    sources = [ f"{pdf}, page {page}" for (pdf, page) in top_sources ]
    source_links = [ build_doc_link(target_model, pdf, page) for (pdf, page) in top_sources ]

    resp_payload = {
        "answer": final_answer,
        "inline": inline,
        "images": [i["image"] for i in inline if i["image"]],
        "sources": sources,
        "source_links": source_links,
        "debug": debug_info
    }
    answer_cache[cache_key] = resp_payload
    return jsonify(resp_payload)


@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename: str):
    # Serve files under the project 'images' directory
    sanitized = filename.replace('\\', '/').lstrip('/')
    if sanitized.startswith('images/'):
        sanitized = sanitized[len('images/'):]
    return send_from_directory('images', sanitized)


@app.route('/docs/<path:filename>', methods=['GET'])
def serve_doc(filename: str):
    # Serve files under the project 'documents' directory
    sanitized = filename.replace('\\', '/').lstrip('/')
    if sanitized.startswith('documents/'):
        sanitized = sanitized[len('documents/'):]
    return send_from_directory('documents', sanitized)


@app.route('/', methods=['GET'])
def debug_page():
    # Minimal HTML page to test the API and preview images/sources
    return (
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fire App Debug</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 20px; line-height: 1.4; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    input, select, button { padding: 8px 10px; font-size: 14px; }
    button { cursor: pointer; }
    .answer { white-space: pre-wrap; background: #fafafa; padding: 12px; border: 1px solid #eee; border-radius: 6px; }
    .images { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin-top: 10px; }
    .images img { width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
    .sources { margin-top: 10px; }
    .sources a { display: block; margin: 4px 0; }
  </style>
  <script>
    async function ask() {
      const q = document.getElementById('q').value;
      const model = document.getElementById('model').value;
      const resEl = document.getElementById('result');
      resEl.textContent = 'Loading...';
      const r = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, model })
      });
      const data = await r.json();
      document.getElementById('answer').textContent = data.answer || JSON.stringify(data);
      const imgWrap = document.getElementById('images');
      imgWrap.innerHTML = '';
      (data.images || []).forEach(src => {
        const img = document.createElement('img');
        img.src = src.startsWith('/images/') ? src : ('/' + src.replace(/^\//, ''));
        if (!img.src.includes('/images/')) img.src = '/images/' + src.replace(/^images\//, '');
        img.alt = 'related image';
        imgWrap.appendChild(img);
      });
      const sources = document.getElementById('sources');
      sources.innerHTML = '';
      const srcs = data.sources || [];
      const links = data.source_links || [];
      for (let i = 0; i < srcs.length; i++) {
        const a = document.createElement('a');
        a.textContent = srcs[i];
        a.href = links[i] || '#';
        a.target = '_blank';
        sources.appendChild(a);
      }
      resEl.textContent = '';
    }
  </script>
</head>
<body>
  <h2>Fire App Debug</h2>
  <div class="row">
    <input id="q" size="60" placeholder="Ask a question... e.g., Where is the first responder loop?" />
    <select id="model">
      <option value="Tesla_Model_3">Tesla_Model_3</option>
      <option value="Tesla_Model_S">Tesla_Model_S</option>
      <option value="Tesla_Model_Y">Tesla_Model_Y</option>
    </select>
    <button onclick="ask()">Ask</button>
    <span id="result"></span>
  </div>
  <h3>Answer</h3>
  <div id="answer" class="answer"></div>
  <h3>Images</h3>
  <div id="images" class="images"></div>
  <h3>Sources</h3>
  <div id="sources" class="sources"></div>
</body>
</html>
        """
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)