# streamlit_app.py
"""
Multimodal Research Copilot — Complete runnable file (Windows-friendly)
- Uses Hugging Face InferenceClient for embeddings (feature_extraction)
- Stores embeddings in ChromaDB (duckdb+parquet)
- File upload (PDF/TXT), preview, copy-to-form
- Chunking & ingestion (configurable chunk size & overlap)
- Robust to different chromadb versions (safe persist, multiple constructors)
- Streamlit UI: add docs, chunked ingestion, search, list, clear

Make sure your .env contains:
    HUGGINGFACE_HUB_TOKEN=hf_xxx_your_token_here
"""
import os
import uuid
import json
import math
import re
from datetime import datetime
from typing import List, Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from huggingface_hub import InferenceClient
import requests

# ---------------------------
# Config
# ---------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")  # *use this name* in your .env
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"
UPLOADS_DIR = "./uploads"

if not HF_TOKEN:
    raise RuntimeError("⚠️ Missing Hugging Face token. Add it to your .env as HUGGINGFACE_HUB_TOKEN=...")

# initialize HF client once
hf_client = InferenceClient(api_key=HF_TOKEN)

# ---------------------------
# Safe helpers
# ---------------------------
def safe_persist(client_obj):
    """Call client.persist() only if implemented by this chromadb version."""
    try:
        if hasattr(client_obj, "persist") and callable(getattr(client_obj, "persist")):
            client_obj.persist()
    except Exception:
        # ignore persistence errors; keep running
        pass

# ---------------------------
# Get embedding (using InferenceClient)
# ---------------------------
def get_embedding(text: str, model: str = DEFAULT_MODEL) -> List[float]:
    raw = hf_client.feature_extraction(text, model=model)

    # Convert numpy arrays to Python lists
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()

    # token-level embeddings -> mean pool
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        arr = np.array(raw, dtype=float)
        vec = arr.mean(axis=0)
        return vec.tolist()

    # already a single vector (list of floats)
    if isinstance(raw, list) and all(isinstance(x, (int, float)) for x in raw):
        return [float(x) for x in raw]

    raise RuntimeError(
        f"Unexpected embedding response type: {type(raw)}. Raw truncated: {str(raw)[:200]}"
    )

# ---------------------------
# Chroma init (robust)
# ---------------------------
def init_chroma(persist_directory: str = CHROMA_DIR):
    """
    Try new-style Settings -> Client; fall back to older Client() patterns.
    Returns (client, collection)
    """
    # Try new-style Settings -> Client
    try:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        client = chromadb.Client(settings)
        try:
            collection = client.get_collection("documents")
        except Exception:
            collection = client.create_collection(name="documents")
        return client, collection
    except Exception as e_new:
        # Try older pattern
        try:
            client = chromadb.Client()
            try:
                collection = client.get_collection("documents")
            except Exception:
                collection = client.create_collection(name="documents", metadata={"created_by": "copilot"})
            return client, collection
        except Exception as e_old:
            raise RuntimeError(
                "Failed to initialize Chroma DB with both new and old client methods.\n\n"
                f"New-style error: {e_new}\nOld-style error: {e_old}\n\n"
                "Quick fixes: pin chromadb to an older compatible version: pip install 'chromadb==0.3.28'\n"
                "Or migrate your DB using chroma-migrate per Chroma docs."
            )

# Initialize globals
client, collection = init_chroma()
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------------------------
# Document helpers
# ---------------------------
def sanitize_metadata(metadata, model: str):
    """Return a guaranteed non-empty metadata dict."""
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": str(metadata)}
    if len(metadata) == 0:
        metadata.update({
            "source": "user",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model": model
        })
    else:
        metadata.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
        metadata.setdefault("model", model)
    return metadata

def add_document(text: str, metadata: dict = None, model: str = DEFAULT_MODEL):
    """
    Add a single document to Chroma with safe persistence.
    """
    meta = sanitize_metadata(metadata, model)
    emb = get_embedding(text, model=model)
    doc_id = str(uuid.uuid4())
    collection.add(ids=[doc_id], documents=[text], metadatas=[meta], embeddings=[emb])
    safe_persist(client)
    return doc_id

def search(query: str, top_k: int = 3, model: str = DEFAULT_MODEL):
    """
    Compute query embedding, run Chroma query and return normalized result dict.
    Always request only allowed include items (documents, metadatas, distances).
    Normalize to return a dict with top-level keys: ids, documents, metadatas, distances.
    """
    q_emb = get_embedding(query, model=model)
    try:
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]  # <-- do NOT include "ids" here
        )
    except Exception as e:
        raise RuntimeError(f"Chroma query failed: {e}")

    # Normalize nested return shapes (many chroma versions return nested lists)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return {"ids": ids, "documents": docs, "metadatas": metas, "distances": dists}


# ---------------------------
# RAG: Retrieval-Augmented Answering (Micro-step 3.1)
# ---------------------------
# Default generation model (text2text / instruction tuned)
DEFAULT_RAG_GEN_MODEL = "llama-3.1-8b-instant"

def build_context_and_sources(res, max_chars=3000):
    """
    Build a combined context string from Chroma results, but make image captions explicit.
    Returns (context_str, sources_list) where sources_list contains dicts with {id, meta, snippet}.
    """
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    pieces = []
    total = 0
    sources = []
    for did, doc, meta in zip(ids, docs, metas):
        # Normalize metadata
        m = meta if isinstance(meta, dict) else {"raw": meta}

        src_tag = m.get("source") or m.get("filename") or "unknown"
        is_image = (m.get("type") == "image") or (isinstance(doc, str) and str(doc).startswith("[image]"))

        # If image, prefer using caption in context (falls back to stored doc text)
        if is_image:
            caption = m.get("caption") or (doc if isinstance(doc, str) else "")
            snippet_text = f"Image caption: {caption}".strip()
        else:
            snippet_text = (doc if isinstance(doc, str) else str(doc)).strip()

        # sanitize & single-line
        snippet = snippet_text.replace("\n", " ").strip()
        # truncate per snippet to keep prompt small
        snippet = snippet[:800]

        prefix = f"[id:{did}][src:{src_tag}]"
        part = f"{prefix} {snippet}"
        part_len = len(part)
        if total + part_len > max_chars:
            break
        pieces.append(part)
        total += part_len

        sources.append({"id": did, "meta": m, "snippet": snippet})

    context = "\n\n".join(pieces)
    return context, sources


# Generation fallback list for REST if requested model is not hosted for generation
GENERATION_MODEL_FALLBACKS = [
    "google/flan-t5-xl",
    "tiiuae/falcon-7b-instruct",
]

# Requires: pip install groq and GROQ_API_KEY in your .env

from groq import Groq
import groq as groq_exceptions  # for typed exceptions

# create a module-level Groq client (reuse across calls)
_groq_client = None
def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("Missing GROQ_API_KEY in environment for Groq generation.")
        _groq_client = Groq(api_key=groq_api_key)
    return _groq_client


def generate_answer_from_context(question: str, context: str,
                                 gen_model: str = DEFAULT_RAG_GEN_MODEL,
                                 max_new_tokens: int = 256,
                                 temperature: float = 0.0) -> str:
    """
    Generation via Groq Chat Completions API (sync).
    - gen_model: model string accepted by Groq (e.g. 'compound-beta-mini', 'compound-beta', 'groq/llama3-70b-8192')
    - returns the generated string, or raises a RuntimeError with useful details on failure.
    """
    client = _get_groq_client()

    # Build a clear instruction-style prompt that asks the model to use the context
    system_msg = {
        "role": "system",
        "content": "You are a concise assistant that answers questions strictly using the provided CONTEXT. If the answer is not present, say 'Not found in context.' Be concise."
    }
    user_prompt = (
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "Answer using only the context above. If the answer is not present in the context, reply: Not found in context."
    )
    user_msg = {"role": "user", "content": user_prompt}

    # Groq client accepts 'model' and 'messages'
    # Map our params into typical Groq chat params. Groq client may support other options; this is minimal & compatible.
    try:
        response = client.chat.completions.create(
            model=gen_model,
            messages=[system_msg, user_msg],
            temperature=float(temperature),
            max_tokens=int(max_new_tokens),
        )
    except groq_exceptions.APIStatusError as e:
        # expose HTTP status and body to help debugging in the UI
        raise RuntimeError(f"Groq APIStatusError (status={e.status_code}): {getattr(e, 'response', str(e))}")
    except groq_exceptions.RateLimitError as e:
        raise RuntimeError(f"Groq RateLimitError: {e}")
    except Exception as e:
        # catch-all (network, auth, etc.)
        raise RuntimeError(f"Groq generation failed: {e}")

    # Parse response: the Groq client uses pydantic models; the normal text is at:
    # response.choices[0].message.content (per docs & examples)
    try:
        if not response or not hasattr(response, "choices") or len(response.choices) == 0:
            # fallback: try response.to_dict or str
            txt = getattr(response, "to_json", lambda: str(response))()
            return str(txt)
        choice0 = response.choices[0]
        # Some responses put message under .message, others might use .text; handle both
        if hasattr(choice0, "message") and getattr(choice0.message, "content", None) is not None:
            return choice0.message.content.strip()
        if hasattr(choice0, "text"):
            return choice0.text.strip()
        # fallback to string conversion
        return str(choice0).strip()
    except Exception as e:
        raise RuntimeError(f"Failed to parse Groq response: {e}\nRaw response: {response}")

# ---------------------------
# Provenance helpers (Micro-step 3.2)
# ---------------------------
def sentence_splitter(text: str):
    """
    Naive sentence splitting suitable for short generated answers.
    Keeps delimiters. Returns list of sentences (trimmed).
    """
    if not text or not text.strip():
        return []
    parts = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9])', text.strip())
    return [p.strip() for p in parts if p.strip()]

def find_support_for_sentence(sentence: str, top_k: int = 1, model: str = DEFAULT_MODEL) -> list:
    """
    Embed `sentence` and query the Chroma collection for nearest chunks.
    Returns list of result dicts (each with id, document, metadata, distance).
    """
    try:
        emb = get_embedding(sentence, model=model)
    except Exception as e:
        return [{"error": f"Embedding failed: {e}"}]

    try:
        # FIX: don't put "ids" in include — Chroma always returns them
        res = collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        return [{"error": f"Chroma query failed: {e}"}]

    # Normalize results
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for i, (did, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
        out.append({
            "id": did,
            "document": doc,
            "metadata": meta,
            "distance": float(dist)
        })
    return out


def show_provenance_for_answer(answer_text: str, support_top_k: int = 1, embed_model: str = DEFAULT_MODEL):
    """
    For each sentence in `answer_text`, find top-k supporting chunks and display them.
    """
    sentences = sentence_splitter(answer_text)
    if not sentences:
        st.info("No answer sentences to provide provenance for.")
        return

    st.subheader("Provenance (supporting chunks per answer sentence)")
    for i, sent in enumerate(sentences):
        with st.expander(f"Sentence {i+1}: \"{sent[:120]}{'...' if len(sent)>120 else ''}\"", expanded=(i==0)):
            st.write(sent)
            with st.spinner("Finding supporting chunks..."):
                supports = find_support_for_sentence(sent, top_k=support_top_k, model=embed_model)
            if not supports:
                st.write("No supporting chunks found.")
                continue
            for j, s in enumerate(supports):
                if "error" in s:
                    st.error(s["error"])
                    continue
                meta = s.get("metadata") or {}
                st.markdown(f"**Support {j+1} — id {s.get('id')} — distance {s.get('distance'):.4f}**")
                st.write(s.get("document")[:1500])
                # Show structured metadata including useful fields if present
                info = {}
                info["source"] = meta.get("source", meta.get("file", "unknown"))
                if "chunk_index" in meta:
                    info["chunk_index"] = meta["chunk_index"]
                if "char_start" in meta and "char_end" in meta:
                    info["char_range"] = (meta["char_start"], meta["char_end"])
                info["created_at"] = meta.get("created_at", "")
                st.json(info)
                st.markdown("---")

# ---------------------------
# File ingestion (upload -> extract)
# ---------------------------
def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from a PDF. Steps:
      1. Use pdfplumber.extract_text() for selectable text.
      2. If result is empty or too small, fall back to OCR:
         a) prefer pytesseract (requires Tesseract binary installed and on PATH),
         b) otherwise try easyocr if installed.
    Returns combined text for all pages.
    """
    text_pieces = []

    try:
        import pdfplumber
    except Exception as e:
        raise RuntimeError(f"pdfplumber not installed or import failed: {e}")

    try:
        with pdfplumber.open(path) as pdf:
            for i, p in enumerate(pdf.pages):
                txt = p.extract_text()
                if txt:
                    text_pieces.append(txt)
    except Exception as e:
        st.warning(f"pdfplumber extraction error (will attempt OCR): {e}")

    combined = "\n\n".join([t for t in text_pieces if t and t.strip()])
    if combined and len(combined) > 50:
        return combined

    st.info("No selectable text found — attempting OCR fallback (this may be slower).")

    def ocr_page_with_pytesseract(page):
        try:
            from PIL import Image
            import pytesseract
        except Exception as e:
            raise RuntimeError(f"pytesseract not available: {e}")

        pil_img = page.to_image(resolution=300).original
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        ocr_text = pytesseract.image_to_string(pil_img)
        return ocr_text

    used_ocr = False
    try:
        import pytesseract
        try:
            import subprocess
            subprocess.run(["tesseract", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tesseract_available = True
        except Exception:
            tesseract_available = False

        if tesseract_available:
            try:
                with pdfplumber.open(path) as pdf:
                    for p in pdf.pages:
                        try:
                            ocr_text = ocr_page_with_pytesseract(p)
                            if ocr_text and ocr_text.strip():
                                text_pieces.append(ocr_text)
                        except Exception as e:
                            st.warning(f"pytesseract OCR failed for a page: {e}")
                used_ocr = True
            except Exception as e:
                st.warning(f"Error running pytesseract OCR: {e}")
    except Exception:
        tesseract_available = False

    if not used_ocr:
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            try:
                with pdfplumber.open(path) as pdf:
                    for p in pdf.pages:
                        try:
                            pil_img = p.to_image(resolution=300).original
                            import numpy as _np
                            img_arr = _np.array(pil_img)
                            result = reader.readtext(img_arr, detail=0, paragraph=True)
                            page_text = "\n".join(result)
                            if page_text and page_text.strip():
                                text_pieces.append(page_text)
                        except Exception as e:
                            st.warning(f"easyocr failed for a page: {e}")
                used_ocr = True
            except Exception:
                try:
                    import fitz
                    doc = fitz.open(path)
                    for page in doc:
                        pix = page.get_pixmap(dpi=300)
                        img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        if img_arr.shape[2] == 4:
                            img_arr = img_arr[:, :, :3]
                        result = reader.readtext(img_arr, detail=0, paragraph=True)
                        page_text = "\n".join(result)
                        if page_text and page_text.strip():
                            text_pieces.append(page_text)
                    used_ocr = True
                except Exception as e:
                    st.warning(f"easyocr fallback render failed: {e}")
        except Exception as ee:
            st.warning(f"easyocr not available or failed to initialize: {ee}")

    combined_ocr = "\n\n".join([t for t in text_pieces if t and t.strip()])
    if combined_ocr and len(combined_ocr) > 10:
        return combined_ocr

    return ""

def save_uploaded_file(uploaded_file) -> str:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    save_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# ---------------------------
# Chunking utility
# ---------------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Character-based chunking with paragraph-preferred strategy.
    Returns list of tuples: (chunk_text, (start_char, end_char))
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    current = ""
    current_start = 0
    pointer = 0
    for p in paragraphs:
        if not current:
            current_start = pointer
        if len(current) + len(p) + 2 <= chunk_size:
            if current:
                current += "\n\n" + p
            else:
                current = p
            pointer += len(p) + 2
        else:
            if current:
                start = current_start
                end = start + len(current)
                chunks.append((current, (start, end)))
                current = p
                current_start = pointer
                pointer += len(p) + 2
            else:
                break

    if current:
        start = current_start
        end = start + len(current)
        chunks.append((current, (start, end)))

    if not chunks or any((end - start) > chunk_size for (_, (start, end)) in chunks):
        chunks = []
        text_len = len(text)
        if text_len == 0:
            return []
        step = chunk_size - overlap
        n_chunks = max(1, math.ceil((text_len - overlap) / step))
        for i in range(n_chunks):
            start = i * step
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            chunks.append((chunk, (start, end)))
            if end == text_len:
                break
    return chunks

# ---------------------------
# Image ingestion: CLIP embeddings + store in Chroma
# ---------------------------
from PIL import Image
import io
import mimetypes

# ---------------------------
# Image -> Gemini caption -> embed (store caption embedding in Chroma)
# ---------------------------
from PIL import Image
import io
import base64
import traceback

# Try to support both new and old Google GenAI SDKs
def _get_gemini_caption(image_bytes: bytes, instruction: str = "Describe this image in detail, include any readable text, dates, names, and other useful metadata."):
    """
    Return a short/medium caption string for an image using Gemini API.
    Tries new google-genai (preferred) then falls back to google.generativeai legacy client.
    Raises RuntimeError with diagnostics on failure.
    """
    # load PIL image (Gemini SDK examples accept PIL.Image)
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"PIL open failed: {e}")

    errors = []

    # Try new SDK: `google-genai` / genai (preferred modern API)
    try:
        try:
            # new SDK style
            from google import genai
            # client will pick credentials from environment; ensure GEMINI_API_KEY in .env
            client = genai.Client()
            # choose a multimodal Gemini model (adjust if unavailable in your account)
            model_id = "gemini-1.5-flash"
            # generate content using a PIL.Image directly in contents
            resp = client.models.generate_content(
                model=model_id,
                contents=[
                    instruction,
                    pil
                ],
                # optional: set temperature/other params via kwargs if SDK supports them
            )
            text = getattr(resp, "text", None)
            if text and text.strip():
                return text.strip()
            # fallthrough if empty
            errors.append(("genai_empty", f"empty text from genai response", str(resp)))
        except Exception as e:
            errors.append(("genai_exception", str(e), traceback.format_exc()))
    except Exception:
        # module not installed or import failed — ignore and try legacy
        errors.append(("genai_import", "google.genai not available", ""))

    # Fallback: older google.generativeai package (legacy)
    try:
        import google.generativeai as gga
        # configure if necessary (the user should have GEMINI_API_KEY in .env and library's configure will pick it up)
        try:
            gga.configure(api_key=os.getenv("GEMINI_API_KEY"))
        except Exception:
            # some versions use genai.GenerativeModel directly without configure
            pass
        try:
            model = gga.GenerativeModel("gemini-1.5-flash")
            res = model.generate_content([instruction, pil])
            text = getattr(res, "text", None)
            if text and text.strip():
                return text.strip()
            errors.append(("generativeai_empty", "empty text from generativeai response", str(res)))
        except Exception as e:
            errors.append(("generativeai_exception", str(e), traceback.format_exc()))
    except Exception as e:
        errors.append(("generativeai_import", str(e), ""))

    # If we reach here, both attempts failed — raise a helpful error with traces
    msg_lines = ["Gemini caption failed. Attempts:"]
    for tag, summary, detail in errors:
        msg_lines.append(f"- {tag}: {summary}")
        if detail:
            msg_lines.append(f"    detail: {str(detail)[:800].replace(chr(10), ' ')}")
    raise RuntimeError("\n".join(msg_lines))




# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Multimodal Research Copilot", layout="wide", page_icon="🔎")

# Top header
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="font-size:36px">🔎</div>
    <div style="line-height:1">
      <h1 style="margin:0;padding:0">Multimodal Research Copilot</h1>
      <div style="color:gray;margin-top:4px">Elegant UI • Gemini captioning • HF embeddings • Chroma DB</div>
    </div>
  </div>
  <div style="text-align:right;color:gray;font-size:14px">
    <div>UI edition</div>
    <div style="margin-top:6px">Status: <span style="color:green;font-weight:700">Ready</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar content: models, keys, quick actions, docs list
with st.sidebar:
    st.header("⚙️ Config & Keys")
    st.markdown("**Models used:**")
    st.markdown(f"- Text embedding: `{DEFAULT_MODEL}`  \n- RAG generator: `{DEFAULT_RAG_GEN_MODEL}`  \n- Image caption: `gemini-1.5-flash`")
    st.markdown("---")
    st.subheader("Required API keys")
    st.write("Set in `.env` or environment:")
    st.code("HUGGINGFACE_HUB_TOKEN=\nGROQ_API_KEY=\nGEMINI_API_KEY=\nGEMINI_API_URL=", language="bash")
    st.markdown("---")
    st.subheader("Quick settings")
    model_name = st.text_input("HF model (feature-extraction)", value=st.session_state.get("model_name", DEFAULT_MODEL))
    st.session_state["model_name"] = model_name
    top_k = st.slider("Search: top k", 1, 10, int(st.session_state.get("top_k", 3)))
    st.session_state["top_k"] = top_k
    st.markdown("---")
    with st.expander("📚 Documents (quick list)", expanded=False):
        try:
            payload = collection.get(include=["documents", "metadatas"])
            ids = payload.get("ids", [])
            metas = payload.get("metadatas", [])
            if not ids:
                st.info("No documents yet.")
            else:
                for i, (did, meta) in enumerate(zip(ids, metas)):
                    src = meta.get("source", meta.get("filename", "unknown")) if isinstance(meta, dict) else "unknown"
                    st.write(f"{i+1}. id={did[:8]} • {src}")
        except Exception as e:
            st.warning(f"Could not fetch docs: {e}")
    with st.expander("🗄️ DB actions", expanded=False):
        if st.button("Show DB stats", use_container_width=True):
            try:
                try:
                    c = collection.count()
                except Exception:
                    payload = collection.get(include=["documents", "metadatas"])
                    c = len(payload.get("ids", []))
                st.success(f"Collection contains ~{c} documents (approx).")
            except Exception as e:
                st.error(f"DB stats failed: {e}")
        if st.button("Clear DB", use_container_width=True):
            try:
                try:
                    client.delete_collection("documents")
                except Exception:
                    try:
                        collection.delete()
                    except Exception:
                        pass
                new_client, new_collection = init_chroma()
                globals()['client'] = new_client
                globals()['collection'] = new_collection
                st.success("DB cleared & recreated.")
            except Exception as e:
                st.error(f"Clear failed: {e}")

# Main tabs
tabs = st.tabs(["📁 Documents", "📷 Images", "🎙️ Voice", "🔍 Search", "❓ Ask (RAG)", "📊 Database"])

# Documents tab
with tabs[0]:
    st.subheader("📁 Document Ingestion & Chunking")
    c1, c2 = st.columns([2,1])
    with c1:
        with st.expander("Upload & extract (PDF / TXT)", expanded=True):
            uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf","txt"], key="ui_upload_file")
            if uploaded_file is not None:
                save_path = save_uploaded_file(uploaded_file)
                extracted_text = ""
                if uploaded_file.name.lower().endswith(".pdf") or uploaded_file.type == "application/pdf":
                    try:
                        extracted_text = extract_text_from_pdf(save_path)
                    except Exception as e:
                        st.error(f"PDF extraction failed: {e}")
                        extracted_text = ""
                else:
                    try:
                        raw = uploaded_file.getvalue()
                        extracted_text = raw.decode("utf-8", errors="ignore")
                    except Exception as e:
                        st.error(f"Text read failed: {e}")
                        extracted_text = ""
                if extracted_text:
                    st.success(f"Extracted {len(extracted_text)} characters from {uploaded_file.name}")
                    st.text_area("Preview extracted text", value=extracted_text[:5000], height=220)
                    if st.button("Copy to Add document box"):
                        st.session_state["doc_text"] = extracted_text
                        st.success("Copied to Add document box.")
    with c2:
        with st.expander("Add document manually", expanded=True):
            with st.form("add_doc_ui"):
                doc_text = st.text_area("Document text", value=st.session_state.get("doc_text",""), height=200, key="ui_doc_text")
                metadata_str = st.text_input("Metadata (JSON) — optional", value="{}", key="ui_metadata_str")
                submitted = st.form_submit_button("Add to DB")
                if submitted:
                    try:
                        metadata = json.loads(metadata_str)
                        if not isinstance(metadata, dict):
                            metadata = {"raw_metadata": metadata}
                    except Exception:
                        metadata = {"raw_metadata": metadata_str or "user_input"}
                    if not doc_text.strip():
                        st.error("Empty doc not allowed")
                    else:
                        with st.spinner("Embedding + storing..."):
                            try:
                                doc_id = add_document(doc_text, metadata, model_name)
                                st.success(f"✅ Added doc id={doc_id[:8]}")
                                st.session_state["doc_text"] = ""
                            except Exception as e:
                                st.error(f"Failed: {e}")
    with st.expander("Chunking & Ingest options", expanded=False):
        chunk_size = st.number_input("Chunk size (characters)", min_value=200, max_value=8000, value=1000, step=100, key="ui_chunk_size")
        overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=max(0,int(chunk_size)-1), value=200, step=50, key="ui_overlap")
        ingest_confirm = st.button("Ingest text as chunks")
        if ingest_confirm:
            txt = st.session_state.get("doc_text","") or ""
            if not txt.strip():
                st.error("No text available to chunk.")
            else:
                chunks = chunk_text(txt, chunk_size=int(chunk_size), overlap=int(overlap))
                st.info(f"Splitting into {len(chunks)} chunks")
                batch_ids, batch_docs, batch_metas, batch_embs = [],[],[],[]
                BATCH_SIZE = 8
                added = 0
                with st.spinner("Embedding chunks and adding to Chroma..."):
                    for idx, (chunk, (start,end)) in enumerate(chunks):
                        chunk_meta = {"source":"pasted_text","chunk_index":idx,"char_start":int(start),"char_end":int(end),"created_at":datetime.utcnow().isoformat()+"Z","model":model_name}
                        try:
                            emb = get_embedding(chunk, model=model_name)
                        except Exception as e:
                            st.error(f"Embedding failed for chunk {idx}: {e}")
                            emb = None
                        if emb is None:
                            continue
                        cid = str(uuid.uuid4())
                        batch_ids.append(cid); batch_docs.append(chunk); batch_metas.append(chunk_meta); batch_embs.append(emb)
                        if len(batch_ids) >= BATCH_SIZE:
                            try:
                                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_embs)
                                safe_persist(client)
                                added += len(batch_ids)
                            except Exception as e:
                                st.error(f"Failed to add chunk batch: {e}")
                            batch_ids, batch_docs, batch_metas, batch_embs = [],[],[],[]
                    if batch_ids:
                        try:
                            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_embs)
                            safe_persist(client)
                            added += len(batch_ids)
                        except Exception as e:
                            st.error(f"Failed to add final chunk batch: {e}")
                st.success(f"Finished ingesting {added} chunks into Chroma.")

# Images tab
with tabs[1]:
    st.subheader("📷 Image Ingestion (Gemini caption → embed)")
    uploaded_images = st.file_uploader("Upload images (png/jpg/webp)", type=["png","jpg","jpeg","webp"], accept_multiple_files=True, key="ui_img_uploader")
    ingest_images_btn = st.button("Ingest uploaded image(s) as captioned docs")
    if uploaded_images and ingest_images_btn:
        added = 0; BATCH_SIZE_IMG = 8
        batch_ids, batch_docs, batch_metas, batch_embs = [],[],[],[]
        for f in uploaded_images:
            try:
                file_bytes = f.getbuffer().tobytes()
            except Exception as e:
                st.error(f"Failed reading {f.name}: {e}"); continue
            with st.spinner(f"Generating caption for {f.name}..."):
                try:
                    caption = _get_gemini_caption(file_bytes)
                except Exception as e:
                    st.error(f"Captioning failed for {f.name}: {e}"); caption = None
            if not caption:
                st.warning(f"No caption generated for {f.name}; skipping ingestion."); continue
            try:
                emb = get_embedding(caption, model=model_name)
            except Exception as e:
                st.error(f"Embedding caption failed for {f.name}: {e}"); continue
            try:
                os.makedirs(UPLOADS_DIR, exist_ok=True)
                save_path = os.path.join(UPLOADS_DIR, f.name)
                with open(save_path,"wb") as fh: fh.write(file_bytes)
            except Exception as e:
                st.warning(f"Could not save {f.name}: {e}"); save_path = f.name
            meta = {"type":"image","filename":f.name,"path":save_path,"caption":caption,"created_at":datetime.utcnow().isoformat()+"Z","model_caption":"gemini-1.5-flash","model_embed":model_name,"source":"upload"}
            cid = str(uuid.uuid4())
            batch_ids.append(cid); batch_docs.append(caption); batch_metas.append(meta); batch_embs.append(emb)
            if len(batch_ids) >= BATCH_SIZE_IMG:
                try:
                    collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_embs)
                    safe_persist(client); added += len(batch_ids)
                except Exception as e:
                    st.error(f"Failed to add image-caption batch: {e}")
                batch_ids, batch_docs, batch_metas, batch_embs = [],[],[],[]
        if batch_ids:
            try:
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_embs); safe_persist(client); added += len(batch_ids)
            except Exception as e:
                st.error(f"Failed to add final image-caption batch: {e}")
        st.success(f"Finished ingesting {added} captioned images into Chroma.")

# Voice tab
with tabs[2]:
    st.subheader("🎙️ Voice Search (Transcribe → Search)")
    audio_file = st.file_uploader("Upload audio file for voice search", type=["wav","mp3","m4a","ogg"], accept_multiple_files=False, key="ui_voice_upload")
    transcribe_btn = st.button("Transcribe & Search (voice)")
    if transcribe_btn:
        if audio_file is None:
            st.error("Please upload an audio file first.")
        else:
            try:
                audio_bytes = audio_file.getbuffer().tobytes()
            except Exception as e:
                st.error(f"Failed to read audio file: {e}"); audio_bytes = None
            if audio_bytes:
                with st.spinner("Transcribing audio (Groq/HF fallback)..."):
                    try:
                        transcription = transcribe_audio_via_groq(audio_bytes, filename=audio_file.name)
                        st.success("Transcription completed.")
                        st.text_area("Transcribed text (editable)", value=transcription, height=140, key="ui_voice_transcript")
                        st.info("Running search with the transcribed text...")
                        search_res = search(transcription, top_k=int(st.session_state.get("top_k",3)), model=model_name)
                        ids = search_res.get("ids", [[]])[0]; docs = search_res.get("documents", [[]])[0]; metas = search_res.get("metadatas", [[]])[0]; dists = search_res.get("distances", [[]])[0]
                        if not ids:
                            st.info("No results for the transcribed query.")
                        else:
                            st.subheader("Search results for voice query")
                            for i, (did, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
                                st.markdown(f"**Result {i+1} — id {did} — distance {dist:.4f}**")
                                st.write(doc if isinstance(doc, str) else str(doc)); st.json(meta); st.markdown("---")
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")

# Search tab
with tabs[3]:
    st.subheader("🔍 Search the vector DB (cross-modal)")
    query = st.text_input("Search query (text)", key="ui_search_query_input")
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        modality = st.selectbox("Show", ["All","Text only","Images only"], index=0, key="ui_search_modality")
    with col2:
        sort_order = st.selectbox("Sort by distance", ["Closest first","Farthest first"], index=0, key="ui_search_sort")
    with col3:
        threshold = st.number_input("Max distance", min_value=0.0, max_value=5.0, value=2.0, step=0.1, key="ui_search_threshold")
    search_btn = st.button("Search", key="ui_search_go")
    if search_btn:
        if not (query and query.strip()):
            st.error("Please type a query")
        else:
            with st.spinner("Searching..."):
                try:
                    res = collection.query(query_texts=[query], n_results=int(st.session_state.get("top_k",3)), include=["documents","metadatas","distances"])
                    ids = res.get("ids", [[]])[0]; docs = res.get("documents", [[]])[0]; metas = res.get("metadatas", [[]])[0]; dists = res.get("distances", [[]])[0]
                    results = list(zip(ids, docs, metas, dists)); results = [r for r in results if (r[3] is None) or (float(r[3]) <= float(threshold))]; reverse_sort = (sort_order=="Farthest first"); results.sort(key=lambda r: float(r[3]) if r[3] is not None else float("inf"), reverse=reverse_sort)
                    normalized_rows = [_normalize_result_row(did, doc, meta, dist) for (did, doc, meta, dist) in results]
                    st.session_state["last_search_results"] = normalized_rows
                    if not results:
                        st.warning("No results within the chosen threshold.")
                    else:
                        for i, (did, doc, meta, dist) in enumerate(results):
                            if not isinstance(meta, dict):
                                try:
                                    meta = json.loads(meta)
                                except Exception:
                                    meta = {"raw": str(meta)}
                            is_image = (meta.get("type")=="image") or (isinstance(doc, str) and str(doc).startswith("[image]"))
                            if modality=="Text only" and is_image:
                                continue
                            if modality=="Images only" and not is_image:
                                continue
                            st.markdown(f"**Result {i+1} — id {did} — distance {float(dist) if dist is not None else float('nan'):.4f}**")
                            if is_image:
                                caption = meta.get("caption") or (doc if isinstance(doc, str) else "<no caption>"); st.write(caption)
                                img_path = meta.get("path") or meta.get("filename")
                                if img_path:
                                    candidate = img_path
                                    if not os.path.exists(candidate):
                                        candidate = os.path.join(UPLOADS_DIR, img_path)
                                    try:
                                        if os.path.exists(candidate):
                                            st.image(candidate, caption=meta.get("filename",""), use_column_width=False, width=380)
                                            try:
                                                with open(candidate, "rb") as fh:
                                                    data_bytes = fh.read()
                                                st.download_button(label="Download file", data=data_bytes, file_name=os.path.basename(candidate), mime=meta.get("mime","application/octet-stream"))
                                            except Exception:
                                                st.warning("Download not available for this file.")
                                        else:
                                            st.info(f"Image file not found locally: {img_path}")
                                    except Exception as e:
                                        st.warning(f"Could not display image preview: {e}")
                                else:
                                    st.info("No local image path available in metadata.")
                                st.json(meta); st.markdown("---")
                            else:
                                st.write(doc); st.json(meta); st.markdown("---")
                except Exception as e:
                    st.error(f"Search failed: {e}")
    if st.session_state.get("last_search_results"):
        rows = st.session_state["last_search_results"]
        try:
            import pandas as _pd
            df = _pd.DataFrame(rows); csv_bytes = df.to_csv(index=False).encode("utf-8")
        except Exception:
            import io as _io, csv as _csv; buf = _io.StringIO()
            if rows:
                keys = list(rows[0].keys()); writer = _csv.DictWriter(buf, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
            csv_bytes = buf.getvalue().encode("utf-8")
        json_bytes = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
        st.markdown("**Export last search results:**")
        colc1, colc2 = st.columns([1,1])
        with colc1:
            st.download_button(label="Download CSV", data=csv_bytes, file_name="search_results.csv", mime="text/csv")
        with colc2:
            st.download_button(label="Download JSON", data=json_bytes, file_name="search_results.json", mime="application/json")

# RAG tab
with tabs[4]:
    st.subheader("❓ Ask (RAG) — Answer using retrieved chunks")
    col_a, col_b = st.columns([3,1])
    with col_a:
        rag_gen_model = st.text_input("Generation model (instruction tuned)", value=DEFAULT_RAG_GEN_MODEL, key="ui_rag_model")
    with col_b:
        rag_max_tokens = st.number_input("Gen max tokens", min_value=32, max_value=1024, value=256, step=32, key="ui_rag_max_tokens")
    try:
        _top_k = int(st.session_state.get("top_k",3))
    except Exception:
        _top_k = 3
    qa_query = st.text_input("Ask a question (RAG)", key="ui_qa_query")
    if st.button("Get answer (RAG)", key="ui_rag_go"):
        if not qa_query.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Retrieving top chunks and generating answer..."):
                try:
                    res = collection.query(query_texts=[qa_query], n_results=_top_k, include=["documents","metadatas","distances"])
                    context, context_sources = build_context_and_sources(res, max_chars=3000)
                    if not context:
                        st.warning("No context retrieved from DB. Add documents or ingest chunks first.")
                        context = ""
                    st.subheader("Retrieved chunks (top results)")
                    ids = res.get("ids", [[]])[0]; docs = res.get("documents", [[]])[0]; metas = res.get("metadatas", [[]])[0]; dists = res.get("distances", [[]])[0]
                    for i, (did, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
                        st.markdown(f"**Result {i+1} — id {did} — distance {dist:.4f}**"); st.write(doc[:1000]); st.json(meta); st.markdown("---")
                    answer = generate_answer_from_context(qa_query, context, gen_model=rag_gen_model, max_new_tokens=rag_max_tokens)
                    if context_sources:
                        citations = []
                        for s in context_sources:
                            src = s["meta"].get("filename") or s["meta"].get("source") or s["id"]
                            chunk = s["meta"].get("chunk_index")
                            if chunk is not None:
                                citations.append(f"[src:{src}, chunk:{chunk}]")
                            else:
                                citations.append(f"[src:{src}]")
                        if citations:
                            answer = f"{answer}\n\nSources: {'; '.join(citations)}"
                    if not answer:
                        st.info("Generation returned empty response. Try a different generation model or increase max tokens.")
                    else:
                        st.subheader("Answer (RAG)"); st.write(answer); show_provenance_for_answer(answer, support_top_k=1, embed_model=DEFAULT_MODEL)
                except Exception as e:
                    st.error(f"RAG failed: {e}")

# Database tab
with tabs[5]:
    st.subheader("📊 Database Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("List all docs", use_container_width=True):
            try:
                payload = collection.get(include=["documents","metadatas"])
                ids = payload.get("ids", []); docs = payload.get("documents", []); metas = payload.get("metadatas", [])
                if not ids:
                    st.info("No documents in the collection yet.")
                else:
                    for did, doc, meta in zip(ids, docs, metas):
                        st.write(f"id={did}"); st.write(doc); st.json(meta); st.markdown("---")
            except Exception as e:
                st.error(f"List failed: {e}")
    with col2:
        st.info("Use the sidebar for quick DB stats or clearing the DB.")

# Footer tips
st.markdown("---")
st.markdown("**Tips:**  \n• Restart Streamlit after changing environment variables.  \n• For best OCR results install `tesseract` and `pytesseract`; optionally install `easyocr` and `opencv-python`.")