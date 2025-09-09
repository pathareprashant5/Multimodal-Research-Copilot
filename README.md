# 🔎 Multimodal Research Copilot

An elegant **Streamlit-based research assistant** that supports **documents, images, and voice queries** with advanced search and RAG (Retrieval-Augmented Generation).  
It integrates Hugging Face embeddings, Gemini image captioning, Groq text generation, and ChromaDB for efficient multimodal search.

---

## ✨ Features

- 📁 **Document ingestion**  
  - Upload PDFs/TXT, extract text (with OCR fallback for scanned PDFs)  
  - Chunk & embed into ChromaDB  
  - Add custom text manually with metadata  

- 📷 **Image ingestion**  
  - Upload images (PNG/JPG/WEBP)  
  - Auto-captioned using Gemini API (`gemini-1.5-flash`)  
  - Captions embedded and stored in vector DB  

- 🎙️ **Voice search**  
  - Upload audio (WAV/MP3/M4A/OGG)  
  - Transcribes speech → converts into a searchable query  

- 🔍 **Cross-modal search**  
  - Search across text & image embeddings  
  - Filter by modality, distance, and export results (CSV/JSON)  

- ❓ **RAG-based Q&A**  
  - Retrieves top chunks from DB  
  - Generates concise answers using Groq LLMs  
  - Shows provenance (supporting context for each sentence)  

- 📊 **Database management**  
  - View stored docs, metadata, stats  
  - Clear and rebuild DB  

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/pathareprashant5/multimodal-research-copilot.git
cd multimodal-research-copilot
pip install -r requirements.txt
