# RAG Document Q&A Bot

A Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and ask questions about it in plain English — answers are grounded in the actual document content, with source chunks shown for verification.

Built as a hands-on project to learn RAG architecture from the ground up — not a tutorial copy-paste, every component (chunking, embedding, retrieval, generation) was built and debugged individually.

## Demo


**Example Q&A** (tested on SRKR's B.Tech R23 Academic Regulations):
- **Q:** "How many academic years does a student have to complete their B.Tech degree?"
- **A:** "Based on the provided context, a student has eight academic years from the year of their admission to fulfill all academic requirements for the B.Tech degree..."

## Tech Stack

- **Orchestration:** LangChain
- **Embeddings & Generation:** Google Gemini API (`gemini-embedding-001`, `gemini-3.6-flash`)
- **Vector Store:** FAISS (local)
- **PDF Parsing:** PyPDF
- **Frontend:** Streamlit
- **Environment:** Python 3.10

## How It Works

1. **Ingestion** — user uploads a PDF; it's parsed page by page.
2. **Chunking** — document is split into ~500-character chunks with 50-character overlap, so context isn't lost at chunk boundaries.
3. **Embedding** — each chunk is converted into a vector via Gemini's embedding model, batched (40 chunks/batch with delay) to respect free-tier rate limits.
4. **Storage** — vectors are stored in a local FAISS index.
5. **Retrieval** — user's query is embedded and compared against all chunk vectors; top-k most similar chunks are retrieved (k adapts: 3 for specific questions, 10 for broad "summarize" requests).
6. **Generation** — retrieved chunks are inserted into a prompt alongside the query, and Gemini generates an answer grounded in that context, with source chunks shown for transparency.

## Setup

```bash
git clone https://github.com/Dharani-new/rag-college-bot.git
cd rag-college-bot
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

## Engineering Challenges Solved

- **Rate limiting:** Google's free-tier embedding API caps at 100 requests/minute. Solved with batched embedding (40 chunks/batch) and delays between batches, with progress saved after each batch to avoid losing work on failure.
- **Model deprecation:** handled multiple mid-development API/model name changes (embedding model, chat model) by reading error messages and updating to current model identifiers.
- **Query-type-aware retrieval:** broad "summarize" queries need more context than specific fact-lookup queries — implemented adaptive `k` and a separate summarization prompt to handle this distinction.

## Future Improvements

This is a working first version, not a finished production system. Concrete next steps I'm aware of and plan to explore:

- **Re-ranking** — add a second-stage re-ranker to reorder retrieved chunks by relevance before passing to the LLM, rather than relying purely on raw similarity search.
- **Retrieval evaluation** — build a formal way to measure retrieval quality (e.g. precision@k against a test set of Q&A pairs) instead of manual spot-checking.
- **Confidence-aware regeneration** — detect low-confidence answers and automatically retry with adjusted retrieval parameters.
- **Context summarization** — summarize retrieved chunks before feeding to the LLM for very broad queries, reducing noise in the context window.
- **Cloud vector DB** — migrate from local FAISS to a hosted vector database (Pinecone or Qdrant) for scalability beyond a single machine.

## What I Learned

Built this to understand RAG at the implementation level after studying the underlying theory (tokenization, embeddings, attention, transformer architecture). Debugging real API rate limits and model deprecations taught me more about production considerations than any tutorial would have.
