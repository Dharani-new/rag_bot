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
2. **Chunking** — pages are merged into one continuous string before splitting (not split page-by-page), so chunk overlap works correctly even across page boundaries. Split into ~500-character chunks with 150-character overlap. Each chunk retains metadata on which page(s) it spans.
3. **Embedding** — each chunk is converted into a vector via Gemini's embedding model, batched (40 chunks/batch with delay) to respect free-tier rate limits.
4. **Storage** — vectors are stored in a local FAISS index.
5. **Retrieval** — user's query is embedded and compared against all chunk vectors. A wide first pass (top-20) is retrieved cheaply via bi-encoder similarity, then a cross-encoder re-ranker (`ms-marco-MiniLM-L6-v2`) re-scores each (query, chunk) pair directly for higher precision. Final top-k (3 for specific questions, 10 for broad "summarize" requests) is kept after re-ranking.
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

## Debugging a Silent Chunking Bug

While building a retrieval evaluation set, I noticed something odd inspecting raw chunk boundaries: consecutive chunks showed **zero character overlap** despite `chunk_overlap=50` being set. Investigation traced this to two compounding issues:

1. **Overlap value too small relative to the text's natural split units.** `RecursiveCharacterTextSplitter` carries overlap back along natural boundaries (line/sentence), not raw characters. With overlap set to 50 and typical line lengths of 90–140 characters in this PDF, the splitter had no room to carry back a full unit — resulting in effectively zero overlap everywhere. Fixed by raising overlap to 150.

2. **Page-independent splitting.** `PyPDFLoader` returns one `Document` per page, and `split_documents()` processes each page separately — meaning the splitter has zero visibility across page boundaries. This caused words to be silently sliced in half at page transitions (e.g. "conditions" → "con-" / "ditions") with no overlap stitching them back together, regardless of the overlap value.

**Fix:** merge all pages into a single string before splitting (preserving page metadata via character-offset tracking), so overlap works correctly across what used to be page boundaries. Verified with controlled before/after tests isolating each root cause independently.

## Retrieval Evaluation

Built a 15-question ground-truth eval set (chunk-ID level, not just answer-level) against a technical PDF, covering direct lookups, multi-location synthesis questions, and deliberately out-of-scope questions to test hallucination resistance.

**Results (k=5):**
- Hit-rate@5: 0.91
- Recall@5: 0.76
- Precision@5: 0.27 (expected — ground truth often has 1-2 correct chunks per question against k=5 retrieved, naturally diluting precision; recall-protective wide retrieval is a deliberate tradeoff)

One ambiguous case surfaced during eval: a "what are the four moment terms" question retrieved a legitimate alternate explanation (the paper's formal moment-definition section) that wasn't in the original ground truth — a useful reminder that ground truth for multi-location answers needs care, not just keyword matching.

## Known Limitations

- **Compound questions dilute retrieval.** A single query asking multiple unrelated things (e.g. "what's this about, who's the author, what metrics were used") gets embedded as one averaged vector, which can underperform on any single sub-question compared to asking it in isolation. Query decomposition (splitting into sub-queries) would address this — not yet implemented.
- **Summary-mode keyword trigger is imprecise.** Broad-summarization detection currently matches loosely on phrases like "what is this," which can misfire on compound factual questions that happen to contain that phrase, causing the summarization prompt (which deliberately omits fine detail) to fire when a detailed answer was actually wanted.

## Future Improvements

- **Query decomposition** — break compound multi-part questions into sub-queries, retrieve for each separately, then synthesize.
- **Confidence-aware regeneration** — detect low-confidence answers and automatically retry with adjusted retrieval parameters.
- **Cloud vector DB** — migrate from local FAISS to a hosted vector database (Pinecone or Qdrant) for scalability beyond a single machine.

## What I Learned

Built this to understand RAG at the implementation level after studying the underlying theory (tokenization, embeddings, attention, transformer architecture). Debugging real API rate limits and model deprecations taught me more about production considerations than any tutorial would have.
