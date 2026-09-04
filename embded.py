from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import time
from dotenv import load_dotenv

load_dotenv()

# ---- Step 1: Load PDF ----
loader = PyPDFLoader("iee_Q.pdf")   # <-- change this filename if you want a different PDF
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# ---- Step 2: Merge pages into one string, tracking page ranges (today's fix) ----
full_text = ""
page_boundaries = []
for doc in documents:
    start = len(full_text)
    full_text += doc.page_content + "\n"
    end = len(full_text)
    page_boundaries.append((start, end, doc.metadata.get("page")))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
raw_chunks = text_splitter.split_text(full_text)

# ---- Step 3: Rebuild chunks as Documents, with chunk_id + pages in metadata ----
chunks = []
search_pos = 0
for idx, chunk_text in enumerate(raw_chunks):
    chunk_start = full_text.find(chunk_text, max(0, search_pos - 50))
    chunk_end = chunk_start + len(chunk_text)
    search_pos = chunk_start + 1

    pages_touched = sorted({
        p for (s, e, p) in page_boundaries
        if chunk_start < e and chunk_end > s
    })

    chunks.append(Document(
        page_content=chunk_text,
        metadata={"pages": pages_touched, "chunk_id": idx + 1}
    ))

print(f"Split into {len(chunks)} chunks")

# ---- Step 4: Embed and save to FAISS ----
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="retrieval_document"
)

batch_size = 40
vector_store = None

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    print(f"Embedding batch {i//batch_size + 1} of {(len(chunks)//batch_size)+1}...")

    if vector_store is None:
        vector_store = FAISS.from_documents(batch, embeddings)
    else:
        vector_store.add_documents(batch)

    vector_store.save_local("faiss_index")
    print(f"Saved progress after batch {i//batch_size + 1}")

    if i + batch_size < len(chunks):
        print("Waiting 65 seconds to stay under free-tier rate limit...")
        time.sleep(65)

print("Done! faiss_index rebuilt with correct chunking + chunk IDs.")