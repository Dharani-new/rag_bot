from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
from openai import embeddings

# Load your API key from .env
load_dotenv()

# Step 1: Load the PDF (same as before)
loader = PyPDFLoader("regulations.pdf")
documents = loader.load()

# Step 2: Chunk it (same as before)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Step 3: Set up the embedding model
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

    vector_store.save_local("faiss_index")  # save after every batch, not just at the end
    print(f"Saved progress after batch {i//batch_size + 1}")

    if i + batch_size < len(chunks):
        print("Waiting 65 seconds to stay under the free-tier rate limit...")
        time.sleep(65)

print("All done! Vector store fully saved to 'faiss_index' folder.")

