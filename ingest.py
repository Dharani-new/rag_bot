from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

loader = PyPDFLoader("iee_Q.pdf")
documents = loader.load()

# Merge pages into one string, tracking which char range came from which page
full_text = ""
page_boundaries = []
for doc in documents:
    start = len(full_text)
    full_text += doc.page_content + "\n"
    end = len(full_text)
    page_boundaries.append((start, end, doc.metadata.get("page")))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
raw_chunks = text_splitter.split_text(full_text)


chunks = []
search_pos = 0
for chunk_text in raw_chunks:
    chunk_start = full_text.find(chunk_text, max(0, search_pos - 50))
    chunk_end = chunk_start + len(chunk_text)
    search_pos = chunk_start + 1

    pages_touched = sorted({
        p for (s, e, p) in page_boundaries
        if chunk_start < e and chunk_end > s
    })

    chunks.append(Document(page_content=chunk_text, metadata={"pages": pages_touched, "chunk_id": len(chunks) + 1}))


#print the contents of each chunk along with the pages it came from
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}:")
#     print(f"Pages: {chunk.metadata['pages']}")
#     print(f"Content: {chunk.page_content}...", "--"*10)  


def find_chunks_containing(phrase, chunks):
    return [i+1 for i, c in enumerate(chunks) if phrase.lower() in c.page_content.lower()]  # +1 for 1-indexed to match your table

# print("Q1 (MVSK):", find_chunks_containing("MVSK", chunks))
# print("Q3 (auxiliary):", find_chunks_containing("auxiliary", chunks))
# print("Q4 (bull neutral bear):", find_chunks_containing("bull, neutral", chunks))
# print("Q6 (bear regime doubled):", find_chunks_containing("doubled", chunks))
# print("Q7 (downside covariance):", find_chunks_containing("zeroing all", chunks))
# print("Q9 (baselines):", find_chunks_containing("simulated annealing", chunks))
# print("Q10 (diagonal):", find_chunks_containing("diagonal elements", chunks))
# print("Q15 (Sharpe not energy):", find_chunks_containing("circuit convergence", chunks))

# print("\n--- Candidate chunks for Q1 ---","***"*12)
# for i in [2,3,5,12,13,17,28,29,38,54,61,62,63,64,68,69,72]:  # q1 candidates
#     print(f"--- Chunk {i} ---\n{chunks[i-1].page_content}\n")

#print sample one chunk with the metadata
print(f"Sample Chunk:")
print(f"chunk_id: {chunks[0].metadata['chunk_id']}")
print(f"Pages: {chunks[0].metadata['pages']}")
print(f"Content: {chunks[0].page_content[:100]}...")