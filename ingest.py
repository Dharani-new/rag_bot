from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load the PDF
loader = PyPDFLoader("regulations.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages from the PDF")

# Step 2: Chunk it
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # roughly how many characters per chunk
    chunk_overlap=50     # slight overlap so context isn't lost at chunk boundaries
)

chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")
print("---")
print("Here's chunk #5 as an example:")
print(chunks[5].page_content)