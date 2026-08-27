# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from dotenv import load_dotenv

# load_dotenv()

# # Step 1: Recreate the same embedding model used during ingestion
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# # Step 2: Load the saved vector store from disk (no need to re-embed anything!)
# vector_store = FAISS.load_local(
#     "faiss_index",
#     embeddings,
#     allow_dangerous_deserialization=True  # safe here since we created this file ourselves
# )

# # Step 3: Try a real question
# query = "How many academic years does a student have to complete their B.Tech degree?"

# results = vector_store.similarity_search(query, k=3)  # top 3 most relevant chunks

# print(f"Query: {query}\n")
# for i, doc in enumerate(results):
#     print(f"--- Result {i+1} (page {doc.metadata.get('page', '?')}) ---")
#     print(doc.page_content)
#     print()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

query = "How many academic years does a student have to complete their B.Tech degree?"
results = vector_store.similarity_search(query, k=3)

context = "\n\n".join([doc.page_content for doc in results])

prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

response = llm.invoke(prompt)

# Extract just the text, handling both old and new response formats
if isinstance(response.content, list):
    answer_text = response.content[0]['text']
else:
    answer_text = response.content

print(answer_text)