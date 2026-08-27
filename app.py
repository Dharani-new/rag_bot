import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import tempfile
import os
import time

load_dotenv()

st.set_page_config(page_title="RAG Regulations Q&A", page_icon="🎓", layout="centered")

# ---- Custom styling ----
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #eef4ff 0%, #dce9fb 45%, #cfe0f7 100%);
        color: #1f2937;
    }
    h1 {
        color: #1e293b !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    p, label, .stMarkdown {
        color: #475569 !important;
    }
    .stTextInput input {
        background-color: #ffffff;
        color: #1e293b;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stTextInput input:focus {
        border: 1px solid #6366f1;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
    }
    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1.5px dashed #94a3b8;
        border-radius: 14px;
        padding: 16px;
    }
    .stAlert {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stVerticalBlock"] > div:has(> div.stMarkdown) {
        border-radius: 12px;
    }
    /* Answer box styling */
    .stMarkdown p {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 RAG Document Q&A")
st.write("Upload a PDF, then ask questions about it — answers are grounded in the document, not guesses.")

# ---- Embedding + LLM setup ----
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash")

embeddings = get_embeddings()
llm = get_llm()

# ---- PDF upload ----
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.doc_name = None

if uploaded_file is not None and st.session_state.doc_name != uploaded_file.name:
    with st.spinner(f"Processing {uploaded_file.name} — chunking and embedding, this may take a minute..."):
        # Save uploaded file temporarily so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Embed in batches to stay under the free-tier rate limit (100 requests/minute)
        batch_size = 40
        vector_store = None
        progress_bar = st.progress(0, text="Embedding chunks...")
        total_batches = (len(chunks) // batch_size) + (1 if len(chunks) % batch_size else 0)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1

            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)

            progress_bar.progress(batch_num / total_batches, text=f"Embedded batch {batch_num} of {total_batches}...")

            if i + batch_size < len(chunks):
                time.sleep(65)  # stay safely under the 100/min free-tier limit

        progress_bar.empty()

        st.session_state.vector_store = vector_store
        st.session_state.doc_name = uploaded_file.name
        os.unlink(tmp_path)  # clean up temp file

    st.success(f"'{uploaded_file.name}' processed — {len(chunks)} chunks ready to search.")

# ---- Query ----
if st.session_state.vector_store is not None:
    query = st.text_input(
        "Your question (press Enter to ask):",
        placeholder="e.g. How many credits are needed for an Honors degree?"
    )

    if query:
        with st.spinner("Searching document and generating answer..."):
            is_summary_request = any(word in query.lower() for word in ["summarize", "summary", "overview", "what is this"])
            k = 10 if is_summary_request else 3

            results = st.session_state.vector_store.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in results])

            if is_summary_request:
                prompt = f"""You are looking at chunks from a document. Based on the context below, first identify what kind of document this is and who/what it's about (e.g. "This is a resume for [Name]" or "This is a policy document about X"). Then give a concise 3-4 sentence summary of its key contents. Do not list every detail — just the essence.

Context:
{context}

Answer:"""
            else:
                prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

            response = llm.invoke(prompt)
            answer_text = response.content[0]['text'] if isinstance(response.content, list) else response.content

            st.markdown(f"""
            <div style="background-color:#ffffff; border-radius:14px; padding:20px 24px;
                        border:1px solid #e2e8f0; box-shadow:0 2px 8px rgba(0,0,0,0.05); margin-top:10px;">
                <p style="color:#6366f1; font-weight:600; font-size:0.9rem; margin-bottom:8px; letter-spacing:0.3px;">ANSWER</p>
                <p style="color:#1e293b; font-size:1.05rem; line-height:1.6; margin:0;">{answer_text}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📄 Show retrieved source chunks"):
                for i, doc in enumerate(results):
                    st.markdown(f"**Chunk {i+1}** (page {doc.metadata.get('page', '?')})")
                    st.text(doc.page_content)
                    st.divider()
else:
    st.info("Upload a PDF above to get started.")