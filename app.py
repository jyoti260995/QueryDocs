import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QueryDocs — AI Document Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Root ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

:root {
    --bg:       #0a0a0f;
    --surface:  #12121a;
    --card:     #1a1a26;
    --border:   #2a2a3d;
    --accent1:  #7c6fff;
    --accent2:  #ff6fb7;
    --accent3:  #6fffd4;
    --text:     #e8e8f0;
    --muted:    #7a7a9a;
    --user-bg:  #1e1e30;
    --ai-bg:    #141420;
}

/* ── Base App Shell ── */
.stApp {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* animated grain overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.6;
}

/* ── Header ── */
.dm-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 2.5rem 0 0.5rem;
    animation: fadeDown 0.6s ease both;
}
.dm-logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 0 28px rgba(124,111,255,0.45);
    flex-shrink: 0;
}
.dm-title-block h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #fff 30%, var(--accent1));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.dm-title-block p {
    font-size: 0.82rem;
    font-weight: 300;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Badge strip ── */
.dm-badge-row {
    display: flex; gap: 10px; margin: 1rem 0 2rem;
    flex-wrap: wrap;
    animation: fadeUp 0.7s 0.2s ease both;
}
.dm-badge {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.05em;
    display: flex; align-items: center; gap: 6px;
}
.dm-badge span { color: var(--accent3); }

/* ── Upload zone ── */
.dm-upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}

[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent1) !important;
    box-shadow: 0 0 20px rgba(124,111,255,0.12) !important;
}

/* ── Success / info ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 0.83rem !important;
}

/* ── Divider ── */
.dm-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}

/* ── Chat bubbles ── */
.dm-chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    padding-bottom: 1rem;
}
.dm-msg {
    display: flex;
    gap: 14px;
    animation: fadeUp 0.35s ease both;
}
.dm-msg.user { flex-direction: row-reverse; }

.dm-avatar {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0; margin-top: 2px;
}
.dm-avatar.user { background: linear-gradient(135deg, var(--accent1), var(--accent2)); }
.dm-avatar.ai   { background: linear-gradient(135deg, var(--accent3), var(--accent1)); }

.dm-bubble {
    max-width: 76%;
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 0.9rem;
    line-height: 1.65;
}
.dm-bubble.user {
    background: var(--user-bg);
    border: 1px solid rgba(124,111,255,0.25);
    border-top-right-radius: 4px;
    color: var(--text);
}
.dm-bubble.ai {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
    color: var(--text);
}

/* ── Typing dots ── */
.dm-typing { display: flex; gap: 5px; padding: 6px 2px; }
.dm-typing span {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent1);
    animation: bounce 1.2s infinite;
}
.dm-typing span:nth-child(2) { animation-delay: 0.2s; background: var(--accent2); }
.dm-typing span:nth-child(3) { animation-delay: 0.4s; background: var(--accent3); }

/* ── Source pills ── */
.dm-sources {
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px;
}
.dm-source-pill {
    background: rgba(124,111,255,0.1);
    border: 1px solid rgba(124,111,255,0.3);
    border-radius: 100px;
    padding: 3px 12px;
    font-size: 0.7rem;
    color: var(--accent1);
    font-weight: 500;
}

/* ── Chat input override ── */
[data-testid="stChatInput"] textarea {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent1) !important;
    box-shadow: 0 0 0 3px rgba(124,111,255,0.15) !important;
}

/* ── Footer ── */
.dm-footer {
    text-align: center;
    font-size: 0.7rem;
    color: var(--muted);
    padding: 1.5rem 0 0.5rem;
    letter-spacing: 0.05em;
}
.dm-footer a { color: var(--accent1); text-decoration: none; }

/* ── Animations ── */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes bounce {
    0%,60%,100% { transform: translateY(0); }
    30%          { transform: translateY(-7px); }
}

/* ── Streamlit chrome cleanup ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 1rem !important; max-width: 860px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Init Session State ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = None

# ─── Models ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = get_llm()
embedding_model = get_embeddings()

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dm-header">
  <div class="dm-logo">🔍</div>
  <div class="dm-title-block">
    <h1>QueryDocs</h1>
    <p>AI-Powered Document Intelligence</p>
  </div>
</div>
<div class="dm-badge-row">
  <div class="dm-badge">⚡ <span>Mistral AI</span></div>
  <div class="dm-badge">🔍 <span>Vector Search</span></div>
  <div class="dm-badge">📄 <span>PDF RAG Pipeline</span></div>
  <div class="dm-badge">🤗 <span>HuggingFace Embeddings</span></div>
</div>
""", unsafe_allow_html=True)

# ─── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="dm-upload-label">📎 Upload Document</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None:
    file_key = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.get("last_file_key") != file_key:
        st.session_state.last_file_key = file_key
        st.session_state.messages = []  # clear chat on new doc

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner(""):
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
            persist_dir = tempfile.mkdtemp()
            vdb = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=persist_dir
            )
            st.session_state.retriever = vdb.as_retriever(search_kwargs={"k": 3})
            st.session_state.doc_meta = {
                "name": uploaded_file.name,
                "pages": len(documents),
                "chunks": len(chunks)
            }

    meta = st.session_state.doc_meta
    st.success(
        f"✅ **{meta['name']}** — {meta['pages']} pages · {meta['chunks']} chunks indexed"
    )
    st.markdown('<div class="dm-divider"></div>', unsafe_allow_html=True)

# ─── Chat History ───────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown('<div class="dm-chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🔍"
        content_html = msg["content"].replace("\n", "<br>")

        if role == "user":
            st.markdown(f"""
            <div class="dm-msg user">
              <div class="dm-avatar user">{avatar}</div>
              <div class="dm-bubble user">{content_html}</div>
            </div>""", unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                pills = "".join(
                    f'<span class="dm-source-pill">p.{s}</span>'
                    for s in msg["sources"]
                )
                sources_html = f'<div class="dm-sources">{pills}</div>'
            st.markdown(f"""
            <div class="dm-msg ai">
              <div class="dm-avatar ai">{avatar}</div>
              <div class="dm-bubble ai">{content_html}{sources_html}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Input ──────────────────────────────────────────────────────────────────────
if st.session_state.retriever:
    question = st.chat_input("Ask anything about your document…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner(""):
            docs = st.session_state.retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            source_pages = sorted(set(
                doc.metadata.get("page", "?") + 1
                for doc in docs
                if doc.metadata.get("page") is not None
            ))
            prompt = f"""You are QueryDocs, an expert AI assistant that answers questions strictly from the provided document context.
Be concise, accurate, and insightful. Format your answer clearly.
If the answer is not in the context, respond: "This information isn't available in the uploaded document."

Context:
{context}

Question: {question}
Answer:"""
            response = llm.invoke(prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.content,
            "sources": source_pages
        })
        st.rerun()
else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #4a4a6a;">
      <div style="font-size:2.5rem; margin-bottom:1rem;">📄</div>
      <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600; color:#6a6a8a;">
        Upload a PDF to begin
      </div>
      <div style="font-size:0.82rem; margin-top:6px; color:#3a3a5a;">
        QueryDocs will index it and answer your questions instantly
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dm-footer">
  Built with ❤️ using <a href="#">Mistral AI</a> · LangChain · ChromaDB · Streamlit
</div>
""", unsafe_allow_html=True)
