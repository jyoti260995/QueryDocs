"""Core RAG pipeline for QueryDocs, kept free of Streamlit so it can be tested."""

from __future__ import annotations

import html
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

NOT_FOUND_MESSAGE = "This information isn't available in the uploaded document."

PROMPT_TEMPLATE = """You are QueryDocs, an expert AI assistant that answers questions strictly from the provided document context.
Be concise, accurate, and insightful. Format your answer clearly.
If the answer is not in the context, respond: "{not_found}"

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class IndexedDocument:
    """A PDF that has been split and indexed, ready to answer questions."""

    retriever: VectorStoreRetriever
    pages: int
    chunks: int
    _persist_dir: str = field(repr=False, default="")

    def close(self) -> None:
        """Delete the on-disk vector store for this document."""
        if self._persist_dir:
            shutil.rmtree(self._persist_dir, ignore_errors=True)
            self._persist_dir = ""


def load_and_split(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[list[Document], list[Document]]:
    """Load a PDF and split it into overlapping chunks.

    Returns (pages, chunks). Raises ValueError if the PDF has no extractable text.
    """
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    if not chunks:
        raise ValueError(
            "No text could be extracted from this PDF (it may be a scanned image)."
        )
    return pages, chunks


def index_pdf_bytes(
    data: bytes,
    embedding: Embeddings,
    k: int = TOP_K,
) -> IndexedDocument:
    """Index an uploaded PDF given as bytes. The temporary PDF file is always removed."""
    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        pages, chunks = load_and_split(pdf_path)
    finally:
        os.remove(pdf_path)

    persist_dir = tempfile.mkdtemp(prefix="querydocs-")
    try:
        store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=persist_dir
        )
    except Exception:
        shutil.rmtree(persist_dir, ignore_errors=True)
        raise
    return IndexedDocument(
        retriever=store.as_retriever(search_kwargs={"k": k}),
        pages=len(pages),
        chunks=len(chunks),
        _persist_dir=persist_dir,
    )


def source_pages(docs: Iterable[Document]) -> list[int]:
    """1-based, sorted, de-duplicated page numbers of the retrieved chunks."""
    pages = set()
    for doc in docs:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            pages.add(page + 1)
    return sorted(pages)


def build_prompt(question: str, docs: Sequence[Document]) -> str:
    """Build the grounded prompt sent to the LLM."""
    context = "\n\n".join(doc.page_content for doc in docs)
    return PROMPT_TEMPLATE.format(
        not_found=NOT_FOUND_MESSAGE, context=context, question=question
    )


def message_html(role: str, content: str, sources: Sequence[int] = ()) -> str:
    """Render one chat message as HTML.

    Both the user's question and the model's answer are escaped, because the
    answer can echo text from the PDF or the question back into the page.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"unknown role: {role!r}")
    css = "user" if role == "user" else "ai"
    avatar = "👤" if role == "user" else "🔍"
    body = html.escape(content).replace("\n", "<br>")
    sources_html = ""
    if role == "assistant" and sources:
        pills = "".join(
            f'<span class="dm-source-pill">p.{int(p)}</span>' for p in sources
        )
        sources_html = f'<div class="dm-sources">{pills}</div>'
    return (
        f'<div class="dm-msg {css}">'
        f'<div class="dm-avatar {css}">{avatar}</div>'
        f'<div class="dm-bubble {css}">{body}{sources_html}</div>'
        f"</div>"
    )
