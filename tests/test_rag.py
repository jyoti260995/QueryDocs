import os
import tempfile

import pytest
from langchain_core.documents import Document

from querydocs.rag import (
    NOT_FOUND_MESSAGE,
    build_prompt,
    index_pdf_bytes,
    load_and_split,
    message_html,
    source_pages,
)
from tests.conftest import make_pdf


# ── load_and_split ──────────────────────────────────────────────────────────────

def test_load_and_split_counts_pages_and_keeps_page_metadata(pdf_factory):
    path = pdf_factory(["Alpha page text", "Beta page text", "Gamma page text"])
    pages, chunks = load_and_split(path)
    assert len(pages) == 3
    assert len(chunks) == 3
    assert [c.metadata["page"] for c in chunks] == [0, 1, 2]
    assert "Beta" in chunks[1].page_content


def test_long_page_is_split_with_overlap(pdf_factory):
    words = " ".join(f"word{i}" for i in range(400))  # ~3,000 characters
    path = pdf_factory([words])
    _, chunks = load_and_split(path, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)
    # consecutive chunks share text because of the overlap
    tail = chunks[0].page_content.split()[-1]
    assert tail in chunks[1].page_content


def test_pdf_without_text_raises_clear_error(pdf_factory):
    path = pdf_factory([""])
    with pytest.raises(ValueError, match="No text could be extracted"):
        load_and_split(path)


# ── index_pdf_bytes ─────────────────────────────────────────────────────────────

def test_index_retrieves_from_uploaded_bytes(fake_embedding):
    data = make_pdf(["The reactor coolant is heavy water.", "Unrelated page."])
    indexed = index_pdf_bytes(data, fake_embedding, k=2)
    try:
        assert indexed.pages == 2
        assert indexed.chunks == 2
        docs = indexed.retriever.invoke("coolant")
        assert len(docs) == 2
        assert {d.metadata["page"] for d in docs} == {0, 1}
    finally:
        indexed.close()


def test_index_removes_temp_pdf_and_close_removes_store(fake_embedding, monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    indexed = index_pdf_bytes(make_pdf(["hello world"]), fake_embedding)
    assert not list(tmp_path.glob("*.pdf")), "uploaded PDF copy was left on disk"
    store_dirs = list(tmp_path.glob("querydocs-*"))
    assert len(store_dirs) == 1
    indexed.close()
    assert not store_dirs[0].exists()
    indexed.close()  # closing twice is safe


def test_index_cleans_up_when_pdf_has_no_text(fake_embedding, monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(ValueError):
        index_pdf_bytes(make_pdf([""]), fake_embedding)
    assert os.listdir(tmp_path) == []


# ── source_pages ────────────────────────────────────────────────────────────────

def test_source_pages_are_one_based_sorted_and_unique():
    docs = [
        Document(page_content="a", metadata={"page": 4}),
        Document(page_content="b", metadata={"page": 0}),
        Document(page_content="c", metadata={"page": 4}),
    ]
    assert source_pages(docs) == [1, 5]


def test_source_pages_skips_chunks_without_a_page_number():
    docs = [
        Document(page_content="a", metadata={}),
        Document(page_content="b", metadata={"page": None}),
        Document(page_content="c", metadata={"page": 2}),
    ]
    assert source_pages(docs) == [3]


# ── build_prompt ────────────────────────────────────────────────────────────────

def test_prompt_contains_question_context_in_order_and_fallback():
    docs = [Document(page_content="first chunk"), Document(page_content="second chunk")]
    prompt = build_prompt("What is first?", docs)
    assert "Question: What is first?" in prompt
    assert prompt.index("first chunk") < prompt.index("second chunk")
    assert NOT_FOUND_MESSAGE in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_prompt_handles_braces_in_question_and_context():
    docs = [Document(page_content="config = {debug: true}")]
    prompt = build_prompt("What does {debug} do?", docs)
    assert "{debug: true}" in prompt
    assert "What does {debug} do?" in prompt


# ── message_html ────────────────────────────────────────────────────────────────

def test_message_html_escapes_user_and_model_text():
    evil = '<img src=x onerror="alert(1)">'
    for role in ("user", "assistant"):
        out = message_html(role, evil)
        assert "<img" not in out
        assert "&lt;img" in out


def test_message_html_keeps_line_breaks_and_shows_sources_for_answers_only():
    answer = message_html("assistant", "line one\nline two", [2, 7])
    assert "line one<br>line two" in answer
    assert 'class="dm-source-pill">p.2<' in answer
    assert "p.7" in answer
    question = message_html("user", "hi", [2])
    assert "dm-source-pill" not in question


def test_message_html_rejects_unknown_role():
    with pytest.raises(ValueError):
        message_html("system", "x")
