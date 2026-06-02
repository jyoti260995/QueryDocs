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

st.set_page_config(
    page_title="AskMyDocs",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AskMyDocs")
st.caption("Upload. Ask. Understand.")

llm = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    with st.spinner("Processing your document..."):

        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Unique temporary ChromaDB folder for every upload
        persist_directory = tempfile.mkdtemp()

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )

        retriever = vector_db.as_retriever(
            search_kwargs={"k": 3}
        )

    st.success(
        f"PDF processed successfully. Pages: {len(documents)}, Chunks: {len(chunks)}"
    )

    question = st.chat_input("Ask a question from your PDF")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Searching document and generating answer..."):

            docs = retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            prompt = f"""
You are a helpful document assistant.

Answer the question using ONLY the context below.

If the answer is not present in the context, say:
"I don't know from the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

            response = llm.invoke(prompt)

        with st.chat_message("assistant"):
            st.write(response.content)

else:
    st.info("Please upload a PDF to start chatting.")