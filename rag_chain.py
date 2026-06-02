import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. Load Mistral LLM
llm = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2
)

# 2. Load same embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Load existing ChromaDB
vector_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# 4. Create retriever
retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

def get_rag_response(question):
    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a helpful RAG assistant.

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

    return response.content


if __name__ == "__main__":
    question = input("Ask a question from your PDF: ")

    answer = get_rag_response(question)

    print("\nAnswer:\n")
    print(answer)