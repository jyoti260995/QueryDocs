from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load PDF
loader = PyPDFLoader("/Users/jyotigunpal/Documents/GENAI/RAG/document loaders/gawno.pdf")
documents = loader.load()

print("Pages loaded:", len(documents))

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))

# 3. Free embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store chunks in ChromaDB
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("ChromaDB created successfully.")