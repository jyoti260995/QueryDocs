from langchain_community.document_loaders import PyPDFLoader
pdf_path = "/Users/jyotigunpal/Documents/GENAI/RAG/document loaders/gawno.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(docs[2].page_content)

