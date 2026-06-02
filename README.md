<div align="center">

# 🔍 QueryDocs

### AI-Powered Document Intelligence

**Upload any PDF. Ask anything. Get instant, accurate answers.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Mistral AI](https://img.shields.io/badge/Mistral_AI-mistral--small-f97316?style=flat-square)](https://mistral.ai)
[![LangChain](https://img.shields.io/badge/LangChain-0.x-1C3C3C?style=flat-square)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-6366f1?style=flat-square)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

![QueryDocs Demo](https://via.placeholder.com/860x400/0a0a0f/7c6fff?text=QueryDocs+—+Demo+Screenshot)

</div>

---

## ✨ What is QueryDocs?

**QueryDocs** is a full-stack RAG (Retrieval-Augmented Generation) chatbot that lets you have intelligent conversations with any PDF document. Built with a production-grade pipeline — from vector embeddings to LLM inference — and wrapped in a sleek, dark-themed UI.

Ask it to summarize a research paper, extract key clauses from a contract, or quiz you on a textbook. QueryDocs finds the right context and answers with precision.

---

## 🚀 Features

- 📄 **PDF Upload & Parsing** — Drag-and-drop any PDF; pages are extracted and chunked automatically
- 🧩 **Semantic Chunking** — `RecursiveCharacterTextSplitter` with overlap for coherent context windows
- 🔢 **Vector Embeddings** — `all-MiniLM-L6-v2` via HuggingFace for fast, accurate similarity search
- 🗄️ **ChromaDB Vector Store** — In-memory vector database per session, no external DB needed
- 🤖 **Mistral AI LLM** — `mistral-small-latest` for concise, grounded answers
- 💬 **Persistent Chat History** — Full multi-turn conversation within a session
- 📍 **Source Page Citations** — Every answer shows which pages were referenced
- 🎨 **Premium Dark UI** — Custom Streamlit CSS with Syne + DM Sans typography

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PyPDFLoader  ──►  RecursiveCharacterTextSplitter  ──►  Chunks
                                                           │
                                                           ▼
                                               HuggingFace Embeddings
                                               (all-MiniLM-L6-v2)
                                                           │
                                                           ▼
                                                     ChromaDB
                                                   Vector Store
                                                           │
                                         User Query ──► Retriever (top-k=3)
                                                           │
                                                           ▼
                                              Mistral AI (mistral-small)
                                            + Grounded Prompt Template
                                                           │
                                                           ▼
                                                    Answer + Sources
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit + Custom CSS |
| **LLM** | Mistral AI (`mistral-small-latest`) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Store** | ChromaDB |
| **PDF Parsing** | LangChain `PyPDFLoader` |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **Env Management** | python-dotenv |

---

## ⚡ Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/querydocs.git
cd querydocs
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get your free Mistral API key at [console.mistral.ai](https://console.mistral.ai)

### 4. Run the app

```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📦 Requirements

```txt
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-chroma
langchain-mistralai
chromadb
sentence-transformers
pypdf
python-dotenv
```

Or install all at once:

```bash
pip install streamlit langchain langchain-community langchain-text-splitters \
  langchain-huggingface langchain-chroma langchain-mistralai chromadb \
  sentence-transformers pypdf python-dotenv
```

---

## 📁 Project Structure

```
querydocs/
├── main.py            # Main Streamlit application
├── .env               # API keys (not committed)
├── .env.example       # Example env file
├── requirements.txt   # Python dependencies
└── README.md          # You are here
```

---

## 🔮 Roadmap

- [ ] Multi-PDF support (upload and query across documents)
- [ ] Export chat history as PDF/Markdown
- [ ] Conversation memory across sessions
- [ ] Support for DOCX, TXT, and web URLs
- [ ] Streaming responses
- [ ] Docker deployment

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repo
2. Create your branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by **[Your Name](https://github.com/yourusername)**

⭐ Star this repo if you found it useful!

</div>
