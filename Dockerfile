# QueryDocs — container image
#
# Build:  docker build -t querydocs .
# Run:    docker run --rm -p 8501:8501 -e MISTRAL_API_KEY=your_key querydocs
# Open:   http://localhost:8501
#
# The embedding model is downloaded at build time, so the container needs
# network access only for the Mistral API, not for HuggingFace.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first: keeps the image ~2 GB smaller than the default CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Install dependencies before copying the code, so code edits don't redo this layer
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image (works offline after build)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy only the app code — never .env or other local files with secrets
COPY app.py .
COPY querydocs/ querydocs/

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
ENV HF_HUB_OFFLINE=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
