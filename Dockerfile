# Dockerfile for Ashcombe & Rowe Estates AI Lead Qualification Chatbot
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Install CPU-only PyTorch (drastically reduces image size from ~5GB to <1.2GB and speeds up build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install Python project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-cache SentenceTransformer model inside the image for fast, offline startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 4. Copy application source code and data (clean context via .dockerignore)
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

ENV PYTHONUNBUFFERED=1
ENV EMBEDDING_PROVIDER=local
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2

# Default command: launch FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
