# Multi-stage Dockerfile for Ashcombe & Rowe Estates AI Lead Qualification Chatbot
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and data
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default environment
ENV PYTHONUNBUFFERED=1
ENV LLM_PROVIDER=groq
ENV EMBEDDING_PROVIDER=local

# Default command: launch FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
