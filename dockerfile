FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy all files
COPY . .

# Expose port
EXPOSE 7860

# Start server — HF Spaces always uses port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]