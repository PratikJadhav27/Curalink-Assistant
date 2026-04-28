FROM python:3.12-slim

WORKDIR /app

# Install system deps (needed for torch + sentencepiece build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install torch CPU-only first (smaller download, avoids pulling CUDA libs)
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the custom model at image build time so the first request is fast.
# If the model doesn't exist on HF Hub yet, this step is skipped gracefully.
RUN python -c "\
import os; \
model_id = os.environ.get('CUSTOM_MODEL_ID', 'Pratik-027/curalink-medical-llm'); \
fallback = 'google/flan-t5-base'; \
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
try: \
    AutoTokenizer.from_pretrained(model_id); \
    AutoModelForSeq2SeqLM.from_pretrained(model_id); \
    print(f'Pre-downloaded custom model: {model_id}'); \
except Exception as e: \
    print(f'Custom model not available ({e}), pre-downloading fallback: {fallback}'); \
    AutoTokenizer.from_pretrained(fallback); \
    AutoModelForSeq2SeqLM.from_pretrained(fallback); \
" || echo "Model pre-download skipped."

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
