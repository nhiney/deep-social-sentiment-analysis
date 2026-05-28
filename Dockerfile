FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install deps trước để tận dụng Docker cache
COPY requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

# Copy source code (không copy models/ hay data/)
COPY app/ app/
COPY src/ src/
COPY configs/ configs/
COPY scripts/hf_startup.py scripts/hf_startup.py

# HF Spaces chạy port 7860
EXPOSE 7860

ENV MODEL_CHECKPOINT=/code/models/best_model
ENV HF_MODEL_REPO=nhiney/viemotion-model

CMD ["python", "scripts/hf_startup.py"]
