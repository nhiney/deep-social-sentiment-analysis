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
COPY reports/figures/ reports/figures/

# Download model files at build time (cached as Docker layer)
RUN python - << 'PYEOF'
import urllib.request, pathlib, sys
ckpt = pathlib.Path("models/best_model")
ckpt.mkdir(parents=True, exist_ok=True)
base = "https://huggingface.co/nhiney/viemotion-model/resolve/main"
files = [
    ("config.json",              "2KB"),
    ("tab_preprocessor.joblib",  "1KB"),
    ("pytorch_model.bin",        "1.1GB"),
]
for fname, size in files:
    dest = ckpt / fname
    print(f"Downloading {fname} ({size})...", flush=True)
    urllib.request.urlretrieve(f"{base}/{fname}", str(dest))
    print(f"  ✓ {fname} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
print("Model download complete.", flush=True)
PYEOF

# HF Spaces chạy port 7860
EXPOSE 7860

ENV MODEL_CHECKPOINT=/code/models/best_model

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
