"""HF Spaces startup: download model from HF Hub then launch uvicorn."""
import os
import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path(os.environ.get("MODEL_CHECKPOINT", "/code/models/best_model"))
HF_REPO   = os.environ.get("HF_MODEL_REPO", "nhiney/viemotion-model")

def download_model():
    from huggingface_hub import snapshot_download
    print(f"[startup] Downloading model from {HF_REPO} → {MODEL_DIR}", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(MODEL_DIR),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print("[startup] Model ready.", flush=True)

def main():
    needs_download = (
        not MODEL_DIR.exists()
        or not (MODEL_DIR / "pytorch_model.bin").exists()
        or not (MODEL_DIR / "tab_preprocessor.joblib").exists()
    )
    if needs_download:
        download_model()
    else:
        print(f"[startup] Model already at {MODEL_DIR}, skipping download.", flush=True)

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "7860",
        "--workers", "1",
    ])

if __name__ == "__main__":
    main()
