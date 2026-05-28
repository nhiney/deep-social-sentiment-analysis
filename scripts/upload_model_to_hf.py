"""Upload models/best_model/ to HuggingFace Hub.

Run once from project root:
    python scripts/upload_model_to_hf.py

Requires: pip install huggingface_hub
Login first: huggingface-cli login
"""
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID    = "nhiney/viemotion-model"
MODEL_DIR  = Path("models/best_model")
UPLOAD_FILES = ["pytorch_model.bin", "config.json", "tab_preprocessor.joblib"]

def main():
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)
    print(f"Repo ready: https://huggingface.co/{REPO_ID}")

    for fname in UPLOAD_FILES:
        fpath = MODEL_DIR / fname
        if not fpath.exists():
            print(f"  SKIP (not found): {fname}")
            continue
        size_mb = fpath.stat().st_size / 1e6
        print(f"  Uploading {fname} ({size_mb:.0f} MB)...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"  ✓ {fname}")

    print(f"\nDone! Model at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
