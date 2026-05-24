"""Zero-shot pseudo-labelling of unlabeled Facebook posts (Apify scrape).

Uses a multilingual NLI model (mDeBERTa-v3-base-mnli-xnli by default) to
assign one of the 7 Ekman emotion labels to each post in the cleaned Apify CSV.

Design decisions
----------------
* **Why mDeBERTa-v3-base-mnli-xnli?**
  Trained on 41-language MNLI+XNLI — Vietnamese is in its pretraining corpus.
  Significantly outperforms BART-large-mnli on non-English text.

* **Vietnamese candidate labels.**
  The NLI model evaluates whether each candidate hypothesis is *entailed* by
  the post. Descriptive Vietnamese phrases ("niềm vui và hạnh phúc") work
  better than bare English keywords ("joy") because the Vietnamese surface
  matches the premise language, reducing cross-lingual transfer noise.

* **Confidence threshold (default 0.35).**
  Pseudo-labels below this threshold are kept in the output CSV but flagged
  with ``pseudo_label_confident = False``. The downstream prepare_data.py
  step can decide whether to include them. Setting 0.35 is intentionally
  conservative — at 7 random-chance labels the expected score is ~0.143,
  so 0.35 means the model is at least 2.5× more confident than chance.

* **All predictions saved.**
  We store ``pseudo_top1``, ``pseudo_top2``, ``pseudo_top3`` + their scores
  so future ensemble / calibration experiments have the full distribution.

Run from project root::

    python -m scripts.pseudo_label_apify

    # Full options:
    python -m scripts.pseudo_label_apify \\
        --input     data/processed/cleaned_unlabeled_posts.csv \\
        --output    data/processed/pseudo_labeled_apify.csv \\
        --model     MoritzLaurer/mDeBERTa-v3-base-mnli-xnli \\
        --batch-size 16 \\
        --threshold  0.35 \\
        --device     auto
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pseudo_label")

# ---------------------------------------------------------------------------
# Emotion schema
# ---------------------------------------------------------------------------
# 7-class Ekman emotions — must match CANONICAL_LABELS in prepare_data.py
CANONICAL_LABELS: List[str] = [
    "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral",
]

# Vietnamese descriptive phrases used as zero-shot candidate labels.
# Each phrase is designed to be unambiguous enough that an NLI model trained
# on paraphrase / entailment can score it against a Vietnamese Facebook post.
CANDIDATE_LABELS_VI: List[str] = [
    "niềm vui và hạnh phúc",            # joy
    "buồn bã và đau khổ",               # sadness
    "tức giận và bực bội",              # anger
    "sợ hãi và lo âu",                  # fear
    "ghê tởm và phản cảm",              # disgust
    "ngạc nhiên và bất ngờ",            # surprise
    "thông tin trung tính không cảm xúc", # neutral
]

# Fallback: English candidates (slightly lower performance for Vietnamese text
# but sometimes produces more calibrated scores for rare emotions)
CANDIDATE_LABELS_EN: List[str] = [
    "joy and happiness",
    "sadness and sorrow",
    "anger and frustration",
    "fear and anxiety",
    "disgust and repulsion",
    "surprise and amazement",
    "neutral information with no clear emotion",
]

# Maps candidate label string → canonical emotion key.
# Vietnamese and English share the same mapping by position.
_VI_TO_EMOTION: Dict[str, str] = dict(zip(CANDIDATE_LABELS_VI, CANONICAL_LABELS))
_EN_TO_EMOTION: Dict[str, str] = dict(zip(CANDIDATE_LABELS_EN, CANONICAL_LABELS))


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def _resolve_device(spec: str) -> int:
    """Return a transformers-compatible device integer (0 = CUDA, -1 = CPU)."""
    if spec == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Auto-selected device: CUDA")
                return 0
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Auto-selected device: MPS (Apple Silicon)")
                # HuggingFace pipeline does not yet accept 'mps' as int — use CPU
                return -1
        except ImportError:
            pass
        logger.info("Auto-selected device: CPU")
        return -1
    if spec == "cuda":
        return 0
    return -1


# ---------------------------------------------------------------------------
# Zero-shot classifier
# ---------------------------------------------------------------------------
def build_classifier(model_name: str, device: int):
    """Load the HuggingFace zero-shot classification pipeline.

    Parameters
    ----------
    model_name : str
        HuggingFace model hub ID.
    device : int
        0 for CUDA, -1 for CPU.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        logger.error("transformers not installed: pip install transformers")
        raise SystemExit(1) from exc

    logger.info("Loading zero-shot model '%s' (device=%s)…", model_name, device)
    t0 = time.time()
    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
        # Batching is handled externally so we can log progress.
        batch_size=1,
    )
    logger.info("Model loaded in %.1fs.", time.time() - t0)
    return clf


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def classify_batch(
    classifier,
    texts: List[str],
    candidate_labels: List[str],
    label_to_emotion: Dict[str, str],
    hypothesis_template: str,
    batch_size: int,
) -> List[Dict]:
    """Run zero-shot classification and return per-text result dicts.

    Each result dict has:
        ``emotion``     — top predicted Ekman emotion
        ``confidence``  — NLI entailment score for the top candidate
        ``top1 / top2 / top3``        — top-3 candidate label strings
        ``score1 / score2 / score3``  — corresponding scores
    """
    try:
        from tqdm import tqdm
        progress = tqdm(total=len(texts), desc="Zero-shot classification",
                        unit="post", dynamic_ncols=True)
    except ImportError:
        progress = None

    results: List[Dict] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        chunk = texts[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        # The pipeline accepts a list of texts.
        raw = classifier(
            chunk,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )

        # ``raw`` is a list if chunk has >1 element, else a single dict.
        if isinstance(raw, dict):
            raw = [raw]

        for item in raw:
            top_labels = item["labels"]    # sorted descending by score
            top_scores = item["scores"]

            top1_cand = top_labels[0]
            results.append({
                "emotion":    label_to_emotion.get(top1_cand, "neutral"),
                "confidence": float(top_scores[0]),
                "top1":  top_labels[0],
                "score1": float(top_scores[0]),
                "top2":  top_labels[1] if len(top_labels) > 1 else "",
                "score2": float(top_scores[1]) if len(top_scores) > 1 else 0.0,
                "top3":  top_labels[2] if len(top_labels) > 2 else "",
                "score3": float(top_scores[2]) if len(top_scores) > 2 else 0.0,
            })

        if progress is not None:
            progress.update(len(chunk))

    if progress is not None:
        progress.close()

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def pseudo_label(args: argparse.Namespace) -> pd.DataFrame:
    """Full pseudo-labeling pipeline. Returns the labelled DataFrame."""
    input_path  = Path(args.input)
    output_path = Path(args.output)

    # ---- Load ----
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info("Loaded %d posts from %s.", len(df), input_path)

    if "text" not in df.columns:
        logger.error("Input CSV must have a 'text' column. Found: %s", df.columns.tolist())
        sys.exit(1)

    # Drop rows with no usable text (defensive — should already be clean).
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    logger.info("After null/empty drop: %d posts remain.", len(df))

    # ---- Select candidate label language ----
    if args.use_english_labels:
        candidate_labels = CANDIDATE_LABELS_EN
        label_to_emotion = _EN_TO_EMOTION
        hypothesis_template = "This text is about {}."
        logger.info("Using English candidate labels.")
    else:
        candidate_labels = CANDIDATE_LABELS_VI
        label_to_emotion = _VI_TO_EMOTION
        hypothesis_template = "Văn bản này thể hiện {}."
        logger.info("Using Vietnamese candidate labels.")

    # ---- Build classifier ----
    device = _resolve_device(args.device)
    clf = build_classifier(args.model, device)

    # ---- Run inference ----
    texts = df["text"].tolist()
    logger.info(
        "Classifying %d texts with batch_size=%d…", len(texts), args.batch_size
    )
    t0 = time.time()
    predictions = classify_batch(
        clf, texts, candidate_labels, label_to_emotion,
        hypothesis_template=hypothesis_template,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0
    logger.info("Inference done in %.1fs (%.2f s/post).", elapsed, elapsed / len(texts))

    # ---- Attach predictions ----
    pred_df = pd.DataFrame(predictions)
    df["label"]              = pred_df["emotion"].values
    df["pseudo_confidence"]  = pred_df["confidence"].values.round(4)
    df["pseudo_confident"]   = (df["pseudo_confidence"] >= args.threshold)
    df["pseudo_top1"]        = pred_df["top1"].values
    df["pseudo_score1"]      = pred_df["score1"].values.round(4)
    df["pseudo_top2"]        = pred_df["top2"].values
    df["pseudo_score2"]      = pred_df["score2"].values.round(4)
    df["pseudo_top3"]        = pred_df["top3"].values
    df["pseudo_score3"]      = pred_df["score3"].values.round(4)
    df["is_crawled"]         = 1      # flag for prepare_data.py

    # ---- Statistics ----
    n_total     = len(df)
    n_confident = df["pseudo_confident"].sum()
    logger.info(
        "Confidence stats:\n%s",
        df["pseudo_confidence"].describe().round(3).to_string(),
    )
    logger.info(
        "Posts above threshold (%.2f): %d / %d (%.1f%%)",
        args.threshold, n_confident, n_total, 100 * n_confident / n_total,
    )
    logger.info(
        "Label distribution (ALL posts):\n%s",
        df["label"].value_counts().to_string(),
    )
    logger.info(
        "Label distribution (confident posts):\n%s",
        df.loc[df["pseudo_confident"], "label"].value_counts().to_string(),
    )

    # ---- Save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Saved %d pseudo-labeled posts → %s\n"
        "  (%d confident, %d below threshold — flagged but retained).",
        n_total, output_path, n_confident, n_total - n_confident,
    )

    # ---- Quick preview ----
    print("\nFirst 5 rows (text | label | confidence | confident):")
    preview_cols = ["text", "label", "pseudo_confidence", "pseudo_confident"]
    with pd.option_context("display.max_colwidth", 60):
        print(df[preview_cols].head().to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Zero-shot pseudo-labelling of unlabeled Facebook posts "
            "using a multilingual NLI model."
        )
    )
    p.add_argument(
        "--input", type=str,
        default="data/processed/cleaned_unlabeled_posts.csv",
        help="Path to the cleaned Apify CSV.",
    )
    p.add_argument(
        "--output", type=str,
        default="data/processed/pseudo_labeled_apify.csv",
        help="Output path for the pseudo-labeled CSV.",
    )
    p.add_argument(
        "--model", type=str,
        default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        help=(
            "HuggingFace model ID for zero-shot classification. "
            "Alternatives: 'joeddav/xlm-roberta-large-xnli'."
        ),
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Number of texts per inference batch.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.35,
        help=(
            "Confidence threshold for 'pseudo_confident' flag. "
            "Posts below this score are retained in the CSV but flagged False."
        ),
    )
    p.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="'auto' picks CUDA if available, else CPU.",
    )
    p.add_argument(
        "--use-english-labels", action="store_true",
        help="Use English candidate labels instead of Vietnamese (for ablation).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pseudo_label(args)


if __name__ == "__main__":
    main()
