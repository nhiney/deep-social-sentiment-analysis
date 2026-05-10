"""End-to-end data preparation pipeline.

Pipeline
--------
    raw sources ─►  loaders  ─►  schema alignment  ─►  agreement check
                                                              │
              teencode normalization ◄─── concat & dedupe ◄───┘
                        │
                        └──►  stratified 70/15/15 split  ──►  data/processed/

Run from project root::

    python -m scripts.prepare_data \
        --uit-vsmec   data/raw/UIT-VSMEC.csv \
        --phonlp      data/raw/phonlp_sentiment.csv \
        --crawled     data/raw/merged_dataset.xlsx \
        --output-dir  data/processed \
        --seed        42

Each loader produces a *canonical* DataFrame with the columns:

    text          : str   — raw social-media text
    label         : str   — sentiment in {"negative", "neutral", "positive"}
    source        : str   — provenance tag ("uit-vsmec", "phonlp", "crawled")
    annotator_a   : str   — primary annotator label  (crawled only — else NaN)
    annotator_b   : str   — secondary annotator label (crawled only — else NaN)

Behavior tabular columns (numerical / categorical) are passed through if
present; the FT-Transformer branch consumes them downstream.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Ensure ``src`` is importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (  # noqa: E402
    TeencodeNormalizer,
    cohens_kappa,
    stratified_split,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_data")

# --------------------------------------------------------------------------- #
# Canonical label schema — 7-way Ekman emotions + Neutral
# --------------------------------------------------------------------------- #
# We adopt the Ekman 6 + Neutral schema since (a) it matches the crawled
# dataset's `Mã nhãn` column and (b) UIT-VSMEC ships 7 compatible emotions.
# A 3-way sentiment "view" is recovered post-hoc via EMOTION_TO_SENTIMENT.
CANONICAL_LABELS: List[str] = [
    "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral",
]

# Optional collapse map for downstream 3-way sentiment evaluation.
EMOTION_TO_SENTIMENT: Dict[str, str] = {
    "joy":      "positive",
    "surprise": "neutral",
    "neutral":  "neutral",
    "sadness":  "negative",
    "anger":    "negative",
    "disgust":  "negative",
    "fear":     "negative",
}

# UIT-VSMEC publishes 7 emotion classes ({Enjoyment, Sadness, Anger, ...}).
UIT_VSMEC_LABEL_MAP: Dict[str, str] = {
    "Enjoyment": "joy",
    "Sadness":   "sadness",
    "Anger":     "anger",
    "Fear":      "fear",
    "Disgust":   "disgust",
    "Surprise":  "surprise",
    "Other":     "neutral",
}

# PhoNLP sentiment subset — only 3-way available; we project up by treating
# POS→joy and NEG→sadness as the dominant proxies (best-effort heuristic).
PHONLP_LABEL_MAP: Dict[str, str] = {
    "POS": "joy",     "positive": "joy",     "1":  "joy",
    "NEU": "neutral", "neutral":  "neutral", "0":  "neutral",
    "NEG": "sadness", "negative": "sadness", "-1": "sadness",
}

# Self-crawled labels in `crawled_emotions.xlsx` use 3-letter Ekman codes
# in the `Mã nhãn` column. We also accept the Vietnamese long form from
# the `Label` column as a fallback.
CRAWLED_LABEL_MAP: Dict[str, str] = {
    # Short codes (Mã nhãn)
    "joy": "joy", "sad": "sadness", "ang": "anger", "fea": "fear",
    "dis": "disgust", "sur": "surprise", "neu": "neutral",
    # Long Vietnamese names (Label) — lowercased
    "hạnh phúc (enjoyment/joy)": "joy",
    "buồn bã (sadness)":         "sadness",
    "giận dữ (anger)":           "anger",
    "sợ hãi (fear)":             "fear",
    "ghê tởm (disgust)":         "disgust",
    "ngạc nhiên (surprise)":     "surprise",
    "trung tính (neutral)":      "neutral",
}


# =========================================================================== #
# Loaders — one per source, returning the canonical schema
# =========================================================================== #
def load_uit_vsmec(path: Path) -> pd.DataFrame:
    """Load UIT-VSMEC and project its 7 emotions into 3-way sentiment.

    The dataset is published as CSV with columns ``Sentence`` and ``Emotion``.

    Parameters
    ----------
    path : Path
        CSV path. Skipped (returns empty DataFrame) if the file is missing.
    """
    if not path.exists():
        logger.warning("UIT-VSMEC not found at %s — skipping.", path)
        return _empty_canonical()

    logger.info("Loading UIT-VSMEC from %s", path)
    df = pd.read_csv(path)

    # The published file uses {Sentence, Emotion}; tolerate small variants.
    text_col = _first_present(df, ["Sentence", "sentence", "text", "Text"])
    label_col = _first_present(df, ["Emotion", "emotion", "label", "Label"])

    out = pd.DataFrame({
        "text":  df[text_col].astype(str),
        "label": df[label_col].astype(str).map(UIT_VSMEC_LABEL_MAP),
        "source": "uit-vsmec",
        "annotator_a": np.nan,  # no per-sample annotator info in this corpus
        "annotator_b": np.nan,
    })
    return _drop_unmapped(out, "uit-vsmec")


def load_phonlp(path: Path) -> pd.DataFrame:
    """Load the PhoNLP sentiment subset (CSV/TSV)."""
    if not path.exists():
        logger.warning("PhoNLP not found at %s — skipping.", path)
        return _empty_canonical()

    logger.info("Loading PhoNLP from %s", path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)

    text_col = _first_present(df, ["text", "sentence", "Sentence"])
    label_col = _first_present(df, ["label", "sentiment", "Sentiment"])

    out = pd.DataFrame({
        "text":  df[text_col].astype(str),
        "label": df[label_col].astype(str).map(PHONLP_LABEL_MAP),
        "source": "phonlp",
        "annotator_a": np.nan,
        "annotator_b": np.nan,
    })
    return _drop_unmapped(out, "phonlp")


def load_crawled(path: Path) -> pd.DataFrame:
    """Load the self-crawled / merged Excel dataset (.xlsx).

    Expects two annotator columns ``annotator_a`` / ``annotator_b`` so we can
    measure inter-annotator agreement. Falls back gracefully if only one
    label column is present (kappa step is then skipped).
    """
    if not path.exists():
        logger.warning("Crawled dataset not found at %s — skipping.", path)
        return _empty_canonical()

    logger.info("Loading crawled dataset from %s", path)
    df = pd.read_excel(path)

    text_col = _first_present(df, ["Text", "text", "content", "post", "Sentence"])

    # Primary label column — ``Mã nhãn`` holds the 3-letter Ekman code.
    # ``Label`` is the long Vietnamese form, kept as a fallback.
    primary_col = _first_present(
        df, ["Mã nhãn", "ma_nhan", "label_code"], required=False,
    )
    fallback_col = _first_present(
        df, ["Label", "label", "emotion"], required=False,
    )
    a_col = _first_present(
        df, ["annotator_a", "label_a", "label1"], required=False,
    )
    b_col = _first_present(
        df, ["annotator_b", "label_b", "label2"], required=False,
    )

    # Build canonical view.
    out = pd.DataFrame({"text": df[text_col].astype(str), "source": "crawled"})

    # ---- Per-annotator labels (used for κ if both columns present) ----
    if a_col is not None:
        out["annotator_a"] = (
            df[a_col].astype(str).str.lower().str.strip().map(CRAWLED_LABEL_MAP)
        )
    else:
        out["annotator_a"] = np.nan
    if b_col is not None:
        out["annotator_b"] = (
            df[b_col].astype(str).str.lower().str.strip().map(CRAWLED_LABEL_MAP)
        )
    else:
        out["annotator_b"] = np.nan

    # ---- Resolve final label ----
    if a_col is not None and b_col is not None:
        # Both annotators present → use consensus rule.
        out["label"] = _resolve_consensus(out["annotator_a"], out["annotator_b"])
    elif primary_col is not None:
        # Single-label dataset → map directly from Mã nhãn / equivalent.
        out["label"] = (
            df[primary_col].astype(str).str.lower().str.strip().map(CRAWLED_LABEL_MAP)
        )
    elif fallback_col is not None:
        out["label"] = (
            df[fallback_col].astype(str).str.lower().str.strip().map(CRAWLED_LABEL_MAP)
        )
    else:
        raise KeyError(
            "Crawled dataset has no label column "
            "(expected one of: Mã nhãn / Label / annotator_a)."
        )

    # Pass through any tabular behavior columns (post_count, etc.) untouched.
    used_cols = {text_col, primary_col, fallback_col, a_col, b_col, "ID"}
    for col in df.columns:
        if col not in used_cols and col not in out.columns:
            out[col] = df[col].values

    return _drop_unmapped(out, "crawled")


# =========================================================================== #
# Helpers
# =========================================================================== #
def _empty_canonical() -> pd.DataFrame:
    """Return an empty DataFrame matching the canonical schema."""
    return pd.DataFrame({
        "text": pd.Series(dtype=str),
        "label": pd.Series(dtype=str),
        "source": pd.Series(dtype=str),
        "annotator_a": pd.Series(dtype=str),
        "annotator_b": pd.Series(dtype=str),
    })


def _first_present(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    """Return the first column from ``candidates`` that exists in ``df``."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"None of {list(candidates)} present in dataframe. "
            f"Got columns: {list(df.columns)}"
        )
    return None


def _drop_unmapped(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Drop rows whose label could not be mapped to the canonical schema."""
    n_before = len(df)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df = df[df["label"].isin(CANONICAL_LABELS)].reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        logger.info(
            "  [%s] dropped %d rows with unmapped/missing labels (%d → %d).",
            source_name, n_before - n_after, n_before, n_after,
        )
    return df


def _resolve_consensus(a: pd.Series, b: pd.Series) -> pd.Series:
    """Merge two annotator label columns into a single consensus label.

    Rules:
        * Both non-null and equal      → that label.
        * Both non-null and disagree   → NaN (filtered out downstream).
        * Exactly one non-null         → that one.
        * Both null                    → NaN.
    """
    consensus: List[Any] = []
    for la, lb in zip(a.tolist(), b.tolist()):
        la_ok = isinstance(la, str) and la in CANONICAL_LABELS
        lb_ok = isinstance(lb, str) and lb in CANONICAL_LABELS
        if la_ok and lb_ok:
            consensus.append(la if la == lb else np.nan)
        elif la_ok:
            consensus.append(la)
        elif lb_ok:
            consensus.append(lb)
        else:
            consensus.append(np.nan)
    return pd.Series(consensus, index=a.index, dtype=object)


# =========================================================================== #
# Cross-annotation quality check
# =========================================================================== #
def report_annotator_agreement(
    df: pd.DataFrame,
    threshold: float = 0.6,
) -> Optional[float]:
    """Compute & log Cohen's Kappa between annotator_a and annotator_b.

    Only rows where BOTH annotators provided a label contribute to κ.

    Parameters
    ----------
    df : DataFrame
        Must contain ``annotator_a`` and ``annotator_b`` columns.
    threshold : float, default=0.6
        Minimum acceptable κ. We log a warning if we drop below it
        (Landis & Koch's "substantial agreement" cut-off).

    Returns
    -------
    float or None
        The computed κ, or ``None`` if there aren't enough dual-labeled rows.
    """
    if "annotator_a" not in df.columns or "annotator_b" not in df.columns:
        logger.info("No dual-annotation columns found — skipping κ check.")
        return None

    # Restrict to rows where both annotators actually provided a label.
    paired = df.dropna(subset=["annotator_a", "annotator_b"])
    if len(paired) < 30:
        logger.warning(
            "Only %d dual-annotated rows available — κ would be unreliable. "
            "Skipping agreement check.", len(paired),
        )
        return None

    kappa = cohens_kappa(
        paired["annotator_a"].tolist(),
        paired["annotator_b"].tolist(),
        labels=CANONICAL_LABELS,
    )
    logger.info("Cohen's κ over %d paired samples = %.3f", len(paired), kappa)

    if kappa < threshold:
        logger.warning(
            "κ=%.3f is BELOW the %.2f threshold — re-train annotators "
            "or revise the labeling guideline before training.",
            kappa, threshold,
        )
    else:
        logger.info(
            "κ=%.3f ≥ %.2f → annotation quality is acceptable.",
            kappa, threshold,
        )
    return kappa


# =========================================================================== #
# Main pipeline
# =========================================================================== #
def main(args: argparse.Namespace) -> None:
    """Run the full data-prep pipeline end-to-end."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. Load each source into the canonical schema -----
    parts: List[pd.DataFrame] = []
    if args.uit_vsmec:
        parts.append(load_uit_vsmec(Path(args.uit_vsmec)))
    if args.phonlp:
        parts.append(load_phonlp(Path(args.phonlp)))
    if args.crawled:
        parts.append(load_crawled(Path(args.crawled)))

    if not parts or all(p.empty for p in parts):
        logger.error("No source datasets loaded — aborting.")
        sys.exit(1)

    # ----- 2. Annotation-quality gate (crawled data only) -----
    crawled = next(
        (p for p in parts if not p.empty and (p["source"] == "crawled").any()),
        None,
    )
    if crawled is not None:
        report_annotator_agreement(crawled, threshold=args.kappa_threshold)

    # ----- 3. Concatenate sources -----
    merged = pd.concat(parts, ignore_index=True, sort=False)
    n_concat = len(merged)
    logger.info("Merged dataset: %d rows total (pre-clean).", n_concat)

    # ----- 4. Teencode normalization (BEFORE dedupe to catch teencode dupes) -----
    # Critical ordering: e.g. "k bik" and "ko biết" are *different* raw strings
    # but BOTH normalize to "không biết". If we deduped first, both copies would
    # survive and could leak across train/val/test splits.
    logger.info("Applying TeencodeNormalizer to %d texts...", len(merged))
    normalizer = TeencodeNormalizer(teencode_dict_path=args.teencode_dict)
    merged["text"] = normalizer.transform(merged["text"])

    # Drop rows that became empty after normalization (e.g. emoji-only posts).
    n_pre_empty = len(merged)
    merged = merged[merged["text"].str.len() > 0].reset_index(drop=True)
    if len(merged) < n_pre_empty:
        logger.info(
            "  Dropped %d rows that normalized to empty strings.",
            n_pre_empty - len(merged),
        )

    # ----- 5. Dedupe AFTER normalization to prevent cross-split leakage -----
    n_pre_dedupe = len(merged)
    merged = (
        merged.drop_duplicates(subset=["text"], keep="first")
        .reset_index(drop=True)
    )
    n_dropped = n_pre_dedupe - len(merged)
    if n_dropped:
        logger.info(
            "  Dropped %d post-normalize duplicates (%d → %d).",
            n_dropped, n_pre_dedupe, len(merged),
        )
    logger.info("Per-source counts:\n%s", merged["source"].value_counts())
    logger.info("Per-label counts:\n%s", merged["label"].value_counts())

    # ----- 6. Stratified 70/15/15 split -----
    logger.info("Stratified split (70/15/15) on label column...")
    train_df, val_df, test_df = stratified_split(
        merged,
        label_column="label",
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        seed=args.seed,
    )
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split["label"].value_counts(normalize=True).round(3).to_dict()
        logger.info("  %s: n=%d, label_dist=%s", name, len(split), dist)

    # ----- 7. Persist as Parquet (compact + dtype-stable) -----
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    logger.info("Wrote splits to %s/", output_dir)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integrate UIT-VSMEC + PhoNLP + crawled data, "
                    "run κ check, normalize teencode, and emit train/val/test "
                    "Parquet files."
    )
    p.add_argument("--uit-vsmec", type=str, default=None,
                   help="Path to UIT-VSMEC CSV.")
    p.add_argument("--phonlp", type=str, default=None,
                   help="Path to PhoNLP sentiment CSV/TSV.")
    p.add_argument("--crawled", type=str, default=None,
                   help="Path to crawled .xlsx file.")
    p.add_argument("--output-dir", type=str, default="data/processed",
                   help="Where to write train/val/test Parquet files.")
    p.add_argument("--teencode-dict", type=str, default=None,
                   help="Optional JSON file extending the embedded teencode map.")
    p.add_argument("--kappa-threshold", type=float, default=0.6,
                   help="Minimum acceptable Cohen's κ (Landis & Koch).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
