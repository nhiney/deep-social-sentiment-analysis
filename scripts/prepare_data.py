"""End-to-end data preparation pipeline — multi-source merge + tabular features.

Pipeline
--------
    ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
    │  crawled_emotions    │  │    UIT-VSMEC.csv      │  │ pseudo_labeled_      │
    │  .xlsx               │  │  (Sentence, Emotion)  │  │ apify.csv            │
    │  (2034 rows)         │  │  (~7000 rows)         │  │  (990 rows)          │
    │  is_crawled=0        │  │  is_crawled=0         │  │  is_crawled=1        │
    └──────────┬───────────┘  └──────────┬────────────┘  └──────────┬───────────┘
               │                         │                            │
               ▼                         ▼                            ▼
         load_crawled()           load_uit_vsmec()          load_pseudo_labeled()
               │                         │                            │
               │          ┌──────────────┴────────────────────────────┤
               │          │       _add_text_surface_features()         │
               │          │       _impute_tabular_features(medians)    │
               │          │       (likes / comments / shares)          │
               │          └──────────────────────────────────────────-┘
               └─────────────────────────┬──────────────────────────────┘
                                         ▼
                               TeencodeNormalizer.transform()
                                         │
                                drop_duplicates(subset=["text"])
                                         │
                              stratified_split(70/15/15, seed)
                                         │
                     ┌───────────────────┼───────────────────┐
                     ▼                   ▼                    ▼
               train.parquet        val.parquet         test.parquet

Tabular feature columns in output Parquet files
------------------------------------------------
    Numerical  : text_length, n_words, n_exclamation, n_question,
                 n_emoji_token, n_hashtag, n_latin_words,
                 likes, comments, shares
    Categorical: has_emoji, has_codeswitch, has_hashtag, is_crawled
    Metadata   : source, annotator_a, annotator_b, pseudo_confidence

Run from project root::

    # Minimum (crawled xlsx only — you already have this):
    python -m scripts.prepare_data \\
        --crawled data/raw/crawled_emotions.xlsx

    # Recommended (all three sources):
    python -m scripts.prepare_data \\
        --crawled        data/raw/crawled_emotions.xlsx \\
        --uit-vsmec      data/raw/UIT-VSMEC.csv \\
        --pseudo-labeled data/processed/pseudo_labeled_apify.csv \\
        --confidence-threshold 0.35 \\
        --output-dir     data/processed
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (  # noqa: E402
    TeencodeNormalizer,
    cohens_kappa,
    stratified_split,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_data")

# ---------------------------------------------------------------------------
# Canonical label schema
# ---------------------------------------------------------------------------
CANONICAL_LABELS: List[str] = [
    "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral",
]

EMOTION_TO_SENTIMENT: Dict[str, str] = {
    "joy": "positive", "surprise": "neutral", "neutral": "neutral",
    "sadness": "negative", "anger": "negative",
    "disgust": "negative", "fear": "negative",
}

# UIT-VSMEC emotion column values → canonical
UIT_VSMEC_LABEL_MAP: Dict[str, str] = {
    "Enjoyment": "joy",   "enjoyment": "joy",
    "Sadness":   "sadness", "sadness": "sadness",
    "Anger":     "anger",   "anger":   "anger",
    "Fear":      "fear",    "fear":    "fear",
    "Disgust":   "disgust", "disgust": "disgust",
    "Surprise":  "surprise","surprise":"surprise",
    "Other":     "neutral", "other":   "neutral",
    # Alternative English forms found in some releases
    "Joy":       "joy",     "Neutral": "neutral",
}

PHONLP_LABEL_MAP: Dict[str, str] = {
    "POS": "joy", "positive": "joy", "1": "joy",
    "NEU": "neutral", "neutral": "neutral", "0": "neutral",
    "NEG": "sadness", "negative": "sadness", "-1": "sadness",
}

CRAWLED_LABEL_MAP: Dict[str, str] = {
    # 3-letter codes (Mã nhãn)
    "joy": "joy", "sad": "sadness", "ang": "anger", "fea": "fear",
    "dis": "disgust", "sur": "surprise", "neu": "neutral",
    # Vietnamese long form (Label column)
    "hạnh phúc (enjoyment/joy)": "joy",
    "buồn bã (sadness)":         "sadness",
    "giận dữ (anger)":           "anger",
    "sợ hãi (fear)":             "fear",
    "ghê tởm (disgust)":         "disgust",
    "ngạc nhiên (surprise)":     "surprise",
    "trung tính (neutral)":      "neutral",
}

# ---------------------------------------------------------------------------
# Tabular feature columns produced by this pipeline
# ---------------------------------------------------------------------------
# These must align with what run_ablation.py / app/inference.py / SocialSentimentDataset expect.
TABULAR_NUM_COLS: List[str] = [
    "text_length", "n_words", "n_exclamation", "n_question",
    "n_emoji_token", "n_hashtag", "n_latin_words",
    "likes", "comments", "shares",
]
TABULAR_CAT_COLS: List[str] = ["has_emoji", "has_codeswitch", "has_hashtag", "is_crawled"]

# Pre-compiled regex — compiled once at module load for performance.
_RE_EMOJI_TOKEN = re.compile(r"\[[A-Z_]+\]")
_RE_HASHTAG     = re.compile(r"#\w+")
_RE_LATIN_WORD  = re.compile(r"\b[A-Za-z]{3,}\b")


# ===========================================================================
# Text-surface feature derivation
# ===========================================================================
def _derive_text_surface_features(text_series: pd.Series) -> pd.DataFrame:
    """Compute cheap deterministic features from post text.

    These are the *same* features used in scripts/run_ablation.py and
    app/inference.py — kept in sync so training and inference pipelines agree.

    Parameters
    ----------
    text_series : pd.Series
        Raw or normalized post text (str).

    Returns
    -------
    pd.DataFrame
        One row per input, columns matching TABULAR_NUM_COLS[0:7] + TABULAR_CAT_COLS.
    """
    s = text_series.astype(str)
    out = pd.DataFrame(index=s.index)

    # Numerical
    out["text_length"]   = s.str.len().astype(np.float32)
    out["n_words"]       = s.str.split().apply(len).astype(np.float32)
    out["n_exclamation"] = s.str.count("!").astype(np.float32)
    out["n_question"]    = s.str.count(r"\?").astype(np.float32)
    out["n_emoji_token"] = s.apply(lambda t: len(_RE_EMOJI_TOKEN.findall(t))).astype(np.float32)
    out["n_hashtag"]     = s.apply(lambda t: len(_RE_HASHTAG.findall(t))).astype(np.float32)
    out["n_latin_words"] = s.apply(lambda t: len(_RE_LATIN_WORD.findall(t))).astype(np.float32)

    # Categorical (stored as string — TabularPreprocessor will encode them)
    out["has_emoji"]      = (out["n_emoji_token"] > 0).map({True: "yes", False: "no"})
    out["has_codeswitch"] = (out["n_latin_words"] >= 2).map({True: "yes", False: "no"})
    out["has_hashtag"]    = out["n_hashtag"].gt(0).map({True: "yes", False: "no"})

    return out


def _impute_interaction_features(
    df: pd.DataFrame,
    medians: Dict[str, float],
) -> pd.DataFrame:
    """Fill missing likes / comments / shares with dataset medians.

    For sources that don't carry real interaction data (UIT-VSMEC,
    crawled_emotions.xlsx), we use the median values computed from the
    pseudo-labeled Apify posts, which have authentic Facebook engagement
    numbers. Median (not mean) is used because the distribution is
    heavy-tailed (Pareto-like).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may be missing 'likes', 'comments', 'shares'.
    medians : dict
        {'likes': float, 'comments': float, 'shares': float}

    Returns
    -------
    pd.DataFrame
        Input df with likes / comments / shares columns guaranteed present.
    """
    for col in ("likes", "comments", "shares"):
        if col not in df.columns:
            df[col] = medians.get(col, 0.0)
        else:
            df[col] = df[col].fillna(medians.get(col, 0.0))
        df[col] = df[col].astype(np.float32)
    return df


# ===========================================================================
# Loaders — one per source, each returns a DataFrame with the full schema
# ===========================================================================
def _empty_canonical() -> pd.DataFrame:
    """Return an empty DataFrame matching the full output schema."""
    return pd.DataFrame({col: pd.Series(dtype=str) for col in
                         ["text", "label", "source", "annotator_a", "annotator_b"]})


def _first_present(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"None of {list(candidates)} found in dataframe. "
            f"Columns present: {list(df.columns)}"
        )
    return None


def _drop_unmapped(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Drop rows whose label could not be mapped to CANONICAL_LABELS."""
    n_before = len(df)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df = df[df["label"].isin(CANONICAL_LABELS)].reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped:
        logger.info("  [%s] dropped %d unmapped rows (%d → %d).",
                    source_name, dropped, n_before, len(df))
    return df


def _resolve_consensus(a: pd.Series, b: pd.Series) -> pd.Series:
    """Merge two annotator columns into a consensus label.

    Agreement → that label. Disagreement → NaN (filtered downstream).
    Single non-null → use it. Both null → NaN.
    """
    out: List[Any] = []
    for la, lb in zip(a.tolist(), b.tolist()):
        la_ok = isinstance(la, str) and la in CANONICAL_LABELS
        lb_ok = isinstance(lb, str) and lb in CANONICAL_LABELS
        if la_ok and lb_ok:
            out.append(la if la == lb else np.nan)
        elif la_ok:
            out.append(la)
        elif lb_ok:
            out.append(lb)
        else:
            out.append(np.nan)
    return pd.Series(out, index=a.index, dtype=object)


# ---------------------------------------------------------------------------
# Source: self-crawled Excel (crawled_emotions.xlsx)
# ---------------------------------------------------------------------------
def load_crawled(path: Path) -> pd.DataFrame:
    """Load the self-crawled / manually-annotated Excel dataset.

    Expects columns ``Text`` + (``Mã nhãn`` or ``Label``) and optionally
    two annotator columns for Cohen's κ calculation.

    No interaction data (likes/comments/shares) in this file — those will
    be median-imputed in the merge step.
    """
    if not path.exists():
        logger.warning("Crawled dataset not found at %s — skipping.", path)
        return _empty_canonical()

    logger.info("Loading crawled dataset from %s", path)
    df = pd.read_excel(path)

    text_col    = _first_present(df, ["Text", "text", "content", "post", "Sentence"])
    primary_col = _first_present(df, ["Mã nhãn", "ma_nhan", "label_code"], required=False)
    fallback_col= _first_present(df, ["Label", "label", "emotion"], required=False)
    a_col       = _first_present(df, ["annotator_a", "label_a", "label1"], required=False)
    b_col       = _first_present(df, ["annotator_b", "label_b", "label2"], required=False)

    out = pd.DataFrame({"text": df[text_col].astype(str), "source": "crawled"})

    if a_col:
        out["annotator_a"] = (
            df[a_col].astype(str).str.lower().str.strip().map(CRAWLED_LABEL_MAP)
        )
    else:
        out["annotator_a"] = np.nan

    if b_col:
        out["annotator_b"] = (
            df[b_col].astype(str).str.lower().str.strip().map(CRAWLED_LABEL_MAP)
        )
    else:
        out["annotator_b"] = np.nan

    if a_col and b_col:
        out["label"] = _resolve_consensus(out["annotator_a"], out["annotator_b"])
    elif primary_col:
        out["label"] = (
            df[primary_col].astype(str).str.lower().str.strip().map(CRAWLED_LABEL_MAP)
        )
    elif fallback_col:
        out["label"] = (
            df[fallback_col].astype(str).str.lower().str.strip().map(CRAWLED_LABEL_MAP)
        )
    else:
        raise KeyError(
            "Crawled dataset has no usable label column. "
            "Expected one of: 'Mã nhãn', 'Label', 'annotator_a'."
        )

    out["is_crawled"]      = "0"   # text-only labeled data
    out["pseudo_confidence"] = np.nan

    return _drop_unmapped(out, "crawled")


# ---------------------------------------------------------------------------
# Source: UIT-VSMEC
# ---------------------------------------------------------------------------
def load_uit_vsmec(path: Path) -> pd.DataFrame:
    """Load UIT-VSMEC CSV.

    The official file uses ``Sentence`` and ``Emotion`` columns.
    Interaction features (likes, comments, shares) are absent — they will
    be imputed with dataset medians in the merge step.

    Parameters
    ----------
    path : Path
        Path to ``UIT-VSMEC.csv``.

    Notes
    -----
    UIT-VSMEC labels: Enjoyment, Sadness, Anger, Fear, Disgust, Surprise, Other.
    'Other' maps to our 'neutral' class.
    """
    if not path.exists():
        logger.warning("UIT-VSMEC not found at %s — skipping.", path)
        return _empty_canonical()

    logger.info("Loading UIT-VSMEC from %s", path)
    df = pd.read_csv(path)

    text_col  = _first_present(df, ["Sentence", "sentence", "text", "Text"])
    label_col = _first_present(df, ["Emotion", "emotion", "label", "Label"])

    out = pd.DataFrame({
        "text":            df[text_col].astype(str),
        "label":           df[label_col].astype(str).map(UIT_VSMEC_LABEL_MAP),
        "source":          "uit-vsmec",
        "annotator_a":     np.nan,
        "annotator_b":     np.nan,
        "is_crawled":      "0",
        "pseudo_confidence": np.nan,
    })
    return _drop_unmapped(out, "uit-vsmec")


# ---------------------------------------------------------------------------
# Source: pseudo-labeled Apify posts
# ---------------------------------------------------------------------------
def load_pseudo_labeled(
    path: Path,
    confidence_threshold: float = 0.35,
    require_confident: bool = False,
) -> pd.DataFrame:
    """Load pseudo-labeled Apify Facebook posts produced by pseudo_label_apify.py.

    This is the only source with REAL social-media interaction features
    (likes, comments, shares scraped from Facebook). These genuine values are
    used both for this source's rows AND to compute the imputation medians
    applied to the other sources.

    Parameters
    ----------
    path : Path
        Output of ``scripts/pseudo_label_apify.py``.
    confidence_threshold : float
        Posts below this zero-shot confidence score are excluded.
    require_confident : bool
        If True, only rows with ``pseudo_confident == True`` are kept.
        If False, use ``confidence_threshold`` for fresh filtering.

    Returns
    -------
    pd.DataFrame
        Canonical-schema DataFrame with real interaction features.
    """
    if not path.exists():
        logger.warning("Pseudo-labeled file not found at %s — skipping.", path)
        return _empty_canonical()

    logger.info("Loading pseudo-labeled posts from %s", path)
    df = pd.read_csv(path)

    # Validate label column
    if "label" not in df.columns:
        logger.error("'label' column missing from pseudo-labeled file.")
        return _empty_canonical()

    # Confidence-based filtering
    if "pseudo_confidence" in df.columns:
        if require_confident and "pseudo_confident" in df.columns:
            before = len(df)
            df = df[df["pseudo_confident"].astype(bool)].copy()
            logger.info(
                "  Kept %d / %d confident pseudo-labels (require_confident=True).",
                len(df), before,
            )
        else:
            before = len(df)
            df = df[df["pseudo_confidence"] >= confidence_threshold].copy()
            logger.info(
                "  Kept %d / %d posts above confidence threshold %.2f.",
                len(df), before, confidence_threshold,
            )
    else:
        logger.warning(
            "'pseudo_confidence' column not found — including all %d rows.", len(df)
        )

    if len(df) == 0:
        logger.warning("No pseudo-labeled rows survived the confidence filter.")
        return _empty_canonical()

    out = pd.DataFrame({
        "text":            df["text"].astype(str),
        "label":           df["label"].astype(str),
        "source":          "apify-pseudo",
        "annotator_a":     np.nan,
        "annotator_b":     np.nan,
        "is_crawled":      "1",
        "pseudo_confidence": (
            df["pseudo_confidence"].astype(np.float32)
            if "pseudo_confidence" in df.columns
            else np.nan
        ),
    })

    # Pass through real interaction features (the key advantage of this source).
    for col in ("likes", "comments", "shares", "time_posted",
                "text_length", "n_words", "n_exclamation", "n_question",
                "n_emoji_token", "n_hashtag"):
        if col in df.columns:
            out[col] = df[col].values

    cleaned = _drop_unmapped(out, "apify-pseudo")
    logger.info(
        "  Label distribution after filtering:\n%s",
        cleaned["label"].value_counts().to_string(),
    )
    return cleaned


# ===========================================================================
# Annotation-quality gate
# ===========================================================================
def report_annotator_agreement(
    df: pd.DataFrame,
    threshold: float = 0.6,
) -> Optional[float]:
    """Compute & log Cohen's κ between annotator_a and annotator_b.

    Only rows where BOTH annotators provided a valid label contribute to κ.
    """
    if "annotator_a" not in df.columns or "annotator_b" not in df.columns:
        return None

    paired = df.dropna(subset=["annotator_a", "annotator_b"])
    paired = paired[
        paired["annotator_a"].isin(CANONICAL_LABELS) &
        paired["annotator_b"].isin(CANONICAL_LABELS)
    ]
    if len(paired) < 30:
        logger.warning(
            "Only %d dual-annotated rows — κ would be unreliable. Skipping.",
            len(paired),
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
            "κ=%.3f < %.2f threshold — consider reviewing annotation guidelines.",
            kappa, threshold,
        )
    else:
        logger.info("κ=%.3f ≥ %.2f → annotation quality acceptable.", kappa, threshold)
    return kappa


# ===========================================================================
# Main pipeline
# ===========================================================================
def main(args: argparse.Namespace) -> None:
    """Run the full multi-source data-prep pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load each source ──────────────────────────────────────────────────
    parts: List[pd.DataFrame] = []

    if args.crawled:
        crawled_df = load_crawled(Path(args.crawled))
        if not crawled_df.empty:
            logger.info("  crawled: %d rows", len(crawled_df))
            parts.append(crawled_df)

    if args.uit_vsmec:
        vsmec_df = load_uit_vsmec(Path(args.uit_vsmec))
        if not vsmec_df.empty:
            logger.info("  uit-vsmec: %d rows", len(vsmec_df))
            parts.append(vsmec_df)

    pseudo_df = pd.DataFrame()     # keep reference for median computation
    if args.pseudo_labeled:
        pseudo_df = load_pseudo_labeled(
            Path(args.pseudo_labeled),
            confidence_threshold=args.confidence_threshold,
        )
        if not pseudo_df.empty:
            logger.info("  apify-pseudo: %d rows", len(pseudo_df))
            parts.append(pseudo_df)

    if not parts or all(p.empty for p in parts):
        logger.error("No source datasets loaded — aborting.")
        sys.exit(1)

    # ── 2. Annotation quality gate (crawled source only) ─────────────────────
    crawled_part = next(
        (p for p in parts if not p.empty and "source" in p.columns
         and (p["source"] == "crawled").any()),
        None,
    )
    if crawled_part is not None:
        report_annotator_agreement(crawled_part, threshold=args.kappa_threshold)

    # ── 3. Compute interaction medians from Apify pseudo-labeled data ─────────
    # These medians are the best available prior for imputing missing interaction
    # features in text-only sources (UIT-VSMEC, crawled_emotions.xlsx).
    #
    # Strategy: if pseudo-labeled data is available use its real Facebook
    # engagement numbers; otherwise fall back to zeros (safe but less informative).
    interaction_medians: Dict[str, float] = {}
    if not pseudo_df.empty:
        raw_apify = pd.read_csv(args.pseudo_labeled) if args.pseudo_labeled else pseudo_df
        for col in ("likes", "comments", "shares"):
            if col in raw_apify.columns:
                med = float(raw_apify[col].median())
                interaction_medians[col] = med
                logger.info("  Imputation median [%s] = %.1f", col, med)
    else:
        logger.warning(
            "No pseudo-labeled Apify data — interaction medians will be 0.0. "
            "Run scripts/pseudo_label_apify.py first for better imputation."
        )
        for col in ("likes", "comments", "shares"):
            interaction_medians[col] = 0.0

    # ── 4. Concat, then add tabular features to ALL rows ─────────────────────
    merged = pd.concat(parts, ignore_index=True, sort=False)
    logger.info("Merged (pre-feature): %d rows from %d sources.", len(merged), len(parts))

    # Derive text-surface features from the raw text (BEFORE normalization so
    # features reflect original surface — exclamation marks, emoji, etc.)
    logger.info("Deriving text-surface features…")
    text_feats = _derive_text_surface_features(merged["text"])
    # Only fill columns that are genuinely missing — don't overwrite real values
    # that the pseudo-labeled loader already provided.
    for col in text_feats.columns:
        if col not in merged.columns:
            merged[col] = text_feats[col].values
        else:
            # Fill NaN cells (e.g., rows from sources that lacked these columns)
            missing_mask = merged[col].isna()
            if missing_mask.any():
                merged.loc[missing_mask, col] = text_feats.loc[missing_mask, col].values

    # Impute missing interaction features with dataset medians.
    merged = _impute_interaction_features(merged, interaction_medians)

    # Ensure is_crawled is present for ALL rows.
    if "is_crawled" not in merged.columns:
        merged["is_crawled"] = "0"
    else:
        merged["is_crawled"] = merged["is_crawled"].fillna("0").astype(str)

    # ── 5. Teencode normalization (BEFORE dedupe — catches cross-form dupes) ──
    logger.info("Applying TeencodeNormalizer to %d texts…", len(merged))
    normalizer = TeencodeNormalizer(teencode_dict_path=args.teencode_dict)
    merged["text"] = normalizer.transform(merged["text"])

    n_pre_empty = len(merged)
    merged = merged[merged["text"].str.strip().str.len() > 0].reset_index(drop=True)
    if len(merged) < n_pre_empty:
        logger.info(
            "  Dropped %d rows that normalized to empty strings.",
            n_pre_empty - len(merged),
        )

    # ── 6. Deduplicate AFTER normalization ──────────────────────────────────
    n_pre_dedupe = len(merged)
    merged = merged.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    n_dropped = n_pre_dedupe - len(merged)
    if n_dropped:
        logger.info(
            "  Dropped %d post-normalize duplicates (%d → %d).",
            n_dropped, n_pre_dedupe, len(merged),
        )

    logger.info("Final dataset: %d rows", len(merged))
    logger.info("Per-source counts:\n%s", merged["source"].value_counts().to_string())
    logger.info("Per-label counts:\n%s",  merged["label"].value_counts().to_string())

    # Enforce correct float types on all tabular numericals.
    for col in TABULAR_NUM_COLS:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype(np.float32)

    # ── 7. Stratified 70 / 15 / 15 split ────────────────────────────────────
    logger.info("Stratified split (70/15/15, seed=%d)…", args.seed)
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
        logger.info("  %-5s: n=%d | dist=%s", name, len(split), dist)

    # ── 8. Save as Parquet (compact, dtype-stable, fast to read) ────────────
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = output_dir / f"{name}.parquet"
        split.to_parquet(out_path, index=False)
        logger.info("  Wrote %s → %s (%d rows, %d cols)",
                    name, out_path, len(split), len(split.columns))

    # ── 9. Final summary ─────────────────────────────────────────────────────
    total = len(train_df) + len(val_df) + len(test_df)
    logger.info(
        "\n╔══════════════════════════════════════════════════╗\n"
        "║              DATA PREPARATION COMPLETE           ║\n"
        "╠══════════════════════════════════════════════════╣\n"
        "║  Total samples : %-30d ║\n"
        "║  Train         : %-30d ║\n"
        "║  Val           : %-30d ║\n"
        "║  Test          : %-30d ║\n"
        "║  Tabular cols  : %-30d ║\n"
        "╚══════════════════════════════════════════════════╝",
        total, len(train_df), len(val_df), len(test_df),
        len(TABULAR_NUM_COLS) + len(TABULAR_CAT_COLS),
    )

    # Feature schema report (useful for debugging TabularPreprocessor config)
    available_num = [c for c in TABULAR_NUM_COLS if c in merged.columns]
    available_cat = [c for c in TABULAR_CAT_COLS if c in merged.columns]
    logger.info("Tabular num cols available: %s", available_num)
    logger.info("Tabular cat cols available: %s", available_cat)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Multi-source data preparation: merge crawled Excel + UIT-VSMEC + "
            "pseudo-labeled Apify posts → train/val/test Parquet splits with "
            "full tabular feature schema."
        )
    )
    p.add_argument(
        "--crawled", type=str, default=None,
        help="Path to self-crawled .xlsx (e.g. data/raw/crawled_emotions.xlsx).",
    )
    p.add_argument(
        "--uit-vsmec", type=str, default=None,
        help="Path to UIT-VSMEC CSV (columns: Sentence, Emotion).",
    )
    p.add_argument(
        "--pseudo-labeled", type=str, default=None,
        help="Path to pseudo_labeled_apify.csv (output of scripts/pseudo_label_apify.py).",
    )
    p.add_argument(
        "--phonlp", type=str, default=None,
        help="(Legacy) Path to PhoNLP sentiment CSV/TSV.",
    )
    p.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Directory for train/val/test Parquet output.",
    )
    p.add_argument(
        "--teencode-dict", type=str, default=None,
        help="Optional JSON extending the embedded teencode map.",
    )
    p.add_argument(
        "--confidence-threshold", type=float, default=0.35,
        help="Min zero-shot confidence to include a pseudo-labeled row (default 0.35).",
    )
    p.add_argument(
        "--kappa-threshold", type=float, default=0.6,
        help="Minimum acceptable Cohen's κ for annotation quality gate.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducible splits.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
