"""Step 1: JSON Data Cleaning & Feature Extraction — Apify Facebook Scrape.

Pipeline
--------
    data/raw/unlabeled_new_posts.json
        └──► load + inspect
        └──► extract core fields: text, likes, comments, shares
        └──► decode post timestamp from Facebook numeric ID in media array
        └──► derive text-based behavioral proxies
        └──► clean & validate
        └──► data/processed/cleaned_unlabeled_posts.csv

Why these choices?
------------------
* ``likes / comments / shares`` are the canonical social-media engagement signals.
  They serve as the **tabular branch** inputs for the FT-Transformer — the whole
  point is to test whether *how people interact* with a post correlates with the
  *emotion expressed* in it.

* ``time_posted`` (hour of day) is a behavioral proxy: anger/venting posts tend
  to appear late at night; joy/sharing posts peak in the morning. We decode it
  from the Facebook numeric post ID embedded in the media array (bits 32–63 of
  a legacy numeric ID encode a Unix timestamp). When the ID is in the newer
  ``pfbid`` format (base64-encoded), we fall back to NaN so downstream code
  can handle it gracefully.

* Text-derived features (``text_length``, ``n_words``, ``n_exclamation``,
  ``n_question``, ``has_hashtag``, ``has_emoji_token``) are cheap deterministic
  features that are 100% reproducible from any post — they act as behavioral
  proxies when real user-metadata is unavailable, exactly matching the design
  of ``scripts/run_ablation.py``.

Run from project root::

    python -m scripts.process_apify_data
    # or with custom paths:
    python -m scripts.process_apify_data \
        --input  data/raw/unlabeled_new_posts.json \
        --output data/processed/cleaned_unlabeled_posts.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is importable when run as ``python scripts/process_apify_data.py``
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("process_apify")

# ---------------------------------------------------------------------------
# Regex helpers (compiled once at module load for performance)
# ---------------------------------------------------------------------------
# Matches the numeric Facebook post/object ID we expect in media[0]["id"].
# Legacy IDs look like: "1545235743634938"
_FB_NUMERIC_ID_RE = re.compile(r"^\d{10,20}$")

# Matches Vietnamese / English emoji-text tokens left by TeencodeNormalizer,
# e.g. "[SMILE]", "[ANGRY]". Counting them gives a rough "emoji density" signal.
_EMOJI_TOKEN_RE = re.compile(r"\[[A-Z_]+\]")

# Matches #hashtags — common in Vietnamese Facebook posts.
_HASHTAG_RE = re.compile(r"#\w+")

# ---------------------------------------------------------------------------
# Facebook ID → timestamp decoder
# ---------------------------------------------------------------------------

def _decode_fb_timestamp(numeric_id: str) -> Optional[int]:
    """Attempt to decode a Unix timestamp (seconds) from a legacy Facebook numeric ID.

    Facebook's legacy object IDs pack a creation timestamp into the upper 32 bits::

        timestamp = numeric_id >> 32        # seconds since Unix epoch

    This works reliably for IDs created before ~2020. Newer IDs use the
    ``pfbid`` base64 scheme which does NOT embed a recoverable timestamp in this
    way, so we return ``None`` for those.

    Parameters
    ----------
    numeric_id : str
        A string that looks like a pure integer (e.g. ``"1545235743634938"``).

    Returns
    -------
    int or None
        Unix timestamp in seconds, or ``None`` if decoding fails / implausible.
    """
    if not _FB_NUMERIC_ID_RE.match(numeric_id):
        return None
    try:
        raw = int(numeric_id)
        ts = raw >> 32
        # Sanity check: timestamps between 2010-01-01 and 2026-12-31.
        # Values outside this range indicate the ID schema is different.
        if 1_262_304_000 <= ts <= 1_798_761_600:
            return int(ts)
    except (ValueError, OverflowError):
        pass
    return None


def _extract_post_timestamp(media_list: Any) -> Optional[int]:
    """Walk the ``media`` array from Apify and return the first decodable timestamp.

    Apify's Facebook Post Scraper may place the post's own numeric ID in
    ``media[0]["id"]`` or inside nested fields. We try the common paths.

    Parameters
    ----------
    media_list : list or NaN
        The raw ``media`` value from the JSON row.

    Returns
    -------
    int or None
        Unix timestamp (seconds), or ``None`` when unavailable.
    """
    if not isinstance(media_list, list):
        return None

    for item in media_list:
        if not isinstance(item, dict):
            continue

        # Path 1: direct "id" field on the media item.
        candidate = str(item.get("id", ""))
        ts = _decode_fb_timestamp(candidate)
        if ts is not None:
            return ts

        # Path 2: "owner" sub-dict (less reliable, but worth trying).
        owner = item.get("owner", {})
        if isinstance(owner, dict):
            ts = _decode_fb_timestamp(str(owner.get("id", "")))
            if ts is not None:
                return ts

    return None


# ---------------------------------------------------------------------------
# Text feature derivation
# ---------------------------------------------------------------------------

def _derive_text_features(text_series: pd.Series) -> pd.DataFrame:
    """Compute cheap, deterministic text-surface features from raw post text.

    These mirror the features used in ``scripts/run_ablation.py`` so that a
    model trained with this script's output can be evaluated consistently.

    Parameters
    ----------
    text_series : pd.Series
        Raw post text (already cast to str, NaN replaced with empty string).

    Returns
    -------
    pd.DataFrame
        One row per input text, columns:
        ``text_length``, ``n_words``, ``n_exclamation``, ``n_question``,
        ``n_emoji_token``, ``n_hashtag``.
    """
    s = text_series.astype(str)
    out = pd.DataFrame(index=s.index)

    out["text_length"]    = s.str.len().astype(np.float32)
    out["n_words"]        = s.str.split().apply(len).astype(np.float32)
    out["n_exclamation"]  = s.str.count("!").astype(np.float32)
    out["n_question"]     = s.str.count(r"\?").astype(np.float32)
    out["n_emoji_token"]  = s.apply(lambda t: len(_EMOJI_TOKEN_RE.findall(t))).astype(np.float32)
    out["n_hashtag"]      = s.apply(lambda t: len(_HASHTAG_RE.findall(t))).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def load_and_clean(json_path: Path) -> pd.DataFrame:
    """Load Apify JSON and return a clean DataFrame ready for downstream use.

    Cleaning decisions (documented so reviewers can reproduce):

    1. **Missing ``text``**: rows where text is NaN or empty string after
       stripping are dropped — an empty post cannot be classified.

    2. **Missing interaction counts** (``likes``, ``comments``, ``shares``):
       filled with ``0``. The missing values represent posts with *no recorded
       interactions*, not corrupt data. Zero is the semantically correct
       imputation here (a post that received 0 reactions was not interacted
       with, not a post we failed to measure).

    3. **``time_posted`` (hour of day)**: decoded from the Facebook numeric ID
       embedded in the media array. When decoding fails (pfbid format or no
       media), the value is ``NaN`` — downstream code in
       ``src/preprocessing.py::TabularPreprocessor`` handles NaN via median
       imputation, so this is safe.

    4. **Duplicates on ``text``**: removed after stripping whitespace. Duplicate
       posts skew the emotion-distribution analysis and can leak across
       train/test splits if this CSV is later merged with labeled data.

    Parameters
    ----------
    json_path : Path
        Path to Apify's raw JSON output file.

    Returns
    -------
    pd.DataFrame
        Columns: ``text``, ``likes``, ``comments``, ``shares``,
        ``time_posted``, ``post_url``, ``text_length``, ``n_words``,
        ``n_exclamation``, ``n_question``, ``n_emoji_token``, ``n_hashtag``.
    """
    logger.info("Loading JSON from %s", json_path)
    raw: List[Dict[str, Any]] = pd.read_json(json_path).to_dict(orient="records")

    # We iterate once to extract the fields we need — faster than multiple
    # DataFrame operations on a nested structure like ``media``.
    rows = []
    for item in raw:
        ts_unix  = _extract_post_timestamp(item.get("media"))
        hour     = (pd.Timestamp(ts_unix, unit="s").hour
                    if ts_unix is not None else np.nan)

        rows.append({
            "text":        item.get("text"),
            "likes":       item.get("likes"),
            "comments":    item.get("comments"),
            "shares":      item.get("shares"),
            "time_posted": hour,          # hour-of-day [0, 23] or NaN
            "post_url":    item.get("url", ""),
        })

    df = pd.DataFrame(rows)
    n_raw = len(df)
    logger.info("Loaded %d raw records.", n_raw)

    # ---- Step 1: Drop rows with no usable text ----
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].notna() & (df["text"] != "") & (df["text"] != "nan")]
    logger.info("After removing empty text: %d rows (dropped %d).", len(df), n_raw - len(df))

    # ---- Step 2: Fill missing interaction counts with 0 ----
    # Rationale: NaN here means "not reported" = 0 interactions observed.
    for col in ("likes", "comments", "shares"):
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].fillna(0)
            logger.info("  Filled %d NaN in '%s' with 0.", missing, col)

    df[["likes", "comments", "shares"]] = df[["likes", "comments", "shares"]].astype(np.int64)

    # ---- Step 3: time_posted — already NaN-safe (filled from ID or NaN) ----
    n_ts_missing = df["time_posted"].isna().sum()
    if n_ts_missing > 0:
        logger.info(
            "  time_posted: %d rows have NaN (pfbid format or no media ID). "
            "Will be imputed by TabularPreprocessor at training time.",
            n_ts_missing,
        )

    # ---- Step 4: Remove duplicate posts (same text content) ----
    n_pre_dedup = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    n_dupes = n_pre_dedup - len(df)
    if n_dupes:
        logger.info("Dropped %d duplicate posts (%d → %d).", n_dupes, n_pre_dedup, len(df))

    # ---- Step 5: Derive text-surface features ----
    text_feats = _derive_text_features(df["text"])
    df = pd.concat([df, text_feats], axis=1)

    # ---- Final summary ----
    logger.info("Clean dataset: %d posts, %d columns.", len(df), len(df.columns))
    logger.info("Interaction stats:\n%s", df[["likes", "comments", "shares"]].describe().to_string())

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean Apify Facebook JSON → processed CSV."
    )
    p.add_argument(
        "--input",  type=str,
        default="data/raw/unlabeled_new_posts.json",
        help="Path to raw Apify JSON file.",
    )
    p.add_argument(
        "--output", type=str,
        default="data/processed/cleaned_unlabeled_posts.csv",
        help="Output path for the cleaned CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = load_and_clean(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved cleaned dataset → %s", output_path)

    # Quick preview for the terminal
    print("\nFirst 5 rows of cleaned dataset:")
    print(df[["text", "likes", "comments", "shares", "time_posted"]].head().to_string())

    # Column dtype summary
    print("\nColumn info:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
