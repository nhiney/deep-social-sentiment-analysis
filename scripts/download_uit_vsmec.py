"""Download UIT-VSMEC dataset and run the full data preparation pipeline.

UIT-VSMEC (Vietnamese Social Media Emotion Corpus) is a publicly available
dataset with ~7000 Facebook posts labeled with 7 Ekman emotions.

Source: https://github.com/uitnlp/UIT-VSMEC

Run from project root::

    # Download only
    python -m scripts.download_uit_vsmec

    # Download + immediately run full prepare_data pipeline
    python -m scripts.download_uit_vsmec --prepare --crawled data/raw/crawled_emotions.xlsx
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_uit_vsmec")

# UIT-VSMEC official raw CSV URLs (train/dev/test splits)
_VSMEC_URLS = {
    "train": "https://raw.githubusercontent.com/uitnlp/UIT-VSMEC/master/data/UIT-VSMEC-TRAIN.csv",
    "dev":   "https://raw.githubusercontent.com/uitnlp/UIT-VSMEC/master/data/UIT-VSMEC-DEV.csv",
    "test":  "https://raw.githubusercontent.com/uitnlp/UIT-VSMEC/master/data/UIT-VSMEC-TEST.csv",
}

# Fallback: full merged file from community mirror / Kaggle-style re-upload
_VSMEC_MERGED_URL = (
    "https://raw.githubusercontent.com/uitnlp/UIT-VSMEC/master/data/UIT-VSMEC.csv"
)


def _download_with_urllib(url: str, dest: Path) -> bool:
    """Download a file via urllib (stdlib, no extra deps)."""
    try:
        import urllib.request
        logger.info("Downloading %s → %s", url, dest)
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:
        logger.warning("urllib download failed: %s", exc)
        return False


def _download_with_requests(url: str, dest: Path) -> bool:
    """Download a file via requests (if installed)."""
    try:
        import requests
        logger.info("Downloading %s → %s (requests)", url, dest)
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as exc:
        logger.warning("requests download failed: %s", exc)
        return False


def download_vsmec(output_dir: Path) -> Optional[Path]:
    """Download and merge UIT-VSMEC splits into a single CSV.

    Tries to download the pre-merged CSV first; if unavailable, downloads
    the three official train/dev/test splits and concatenates them.

    Parameters
    ----------
    output_dir : Path
        Directory where ``UIT-VSMEC.csv`` will be written.

    Returns
    -------
    Path or None
        Path to the merged CSV, or None if all downloads failed.
    """
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dest = output_dir / "UIT-VSMEC.csv"

    if merged_dest.exists():
        logger.info("UIT-VSMEC.csv already exists at %s — skipping download.", merged_dest)
        _verify_vsmec(merged_dest)
        return merged_dest

    # ── Try merged URL first ────────────────────────────────────────────────
    logger.info("Attempting to download pre-merged UIT-VSMEC.csv …")
    success = _download_with_urllib(_VSMEC_MERGED_URL, merged_dest)
    if not success:
        success = _download_with_requests(_VSMEC_MERGED_URL, merged_dest)

    if success and merged_dest.exists() and merged_dest.stat().st_size > 1000:
        logger.info("Merged file downloaded successfully.")
        _verify_vsmec(merged_dest)
        return merged_dest

    # ── Fallback: download splits individually and concat ──────────────────
    logger.info("Merged URL unavailable — downloading train/dev/test splits …")
    dfs = []
    for split, url in _VSMEC_URLS.items():
        split_path = output_dir / f"UIT-VSMEC-{split.upper()}.csv"
        ok = _download_with_urllib(url, split_path)
        if not ok:
            ok = _download_with_requests(url, split_path)
        if ok and split_path.exists():
            try:
                df = pd.read_csv(split_path, encoding="utf-8")
                dfs.append(df)
                logger.info("  %s split: %d rows", split, len(df))
            except Exception as exc:
                logger.warning("  Failed to read %s: %s", split_path, exc)

    if not dfs:
        logger.error(
            "\n"
            "  ╔═══════════════════════════════════════════════════════════════╗\n"
            "  ║  All automatic downloads failed.                              ║\n"
            "  ║                                                               ║\n"
            "  ║  Manual steps:                                                ║\n"
            "  ║  1. Go to: https://github.com/uitnlp/UIT-VSMEC               ║\n"
            "  ║  2. Download the CSV file(s) from the /data folder            ║\n"
            "  ║  3. Place as:  data/raw/UIT-VSMEC.csv                        ║\n"
            "  ║  4. Re-run this script with --prepare to continue pipeline    ║\n"
            "  ╚═══════════════════════════════════════════════════════════════╝\n"
        )
        return None

    merged = pd.concat(dfs, ignore_index=True)
    logger.info("Merged %d splits → %d total rows", len(dfs), len(merged))

    # Normalize column names (some releases use different casing)
    merged.columns = [c.strip() for c in merged.columns]
    col_map = {c: c for c in merged.columns}
    for c in merged.columns:
        if c.lower() == "sentence":
            col_map[c] = "Sentence"
        elif c.lower() in ("emotion", "label"):
            col_map[c] = "Emotion"
    merged = merged.rename(columns=col_map)

    merged.to_csv(merged_dest, index=False, encoding="utf-8")
    logger.info("Saved merged file → %s", merged_dest)
    _verify_vsmec(merged_dest)
    return merged_dest


def _verify_vsmec(path: Path) -> None:
    """Log a quick sanity check on the downloaded file."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
        logger.info(
            "  Verification: %d rows, columns: %s",
            len(df), list(df.columns),
        )
        emotion_col = next(
            (c for c in df.columns if c.lower() in ("emotion", "label")), None
        )
        if emotion_col:
            logger.info(
                "  Label distribution:\n%s",
                df[emotion_col].value_counts().to_string(),
            )
    except Exception as exc:
        logger.warning("Verification failed: %s", exc)


def run_prepare_data(
    vsmec_path: Path,
    crawled: Optional[str],
    pseudo_labeled: Optional[str],
    output_dir: str,
    confidence_threshold: float,
) -> None:
    """Invoke scripts/prepare_data.py with all available sources."""
    cmd = [sys.executable, "-m", "scripts.prepare_data"]

    if crawled:
        cmd += ["--crawled", crawled]

    cmd += ["--uit-vsmec", str(vsmec_path)]

    if pseudo_labeled:
        cmd += ["--pseudo-labeled", pseudo_labeled,
                "--confidence-threshold", str(confidence_threshold)]

    cmd += ["--output-dir", output_dir]

    logger.info("Running data preparation:\n  %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error("prepare_data.py exited with code %d", result.returncode)
        sys.exit(result.returncode)

    logger.info(
        "\n"
        "  ╔══════════════════════════════════════════════════════════╗\n"
        "  ║  Data preparation complete!                              ║\n"
        "  ║                                                          ║\n"
        "  ║  Next step — train the model:                           ║\n"
        "  ║    python -m src.train --config configs/config.yaml     ║\n"
        "  ╚══════════════════════════════════════════════════════════╝\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download UIT-VSMEC and optionally run data preparation pipeline."
    )
    p.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Directory to save UIT-VSMEC.csv (default: data/raw).",
    )
    p.add_argument(
        "--prepare", action="store_true",
        help="After downloading, run scripts/prepare_data.py with all sources.",
    )
    p.add_argument(
        "--crawled", type=str, default="data/raw/crawled_emotions.xlsx",
        help="Path to crawled_emotions.xlsx (used with --prepare).",
    )
    p.add_argument(
        "--pseudo-labeled", type=str, default=None,
        help="Path to pseudo_labeled_apify.csv (used with --prepare, optional).",
    )
    p.add_argument(
        "--processed-dir", type=str, default="data/processed",
        help="Output directory for Parquet splits (used with --prepare).",
    )
    p.add_argument(
        "--confidence-threshold", type=float, default=0.35,
        help="Confidence threshold for pseudo-labeled data (default: 0.35).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Step 1: Download UIT-VSMEC
    vsmec_path = download_vsmec(output_dir)
    if vsmec_path is None:
        sys.exit(1)

    # Step 2: Optionally run full data prep pipeline
    if args.prepare:
        crawled = args.crawled if Path(args.crawled).exists() else None
        if not crawled:
            logger.warning(
                "Crawled dataset not found at %s — proceeding without it.",
                args.crawled,
            )
        pseudo = args.pseudo_labeled
        if pseudo and not Path(pseudo).exists():
            logger.warning(
                "Pseudo-labeled file not found at %s — proceeding without it.",
                pseudo,
            )
            pseudo = None

        run_prepare_data(
            vsmec_path=vsmec_path,
            crawled=crawled,
            pseudo_labeled=pseudo,
            output_dir=args.processed_dir,
            confidence_threshold=args.confidence_threshold,
        )
    else:
        logger.info(
            "\n"
            "  UIT-VSMEC downloaded. To run the full pipeline now:\n\n"
            "    python -m scripts.download_uit_vsmec --prepare \\\n"
            "        --crawled data/raw/crawled_emotions.xlsx \\\n"
            "        --pseudo-labeled data/processed/pseudo_labeled_apify.csv\n\n"
            "  Or run prepare_data.py directly:\n\n"
            "    python -m scripts.prepare_data \\\n"
            "        --crawled        data/raw/crawled_emotions.xlsx \\\n"
            "        --uit-vsmec      data/raw/UIT-VSMEC.csv \\\n"
            "        --pseudo-labeled data/processed/pseudo_labeled_apify.csv \\\n"
            "        --output-dir     data/processed\n"
        )


if __name__ == "__main__":
    main()
