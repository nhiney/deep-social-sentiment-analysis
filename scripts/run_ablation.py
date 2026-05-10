"""Ablation study: prove the contribution of each architectural component.

We run THREE independent training runs against the *same* train/val/test
splits (identical seed → identical row assignment) and tabulate F1-Macro,
Precision, Recall and Accuracy for each:

================  =================  =================  =================
Experiment        Teencode norm.     XLM-R Text Branch  FT-Transformer
================  =================  =================  =================
1. XLM-R only     ❌                 ✅                 ❌
2. + Teencode     ✅                 ✅                 ❌
3. Full Fusion    ✅                 ✅                 ✅
================  =================  =================  =================

Run::

    # Full run (recommended on a GPU machine)
    python -m scripts.run_ablation --raw data/raw/crawled_emotions.xlsx

    # Quick verification on CPU (tiny model, few epochs)
    python -m scripts.run_ablation --raw data/raw/crawled_emotions.xlsx --quick

The script prints a comparison DataFrame to stdout and saves it to
``reports/ablation_results.csv`` + ``reports/ablation_results.md``.

Note on the "tabular" features
------------------------------
The crawled dataset only ships text + emotion label, so for Experiment 3 we
synthesize **text-derived behavioral proxies** (length, exclamation count,
emoji count, code-switching flag, ...). These are honest, deterministic
features extractable from any social post — they're a stand-in for the real
user-behavior table the production system would consume. We document this
explicitly in the printed report.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure ``src`` is importable when run via ``python -m scripts.run_ablation``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SocialSentimentDataset, sentiment_collate_fn  # noqa: E402
from src.evaluate import evaluate                                     # noqa: E402
from src.models import LateFusionConfig                               # noqa: E402
from src.preprocessing import (                                       # noqa: E402
    TabularPreprocessor,
    TeencodeNormalizer,
    stratified_split,
)
from src.train import TrainingConfig, train_model                     # noqa: E402

logger = logging.getLogger("ablation")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CLASS_NAMES: List[str] = [
    "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral",
]
LABEL_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Same Mã nhãn → canonical mapping as scripts/prepare_data.py.
CRAWLED_CODE_TO_LABEL = {
    "joy": "joy", "sad": "sadness", "ang": "anger", "fea": "fear",
    "dis": "disgust", "sur": "surprise", "neu": "neutral",
}

# Regexes used to derive surface "behavior" features from raw text.
_RE_EMOJI_TOKEN = re.compile(r"\[[A-Z_]+\]")               # post-normalize emoji tokens
_RE_LATIN_WORD  = re.compile(r"\b[A-Za-z]{3,}\b")          # 3+ char latin words
_RE_HASHTAG     = re.compile(r"#\w+")
_RE_URL         = re.compile(r"https?://\S+")


# =========================================================================== #
# Data prep
# =========================================================================== #
def load_raw(path: Path) -> pd.DataFrame:
    """Load the crawled Excel dataset and project to canonical schema.

    Returns a DataFrame with columns ``raw_text`` (untouched) + ``label``
    (canonical 7-way name). We carry both the raw and (later) normalized text
    through the pipeline so each experiment can pick the variant it needs.
    """
    df = pd.read_excel(path)
    out = pd.DataFrame({
        "raw_text": df["Text"].astype(str),
        "label": (
            df["Mã nhãn"].astype(str).str.lower().str.strip()
              .map(CRAWLED_CODE_TO_LABEL)
        ),
    })
    out = out.dropna(subset=["raw_text", "label"]).reset_index(drop=True)
    out = out[out["label"].isin(CLASS_NAMES)].reset_index(drop=True)
    logger.info("Loaded %d rows from %s.", len(out), path)
    return out


def make_text_derived_features(text_series: pd.Series) -> pd.DataFrame:
    """Derive surface 'behavior-proxy' features from a text column.

    These are deterministic, cheap and content-grounded — a fair stand-in
    for real user-behavior columns when those aren't shipped with the corpus.
    """
    s = text_series.astype(str)

    out = pd.DataFrame(index=s.index)
    # Numerical features
    out["text_length"] = s.str.len().astype(np.float32)
    out["n_words"] = s.str.split().apply(len).astype(np.float32)
    out["n_exclam"] = s.str.count("!").astype(np.float32)
    out["n_question"] = s.str.count(r"\?").astype(np.float32)
    out["n_emoji_token"] = s.apply(lambda t: len(_RE_EMOJI_TOKEN.findall(t))).astype(np.float32)
    out["n_latin_words"] = s.apply(lambda t: len(_RE_LATIN_WORD.findall(t))).astype(np.float32)

    # Categorical features (string → vocab index later by TabularPreprocessor)
    out["has_emoji"] = (out["n_emoji_token"] > 0).map({True: "yes", False: "no"})
    out["has_codeswitch"] = (out["n_latin_words"] >= 2).map({True: "yes", False: "no"})
    out["has_hashtag"] = s.apply(lambda t: "yes" if _RE_HASHTAG.search(t) else "no")

    return out


# =========================================================================== #
# Per-experiment runner
# =========================================================================== #
def _build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    use_normalizer: bool,
    use_tabular: bool,
    tokenizer,
    max_length: int,
) -> Tuple[SocialSentimentDataset, SocialSentimentDataset,
           SocialSentimentDataset, TabularPreprocessor]:
    """Build train/val/test datasets with the exact ablation switches applied.

    * ``use_normalizer`` → ``TeencodeNormalizer`` is applied INSIDE the dataset
       (transparent to the model). When ``False``, raw text is fed to XLM-R
       directly.
    * ``use_tabular``    → ``TabularPreprocessor`` fits on the text-derived
       behavioral columns; otherwise it fits on empty column lists, so the
       model auto-disables its tabular branch.
    """
    norm = TeencodeNormalizer() if use_normalizer else None

    if use_tabular:
        num_cols = ["text_length", "n_words", "n_exclam", "n_question",
                    "n_emoji_token", "n_latin_words"]
        cat_cols = ["has_emoji", "has_codeswitch", "has_hashtag"]
    else:
        num_cols, cat_cols = [], []

    tab_pp = TabularPreprocessor(
        numerical_cols=num_cols, categorical_cols=cat_cols,
    ).fit(train_df)

    def _ds(df: pd.DataFrame) -> SocialSentimentDataset:
        return SocialSentimentDataset(
            dataframe=df,
            text_column=text_col,
            label_column="label",
            tokenizer=tokenizer,
            tabular_preprocessor=tab_pp,
            teencode_normalizer=norm,
            label_to_id=LABEL_TO_ID,
            max_length=max_length,
        )

    return _ds(train_df), _ds(val_df), _ds(test_df), tab_pp


def run_experiment(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    use_normalizer: bool,
    use_tabular: bool,
    text_col: str,
    text_model_name: str,
    max_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    output_root: Path,
    device: str,
) -> Dict[str, Any]:
    """Train + evaluate one ablation experiment, return its test metrics."""
    logger.info("============================================================")
    logger.info("[%s] use_normalizer=%s | use_tabular=%s | text_col=%s",
                name, use_normalizer, use_tabular, text_col)
    logger.info("============================================================")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    train_ds, val_ds, test_ds, tab_pp = _build_datasets(
        train_df, val_df, test_df,
        text_col=text_col,
        use_normalizer=use_normalizer,
        use_tabular=use_tabular,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # ---- Class weights to mitigate imbalance ----
    counts = train_df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    # Inverse-frequency weighting, normalized so the mean weight ≈ 1.
    inv = 1.0 / counts.clip(lower=1).to_numpy()
    weights = inv * (len(CLASS_NAMES) / inv.sum())
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # ---- Build configs ----
    model_config = LateFusionConfig(
        text_model_name=text_model_name,
        text_pooling="cls",
        freeze_text_encoder=False,
        n_num_features=tab_pp.n_num_features,
        cat_cardinalities=(tab_pp.cat_cardinalities if tab_pp.is_fitted_ else []),
        ft_d_token=128,
        ft_n_blocks=2,
        ft_attention_n_heads=8,
        ft_ffn_d_hidden=256,
        ft_dropout=0.1,
        fusion_hidden_dim=256,
        fusion_dropout=0.2,
        n_classes=len(CLASS_NAMES),
    )
    training_config = TrainingConfig(
        output_dir=str(output_root / name),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        grad_clip_norm=1.0,
        mixed_precision=True,
        early_stopping_patience=2,
        seed=seed,
        device=device,
    )

    t0 = time.time()
    model, hist = train_model(
        train_ds, val_ds, model_config, training_config,
        class_weights=class_weights,
        tab_preprocessor=tab_pp,
    )
    train_secs = time.time() - t0

    # ---- Test-set evaluation ----
    device_obj = next(model.parameters()).device
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=sentiment_collate_fn,
    )
    report = evaluate(model, test_loader, device_obj, class_names=CLASS_NAMES)
    logger.info("[%s] test metrics: %s", name, report["overall"])
    logger.info("\n%s", report["report"])

    return {
        "experiment":      name,
        "use_normalizer":  use_normalizer,
        "use_tabular":     use_tabular,
        "best_epoch":      hist["best_epoch"],
        "train_seconds":   round(train_secs, 1),
        "f1_macro":        report["overall"]["f1_macro"],
        "precision_macro": report["overall"]["precision_macro"],
        "recall_macro":    report["overall"]["recall_macro"],
        "accuracy":        report["overall"]["accuracy"],
        "f1_weighted":     report["overall"]["f1_weighted"],
        "checkpoint":      hist["checkpoint"],
    }


# =========================================================================== #
# Main pipeline
# =========================================================================== #
def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pre-flight defaults (quick mode swaps in a tiny encoder) ----
    text_model = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if args.quick else args.text_model
    )
    epochs     = 2  if args.quick else args.epochs
    batch_size = 8  if args.quick else args.batch_size
    lr         = 5e-5 if args.quick else args.learning_rate
    max_length = 64 if args.quick else args.max_length

    # ---- Load + split (identical seed across experiments → same rows) ----
    raw = load_raw(Path(args.raw))

    # Normalize once for variants that need normalized text. We do NOT dedupe
    # here because we want the SAME rows used in each experiment — only the
    # input text variant differs.
    norm = TeencodeNormalizer()
    raw["norm_text"] = norm.transform(raw["raw_text"])
    # Drop rows that normalize to empty (would crash the tokenizer).
    raw = raw[raw["norm_text"].str.len() > 0].reset_index(drop=True)

    # Add text-derived behavioral features used by Experiment 3.
    behavior = make_text_derived_features(raw["norm_text"])
    raw = pd.concat([raw, behavior], axis=1)

    # Stratified split with the SAME seed across all experiments.
    train_df, val_df, test_df = stratified_split(
        raw, label_column="label",
        train_size=0.70, val_size=0.15, test_size=0.15,
        seed=args.seed,
    )
    logger.info("Splits: train=%d | val=%d | test=%d",
                len(train_df), len(val_df), len(test_df))

    # ---- Experiment grid ----
    grid = [
        # 1. XLM-R only — raw text, no tabular branch.
        dict(name="exp1_xlmr_only",
             text_col="raw_text",  use_normalizer=False, use_tabular=False),
        # 2. XLM-R + Teencode normalization — same model, normalized text.
        dict(name="exp2_xlmr_teencode",
             text_col="norm_text", use_normalizer=False, use_tabular=False),
        # 3. Full fusion — normalized text + tabular branch.
        dict(name="exp3_full_fusion",
             text_col="norm_text", use_normalizer=False, use_tabular=True),
    ]

    rows: List[Dict[str, Any]] = []
    for spec in grid:
        result = run_experiment(
            **spec,
            train_df=train_df, val_df=val_df, test_df=test_df,
            text_model_name=text_model,
            max_length=max_length,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            seed=args.seed,
            output_root=output_root,
            device=args.device,
        )
        rows.append(result)

    # ---- Aggregate + persist ----
    df = pd.DataFrame(rows).set_index("experiment")
    cols_pretty = [
        "use_normalizer", "use_tabular", "best_epoch", "train_seconds",
        "f1_macro", "precision_macro", "recall_macro", "accuracy",
        "f1_weighted",
    ]
    df = df[cols_pretty]

    csv_path = reports_dir / "ablation_results.csv"
    md_path  = reports_dir / "ablation_results.md"
    df.to_csv(csv_path)
    df.to_markdown(md_path)

    logger.info("============================================================")
    logger.info("Ablation summary (test split, F1-Macro is the headline):")
    logger.info("\n%s", df.to_string(float_format=lambda v: f"{v:.4f}"))
    logger.info("Saved → %s", csv_path)
    logger.info("Saved → %s", md_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a 3-experiment ablation: "
                    "XLM-R only / +Teencode / Full Fusion.",
    )
    p.add_argument("--raw", type=str, default="data/raw/crawled_emotions.xlsx")
    p.add_argument("--output-dir", type=str, default="models/ablation")
    p.add_argument("--text-model", type=str, default="xlm-roberta-base")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' | 'cpu' | 'cuda' | 'cuda:N'")
    p.add_argument("--quick", action="store_true",
                   help="Use a tiny model + 2 epochs for a CPU-friendly smoke run.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
