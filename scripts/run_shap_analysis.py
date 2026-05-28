"""SHAP attribution analysis for the tabular branch of LateFusionModel.

Strategy
--------
To isolate the *tabular branch contribution* we freeze text at a single
neutral sentence and vary only the numerical tabular features via SHAP
KernelExplainer.  This gives a clean (feature × emotion) attribution
heatmap answering "which behavioral signal matters most for each emotion?"

Run from project root::

    python -m scripts.run_shap_analysis

Output
------
* ``reports/figures/shap_tabular.png``   — mean |SHAP| heatmap (num features × 7 emotions)
* ``reports/figures/shap_summary.png``   — beeswarm summary plot (top 8 features)
* ``reports/shap_values.npy``            — raw SHAP array (n_samples, n_features, n_classes)
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap

from src.models import LateFusionConfig, LateFusionModel
from src.preprocessing import TabularPreprocessor, TeencodeNormalizer
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shap_analysis")

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
CHECKPOINT  = Path("models/best_model")
DATA_PATH   = Path("data/processed/test.parquet")
FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Neutral text used to freeze the text branch while SHAP varies tabular features
NEUTRAL_TEXT = "bài viết này"

# Number of background samples for KernelExplainer (more → more accurate, slower)
# For full Colab run: N_BACKGROUND=30, N_EXPLAIN=80, nsamples=100
N_BACKGROUND = int(os.environ.get("SHAP_BG", "10"))
# Number of test samples to explain
N_EXPLAIN    = int(os.environ.get("SHAP_N", "15"))
RANDOM_SEED  = 42


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model_and_preprocessor(checkpoint_dir: Path, device: str):
    config_path = checkpoint_dir / "config.json"
    import json
    with open(config_path) as f:
        cfg = json.load(f)

    model_cfg = LateFusionConfig(**{
        k: v for k, v in cfg.items()
        if k in LateFusionConfig.__dataclass_fields__
    })
    model = LateFusionModel(model_cfg)
    state = torch.load(checkpoint_dir / "pytorch_model.bin",
                       map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    logger.info("Model loaded from %s on %s", checkpoint_dir, device)

    tab_pp: TabularPreprocessor = joblib.load(checkpoint_dir / "tab_preprocessor.joblib")
    logger.info("TabularPreprocessor: %d num + %d cat features",
                tab_pp.n_num_features, len(tab_pp.categorical_cols))
    return model, tab_pp


# ── Predict wrapper (tabular-only variations) ────────────────────────────────
def make_predict_fn(model, tab_pp, tokenizer, device, num_cols):
    """Return a function: num_array (N, n_num) → proba (N, 7).

    Text is fixed at NEUTRAL_TEXT so SHAP attribution reflects only the
    tabular signal.
    """
    norm = TeencodeNormalizer()
    fixed_text = norm(NEUTRAL_TEXT)

    @torch.no_grad()
    def predict_fn(num_array: np.ndarray) -> np.ndarray:
        n = len(num_array)
        # Build a minimal DataFrame with all expected columns
        df = pd.DataFrame(num_array, columns=num_cols)
        # Fill categorical columns with their most common value (mode)
        for col in tab_pp.categorical_cols:
            if col == "has_emoji":
                df[col] = "no"
            elif col == "has_codeswitch":
                df[col] = "no"
            elif col == "has_hashtag":
                df[col] = "no"
            elif col == "is_crawled":
                df[col] = "1"
            else:
                df[col] = "no"

        tab = tab_pp.transform(df)
        num_t = torch.from_numpy(tab["num_features"]).float().to(device)
        cat_t = torch.from_numpy(tab["cat_features"]).long().to(device)

        # Tokenize the fixed text (broadcast to batch size n)
        enc = tokenizer(
            [fixed_text] * n,
            padding="max_length", truncation=True,
            max_length=128, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_features=num_t,
            cat_features=cat_t,
        )
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    return predict_fn


# ── SHAP computation ──────────────────────────────────────────────────────────
def run_shap(predict_fn, test_num: np.ndarray, num_cols: list):
    np.random.seed(RANDOM_SEED)
    bg_idx  = np.random.choice(len(test_num), size=min(N_BACKGROUND, len(test_num)), replace=False)
    exp_idx = np.random.choice(len(test_num), size=min(N_EXPLAIN,    len(test_num)), replace=False)

    background = test_num[bg_idx]
    explain_X  = test_num[exp_idx]

    logger.info("KernelExplainer: background=%d, explain=%d", len(background), len(explain_X))
    explainer = shap.KernelExplainer(predict_fn, background)

    n_samples = int(os.environ.get("SHAP_SAMPLES", "50"))
    shap_values = explainer.shap_values(explain_X, nsamples=n_samples, silent=True)

    # Normalize to list-of-classes: list[n_classes] of (n_samples, n_features)
    # Older SHAP: already a list. Newer SHAP: single ndarray (n_samples, n_features, n_classes)
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            shap_values = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
        elif shap_values.ndim == 2:
            shap_values = [shap_values]

    logger.info("SHAP done. n_classes=%d, shape per class: %s",
                len(shap_values), np.array(shap_values[0]).shape)
    return shap_values, explain_X


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_heatmap(shap_values, num_cols, out_path: Path):
    """Mean |SHAP| heatmap: rows=features, columns=emotions."""
    # shap_values: list[n_classes] of (n_samples, n_features)
    n_features = len(num_cols)
    n_classes  = len(CLASS_NAMES)

    mean_abs = np.zeros((n_features, n_classes))
    for c, sv in enumerate(shap_values):
        mean_abs[:, c] = np.abs(sv).mean(axis=0)

    # Row-normalise so each feature's max attribution = 1 (better colour contrast)
    row_max = mean_abs.max(axis=1, keepdims=True).clip(min=1e-9)
    norm_abs = mean_abs / row_max

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(norm_abs, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(num_cols, fontsize=10)
    ax.set_xlabel("Emotion Class", fontsize=12)
    ax.set_ylabel("Tabular Feature", fontsize=12)
    ax.set_title(
        "SHAP Attribution — Tabular Branch\n"
        "(Mean |SHAP| per feature per emotion, row-normalised)",
        fontsize=13, fontweight="bold",
    )

    # Annotate cells with raw mean |SHAP| values
    for r in range(n_features):
        for c in range(n_classes):
            val = mean_abs[r, c]
            ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color="black" if norm_abs[r, c] < 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Normalised attribution", shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved heatmap → %s", out_path)


def plot_summary_bar(shap_values, num_cols, out_path: Path):
    """Global feature importance bar chart (mean |SHAP| summed across classes)."""
    total_importance = np.zeros(len(num_cols))
    for sv in shap_values:
        total_importance += np.abs(sv).mean(axis=0)

    order = np.argsort(total_importance)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(num_cols)))
    ax.barh(
        [num_cols[i] for i in order[::-1]],
        total_importance[order[::-1]],
        color=colors,
        edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP| (summed across emotions)", fontsize=11)
    ax.set_title("Tabular Feature Importance (SHAP)\nSum across all 7 emotion classes",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved summary bar → %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Load model
    model, tab_pp = load_model_and_preprocessor(CHECKPOINT, device)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    num_cols = tab_pp.numerical_cols

    # Load test data + preprocess numerical features
    logger.info("Loading test data from %s", DATA_PATH)
    test_df = pd.read_parquet(DATA_PATH)

    # Build the same numerical features the model was trained on
    from app.inference import make_text_derived_features
    # Ensure all num_cols exist in test_df
    derived = make_text_derived_features(
        test_df.get("normalized_text", test_df.get("text", test_df.iloc[:, 0]))
    )
    for col in num_cols:
        if col not in test_df.columns and col in derived.columns:
            test_df[col] = derived[col]
        elif col not in test_df.columns:
            test_df[col] = 0.0

    # Fill NaN with median
    test_df[num_cols] = test_df[num_cols].fillna(test_df[num_cols].median())
    test_num = test_df[num_cols].to_numpy(dtype=np.float32)
    logger.info("Test numerical array: %s", test_num.shape)

    # Build predict fn
    predict_fn = make_predict_fn(model, tab_pp, tokenizer, device, num_cols)

    # Smoke test
    sample_out = predict_fn(test_num[:3])
    assert sample_out.shape == (3, len(CLASS_NAMES)), f"Bad output shape: {sample_out.shape}"
    logger.info("Predict fn smoke test OK: %s", sample_out.shape)

    # Run SHAP
    shap_values, explain_X = run_shap(predict_fn, test_num, num_cols)

    # Save raw values
    raw_path = REPORTS_DIR / "shap_values.npy"
    np.save(raw_path, np.array(shap_values))
    logger.info("Saved raw SHAP values → %s", raw_path)

    # Plots
    plot_heatmap(shap_values, num_cols, FIGURES_DIR / "shap_tabular.png")
    plot_summary_bar(shap_values, num_cols, FIGURES_DIR / "shap_summary.png")

    # Print top features per emotion
    logger.info("\n=== Top 3 features per emotion ===")
    for c, cls in enumerate(CLASS_NAMES):
        sv = np.abs(shap_values[c]).mean(axis=0)
        top3 = np.argsort(sv)[::-1][:3]
        top3_str = " | ".join(f"{num_cols[i]}={sv[i]:.4f}" for i in top3)
        logger.info("  %-9s: %s", cls, top3_str)

    logger.info("\n✅ SHAP analysis complete.")
    logger.info("   reports/figures/shap_tabular.png")
    logger.info("   reports/figures/shap_summary.png")


if __name__ == "__main__":
    main()
