"""Streamlit UI components for the demo app.

Each top-level function renders one section of the UI. Components are pure
in the sense that they take a :class:`LateFusionPredictor` (and optionally a
:class:`TextExplainer`) and a Streamlit container — they don't touch global
state. This keeps ``app.py`` a thin orchestration layer.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.explainer import ExplanationResult, TextExplainer
from app.inference import (
    DEFAULT_CAT_COLS,
    DEFAULT_NUM_COLS,
    LateFusionPredictor,
    make_text_derived_features,
)

logger = logging.getLogger(__name__)


# =========================================================================== #
# Sidebar
# =========================================================================== #
def render_sidebar(default_ckpt: str = "models/best_model") -> Dict[str, Any]:
    """Render the sidebar configuration panel.

    Returns
    -------
    dict
        ``{"checkpoint_dir", "device", "max_length", "num_lime_samples",
           "apply_normalizer"}``.
    """
    with st.sidebar:
        st.header("Configuration")

        ckpt = st.text_input(
            "Checkpoint directory",
            value=default_ckpt,
            help=("Directory produced by `LateFusionModel.save_pretrained` "
                  "(typically `models/best_model` or `models/ablation/exp3_full_fusion/best_model`)."),
        )

        device = st.selectbox(
            "Device",
            options=["auto", "cpu", "cuda", "mps"],
            index=0,
            help="`auto` picks CUDA → MPS → CPU in that order.",
        )

        max_length = st.slider(
            "Max tokens",
            min_value=32, max_value=256, value=128, step=16,
            help="Tokenizer truncation length. 128 is a good fit for short posts.",
        )

        apply_normalizer = st.toggle(
            "Apply teencode normalization",
            value=True,
            help="Recommended ON — matches the training-time pipeline.",
        )

        st.divider()
        st.subheader("Explainability")
        num_lime_samples = st.slider(
            "LIME samples",
            min_value=50, max_value=1000, value=200, step=50,
            help="More samples → more stable explanations but slower (~2× at 400).",
        )

        st.divider()
        st.caption("Project: Deep Social Sentiment Analysis — Late Fusion XLM-R + FT-Transformer.")

    return dict(
        checkpoint_dir=ckpt,
        device=device,
        max_length=max_length,
        num_lime_samples=num_lime_samples,
        apply_normalizer=apply_normalizer,
    )


# =========================================================================== #
# Single-inference tab
# =========================================================================== #
def render_single_tab(
    predictor: LateFusionPredictor,
    explainer: Optional[TextExplainer],
) -> None:
    """Render the single-inference tab with optional LIME highlighting."""
    st.subheader("Single-text inference")
    st.markdown(
        "Paste a Vietnamese (or code-switched) social-media post. "
        "Optionally tweak the **behavior signals** to simulate different "
        "user profiles, then click **Predict**."
    )

    text = st.text_area(
        "Input text",
        value="Hôm nay vui quá luôn 😊, mọi việc đều suôn sẻ!",
        height=120,
        max_chars=2000,
    )

    # Behavior-signal sliders are auto-populated from the typed text but
    # remain editable so reviewers can stress-test the tabular branch.
    overrides = _render_behavior_panel(text, predictor)

    target_label = st.selectbox(
        "Explain prediction for",
        options=["(top prediction)"] + list(predictor.class_names),
        help="Which class LIME should attribute against.",
    )
    target = None if target_label == "(top prediction)" else target_label

    if not st.button("Predict", type="primary"):
        return

    # ---- Inference ----
    try:
        with st.spinner("Running model..."):
            result = predictor.predict([text], tabular_overrides=overrides)[0]
    except Exception as e:           # noqa: BLE001
        st.error(f"Inference failed: {e}")
        logger.exception("Single inference failed.")
        return

    _render_prediction_card(result, predictor.class_names)
    _render_probability_bar(result["probs"])

    # ---- LIME explanation ----
    if explainer is None:
        st.info("LIME explainer is disabled (no checkpoint loaded).")
        return

    with st.spinner(f"Computing LIME explanation ({explainer.num_samples} samples)..."):
        try:
            explanation = explainer.explain(text, target_label=target)
        except Exception as e:        # noqa: BLE001
            st.error(f"LIME failed: {e}")
            logger.exception("LIME explain_instance failed.")
            return

    _render_explanation(explanation, predictor.class_names)


def _render_behavior_panel(
    text: str,
    predictor: LateFusionPredictor,
) -> Dict[str, Any]:
    """Render editable behavior sliders. Returns overrides for non-default values."""
    if not predictor.tab_pp or not (predictor.num_cols + predictor.cat_cols):
        # Text-only checkpoint — no panel to render.
        return {}

    # Auto-derive defaults from the current text — normalize first so the
    # features match what the model was trained on (e.g. ``n_emoji_token``
    # counts ``[SMILE]`` tokens, not raw 😊 codepoints).
    if predictor.normalizer is not None:
        norm_text = predictor.normalizer(text or "")
    else:
        norm_text = text or ""
    derived = make_text_derived_features(pd.Series([norm_text])).iloc[0].to_dict()

    overrides: Dict[str, Any] = {}
    with st.expander("Behavior signals (auto-derived from text — editable)", expanded=False):
        st.caption(
            "Numeric sliders default to features extracted from the text above. "
            "Override them to simulate different user behavior patterns."
        )

        cols = st.columns(3)
        # ---- Numerical features ----
        for i, name in enumerate(predictor.num_cols):
            with cols[i % 3]:
                default = float(derived.get(name, 0.0))
                # Reasonable upper bound = 4× the default (or a hard min).
                upper = max(20.0, default * 4)
                v = st.slider(
                    name, min_value=0.0, max_value=upper,
                    value=default, step=1.0, key=f"num_{name}",
                )
                if not np.isclose(v, default):
                    overrides[name] = v

        # ---- Categorical features ----
        for i, name in enumerate(predictor.cat_cols):
            with cols[i % 3]:
                vocab_keys = list(predictor.tab_pp.cat_vocab_[name].keys())
                default = derived.get(name, vocab_keys[0])
                idx = vocab_keys.index(default) if default in vocab_keys else 0
                v = st.selectbox(name, options=vocab_keys, index=idx,
                                 key=f"cat_{name}")
                if v != default:
                    overrides[name] = v
    return overrides


def _render_prediction_card(
    result: Dict[str, Any],
    class_names: Sequence[str],   # noqa: ARG001 — kept for API symmetry
) -> None:
    """Headline label + confidence."""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted emotion", result["label"])
    with col2:
        st.metric("Confidence", f"{result['confidence']*100:.1f}%")


def _render_probability_bar(probs: Dict[str, float]) -> None:
    """Horizontal Plotly bar of per-class probabilities."""
    df = (
        pd.DataFrame({"emotion": list(probs.keys()), "probability": list(probs.values())})
        .sort_values("probability", ascending=True)
    )
    fig = px.bar(
        df, x="probability", y="emotion", orientation="h",
        text=df["probability"].apply(lambda v: f"{v*100:.1f}%"),
        color="probability", color_continuous_scale="Blues",
    )
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        xaxis=dict(range=[0, 1], tickformat=".0%", title=""),
        yaxis=dict(title=""),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def _render_explanation(
    explanation: ExplanationResult,
    class_names: Sequence[str],     # noqa: ARG001
) -> None:
    """Render LIME tokens + colored highlight."""
    st.markdown("### Explainability (LIME)")
    st.caption(
        f"Attribution against **{explanation.label}** "
        f"(model confidence: {explanation.confidence*100:.1f}%, "
        f"computed over {explanation.n_samples} perturbations)."
    )

    # 1) Inline highlight.
    st.markdown(explanation.highlight_html, unsafe_allow_html=True)

    # 2) Token table.
    if explanation.tokens:
        df = pd.DataFrame(explanation.tokens, columns=["token", "weight"])
        df["direction"] = df["weight"].apply(
            lambda w: "supports" if w > 0 else "opposes"
        )
        df = df.reindex(df["weight"].abs().sort_values(ascending=False).index)
        st.dataframe(
            df.reset_index(drop=True), use_container_width=True, hide_index=True,
        )


# =========================================================================== #
# Batch-processing tab
# =========================================================================== #
def render_batch_tab(predictor: LateFusionPredictor) -> None:
    """Render the CSV upload → batch prediction → dashboard tab."""
    st.subheader("Batch processing dashboard")
    st.markdown(
        "Upload a CSV file with a **`text`** column (other columns are kept "
        "and shown alongside predictions). An optional **`category`** column "
        "(or `domain` / `topic`) unlocks the per-category drill-down chart."
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    with col_b:
        batch_size = st.number_input(
            "Batch size", min_value=1, max_value=128, value=16, step=1,
        )

    if uploaded is None:
        st.info("Awaiting CSV upload...")
        return

    # ---- Robust CSV loading ----
    df = _safe_read_csv(uploaded)
    if df is None:
        return

    if "text" not in df.columns:
        st.error(
            f"CSV must contain a 'text' column. Found: {list(df.columns)}"
        )
        return

    n_rows = len(df)
    if n_rows == 0:
        st.warning("CSV is empty.")
        return
    st.success(f"Loaded {n_rows:,} rows × {len(df.columns)} columns.")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)

    if not st.button("Run batch inference", type="primary"):
        return

    # ---- Batched inference with progress feedback ----
    try:
        preds_df = _run_batch_inference(predictor, df, int(batch_size))
    except Exception as e:           # noqa: BLE001
        st.error(f"Batch inference failed: {e}")
        logger.exception("Batch inference failed.")
        return

    # ---- Visualizations ----
    st.markdown("### Results")
    st.dataframe(preds_df.head(20), use_container_width=True, hide_index=True)

    st.download_button(
        "Download predictions CSV",
        data=preds_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

    _render_batch_charts(preds_df, predictor.class_names)


def _safe_read_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Read CSV with explicit error reporting (no app crash on bad files)."""
    try:
        # Try common encodings — Vietnamese files are often utf-8 or utf-8-sig.
        for enc in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                return df
            except UnicodeDecodeError:
                continue
        st.error("Could not decode the CSV — try saving it as UTF-8.")
        return None
    except pd.errors.EmptyDataError:
        st.error("Uploaded file is empty.")
    except pd.errors.ParserError as e:
        st.error(f"Failed to parse CSV: {e}")
    except Exception as e:           # noqa: BLE001
        st.error(f"Unexpected error reading file: {e}")
        logger.exception("Failed to read uploaded CSV.")
    return None


def _run_batch_inference(
    predictor: LateFusionPredictor,
    df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    """Run inference in mini-batches with a Streamlit progress bar."""
    texts = df["text"].astype(str).fillna("").tolist()
    n = len(texts)

    progress = st.progress(0.0, text="Running model...")
    all_preds: List[Dict[str, Any]] = []
    for start in range(0, n, batch_size):
        chunk = texts[start:start + batch_size]
        all_preds.extend(predictor.predict(chunk))
        progress.progress(min(1.0, (start + len(chunk)) / n))
    progress.empty()

    # Flatten into columns for display & export.
    out = df.copy()
    out["predicted_label"] = [p["label"] for p in all_preds]
    out["confidence"] = [p["confidence"] for p in all_preds]
    # Add per-class probability columns for downstream analysis.
    for cls in predictor.class_names:
        out[f"prob_{cls}"] = [p["probs"][cls] for p in all_preds]
    return out


def _render_batch_charts(
    df: pd.DataFrame,
    class_names: Sequence[str],
) -> None:
    """Pie chart + per-category bar chart + emotion-confidence violin."""
    st.markdown("### Distribution")

    cols = st.columns(2)

    # 1) Pie chart — overall emotion distribution.
    with cols[0]:
        counts = (
            df["predicted_label"]
            .value_counts()
            .reindex(class_names, fill_value=0)
            .reset_index()
        )
        counts.columns = ["emotion", "count"]
        fig_pie = px.pie(
            counts, values="count", names="emotion",
            title="Emotion mix",
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 2) Bar chart — count per emotion (with negative emotions grouped).
    with cols[1]:
        negative_set = {"sadness", "anger", "fear", "disgust"}
        counts["polarity"] = counts["emotion"].apply(
            lambda x: "negative" if x in negative_set
            else ("positive" if x == "joy" else "neutral")
        )
        fig_bar = px.bar(
            counts, x="emotion", y="count", color="polarity",
            color_discrete_map={
                "negative": "#ef4444",
                "neutral": "#94a3b8",
                "positive": "#22c55e",
            },
            title="Counts per emotion (colored by polarity)",
        )
        fig_bar.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    # 3) Per-category drill-down — only when a category-like column exists.
    cat_col = _find_category_column(df)
    if cat_col is not None:
        st.markdown(f"### Negativity by `{cat_col}`")
        # Aggregate share of negative emotions per category.
        df_neg = df.copy()
        df_neg["is_negative"] = df_neg["predicted_label"].isin(
            {"sadness", "anger", "fear", "disgust"}
        )
        agg = (
            df_neg.groupby(cat_col)["is_negative"]
            .mean()
            .reset_index()
            .rename(columns={"is_negative": "negative_share"})
            .sort_values("negative_share", ascending=False)
        )
        fig_cat = px.bar(
            agg, x=cat_col, y="negative_share",
            text=agg["negative_share"].apply(lambda v: f"{v*100:.0f}%"),
            color="negative_share",
            color_continuous_scale="Reds",
            title=f"Share of negative emotions per {cat_col}",
        )
        fig_cat.update_layout(
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            margin=dict(t=40, b=10, l=10, r=10), height=380,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cat, use_container_width=True)


def _find_category_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first plausible category column for grouping plots."""
    for cand in ("category", "domain", "topic", "lĩnh_vực", "linh_vuc", "source"):
        if cand in df.columns:
            return cand
    return None
