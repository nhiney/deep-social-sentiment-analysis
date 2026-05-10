"""Streamlit demo app for the Late Fusion sentiment / emotion classifier.

Run from the project root::

    streamlit run app/app.py

The app exposes two modes via tabs:

1. **Single inference** — paste a post, optionally tweak behavior signals,
   get a prediction with a LIME-based word-level explanation.
2. **Batch processing** — upload a CSV, run inference at scale, see
   distribution charts and (when applicable) per-category drill-downs.

Layout: a configurable sidebar (checkpoint path, device, LIME budget) and
two-tab main panel — clean separation of concerns delegated to
:mod:`app.components`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

# Project-root resolution so ``streamlit run app/app.py`` works regardless
# of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.components import (  # noqa: E402
    render_batch_tab,
    render_sidebar,
    render_single_tab,
)
from app.explainer import TextExplainer       # noqa: E402
from app.inference import LateFusionPredictor  # noqa: E402

logger = logging.getLogger("app")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Class names matching the 7-way Ekman+Neutral schema used during training.
CLASS_NAMES = [
    "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral",
]


# =========================================================================== #
# Page setup
# =========================================================================== #
def _setup_page() -> None:
    """Wide layout, custom title, light branding."""
    st.set_page_config(
        page_title="Deep Social Sentiment — Demo",
        page_icon=":speech_balloon:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Top banner / header.
    st.title("Deep Social Sentiment Analysis")
    st.markdown(
        "**Late Fusion** of `XLM-R` (text branch) and `FT-Transformer` "
        "(behavior branch), with a Vietnamese teencode normalizer in the "
        "preprocessing pipeline. Use the sidebar to point at a trained "
        "checkpoint, then explore single-text inference (with LIME "
        "explanations) or run batch analytics over a CSV."
    )
    st.divider()


# =========================================================================== #
# Resource loading (cached across reruns)
# =========================================================================== #
@st.cache_resource(show_spinner="Loading model checkpoint...")
def _load_predictor(
    checkpoint_dir: str,
    device: str,
    max_length: int,
    apply_normalizer: bool,
) -> LateFusionPredictor:
    """Construct a :class:`LateFusionPredictor` once per (config) tuple."""
    return LateFusionPredictor(
        checkpoint_dir=checkpoint_dir,
        class_names=CLASS_NAMES,
        device=device,
        max_length=max_length,
        apply_normalizer=apply_normalizer,
    )


@st.cache_resource(show_spinner=False)
def _load_explainer(
    _predictor: LateFusionPredictor,
    num_samples: int,
) -> TextExplainer:
    """Build a LIME explainer bound to ``_predictor``'s prediction function.

    The leading underscore on ``_predictor`` is a Streamlit cache convention
    that opts the predictor object out of hashing (it isn't hash-friendly).
    """
    return TextExplainer(
        class_names=_predictor.class_names,
        predict_proba_fn=_predictor.predict_proba_for_lime,
        num_samples=num_samples,
        num_features=10,
        bow=False,
    )


def _safe_load_resources(
    cfg: dict,
) -> Tuple[Optional[LateFusionPredictor], Optional[TextExplainer]]:
    """Load model + explainer with explicit error handling for the UI.

    Returns ``(None, None)`` and renders an actionable error banner if
    anything goes wrong (missing checkpoint, corrupted file, OOM, ...).
    """
    ckpt = Path(cfg["checkpoint_dir"])
    if not ckpt.exists():
        st.error(
            f"Checkpoint directory not found: `{ckpt}`. "
            "Train a model first (e.g. `python -m src.train --config "
            "configs/config.yaml`) or update the path in the sidebar."
        )
        return None, None

    try:
        predictor = _load_predictor(
            checkpoint_dir=str(ckpt),
            device=cfg["device"],
            max_length=cfg["max_length"],
            apply_normalizer=cfg["apply_normalizer"],
        )
    except FileNotFoundError as e:
        st.error(f"Checkpoint files missing or unreadable: {e}")
        return None, None
    except RuntimeError as e:
        # Common causes: state-dict / config mismatch, CUDA OOM, AMP issues.
        st.error(f"Runtime error while loading model: {e}")
        return None, None
    except Exception as e:  # noqa: BLE001
        st.error(f"Unexpected error loading model: {e}")
        logger.exception("Failed to load predictor.")
        return None, None

    try:
        explainer = _load_explainer(predictor, cfg["num_lime_samples"])
    except Exception as e:  # noqa: BLE001
        # Non-fatal — we can still serve predictions without LIME.
        st.warning(f"LIME explainer unavailable: {e}")
        logger.exception("Failed to build LIME explainer.")
        explainer = None

    # Quick health banner so reviewers know the model is live.
    has_tab = bool(predictor.num_cols or predictor.cat_cols)
    branch_str = "Text-only" if not has_tab else "Text + Tabular fusion"
    st.success(
        f"Model loaded — {branch_str} | "
        f"text encoder: `{predictor.model.config.text_model_name}` | "
        f"device: `{predictor.device}`"
    )
    return predictor, explainer


# =========================================================================== #
# Main
# =========================================================================== #
def main() -> None:
    _setup_page()

    cfg = render_sidebar()
    predictor, explainer = _safe_load_resources(cfg)

    tab_single, tab_batch, tab_about = st.tabs(
        ["Single inference", "Batch processing", "About"]
    )

    with tab_single:
        if predictor is None:
            st.info("Load a valid checkpoint to enable single-text inference.")
        else:
            try:
                render_single_tab(predictor, explainer)
            except Exception as e:  # noqa: BLE001
                # Last-line-of-defense: keep the app responsive on any error.
                st.error(f"Single inference tab crashed: {e}")
                logger.exception("Single tab crashed.")

    with tab_batch:
        if predictor is None:
            st.info("Load a valid checkpoint to enable batch processing.")
        else:
            try:
                render_batch_tab(predictor)
            except Exception as e:  # noqa: BLE001
                st.error(f"Batch tab crashed: {e}")
                logger.exception("Batch tab crashed.")

    with tab_about:
        _render_about_tab()


def _render_about_tab() -> None:
    """Static about page — pipeline overview + dataset description."""
    st.subheader("Project overview")
    st.markdown(
        """
**Pipeline** (`Late Fusion`)
1. **Teencode normalizer** — rule-based dictionary + emoji → `[TOKEN]`
   replacement (`khum → không`, `j → gì`, `😊 → [SMILE]`).
2. **Text branch** — `XLM-R` encoder, `[CLS]` pooling → `h_text`.
3. **Tabular branch** — `FT-Transformer` over behavior features
   (text length, exclamation count, emoji count, code-switching, ...) → `h_tab`.
4. **Fusion head** — `[h_text ⊕ h_tab] → Linear → ReLU → Dropout → Linear → softmax`.

**Labels (7-way Ekman + Neutral)**
- `joy`, `sadness`, `anger`, `fear`, `disgust`, `surprise`, `neutral`

**Headline metric** — F1-Macro (each class weighted equally regardless of
the class imbalance native to social-media corpora).

**Explainability** — LIME over the full pipeline; tokens that pushed the
prediction toward the chosen class show in green, tokens pushing against
show in red. Magnitude is encoded by opacity.
        """
    )


if __name__ == "__main__":
    main()
