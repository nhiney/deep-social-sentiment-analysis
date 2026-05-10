"""LIME-based explainability for the demo app.

We use ``lime.lime_text.LimeTextExplainer`` because:

* It is **model-agnostic** — works with the full Late Fusion pipeline
  (text branch + tabular branch + fusion head) without us having to
  attribute through specific layers.
* It produces a ``[(token, weight)]`` list that maps cleanly to colored
  HTML for the Streamlit UI.

For each prediction we render two visual artifacts:

1. ``highlight_html`` — the input text with each token wrapped in a span
   colored by its LIME weight (green = pushes prediction toward the
   chosen class, red = pushes against).
2. ``token_table`` — sorted ``DataFrame`` of (token, weight) pairs for
   tabular display.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from lime.lime_text import LimeTextExplainer

logger = logging.getLogger(__name__)


# =========================================================================== #
# Data containers
# =========================================================================== #
@dataclass
class ExplanationResult:
    """Bundle of artifacts produced by :meth:`TextExplainer.explain`."""

    label: str
    label_id: int
    confidence: float
    tokens: List[Tuple[str, float]] = field(default_factory=list)
    highlight_html: str = ""
    n_samples: int = 0


# =========================================================================== #
# Explainer
# =========================================================================== #
class TextExplainer:
    """Wrap LIME with the predict_fn provided by :class:`LateFusionPredictor`.

    Parameters
    ----------
    class_names : sequence of str
        Class names in id order.
    predict_proba_fn : callable
        ``(list[str]) -> np.ndarray[(N, n_classes)]``. Typically
        ``LateFusionPredictor.predict_proba_for_lime``.
    num_samples : int, default=200
        How many perturbations LIME should evaluate per explanation.
        500 is the LIME default; we lower it for demo responsiveness.
    num_features : int, default=10
        Max number of tokens highlighted per explanation.
    bow : bool, default=False
        ``False`` preserves token positions — important for Vietnamese where
        the same token can appear with different sentiment in different
        positions.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        predict_proba_fn: Callable[[Sequence[str]], np.ndarray],
        num_samples: int = 200,
        num_features: int = 10,
        bow: bool = False,
    ) -> None:
        self.class_names = list(class_names)
        self.predict_proba_fn = predict_proba_fn
        self.num_samples = int(num_samples)
        self.num_features = int(num_features)
        self.lime = LimeTextExplainer(
            class_names=self.class_names,
            bow=bow,
            split_expression=r"\s+",   # whitespace split — preserves emojis [TOKEN]
            random_state=42,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def explain(
        self,
        text: str,
        target_label: Optional[str] = None,
    ) -> ExplanationResult:
        """Generate a LIME explanation for ``text``.

        Parameters
        ----------
        text : str
            Input text to explain.
        target_label : str, optional
            Class to attribute against. Defaults to the model's top prediction.

        Returns
        -------
        ExplanationResult
        """
        if not text or not text.strip():
            return ExplanationResult(
                label="(empty)",
                label_id=-1,
                confidence=0.0,
                highlight_html="<em>Empty input — nothing to explain.</em>",
            )

        # 1) Get base probabilities so we know which class to attribute.
        probs = self.predict_proba_fn([text])[0]              # shape (n_classes,)
        if target_label is not None and target_label in self.class_names:
            target_id = self.class_names.index(target_label)
        else:
            target_id = int(np.argmax(probs))

        # 2) LIME explanation. ``labels=(target_id,)`` restricts the (slow)
        #    surrogate-fitting work to the single class we actually display.
        explanation = self.lime.explain_instance(
            text,
            classifier_fn=self.predict_proba_fn,
            labels=(target_id,),
            num_features=self.num_features,
            num_samples=self.num_samples,
        )
        # Returns [(token, weight)] sorted by absolute importance.
        token_weights: List[Tuple[str, float]] = list(
            explanation.as_list(label=target_id)
        )

        return ExplanationResult(
            label=self.class_names[target_id],
            label_id=target_id,
            confidence=float(probs[target_id]),
            tokens=token_weights,
            highlight_html=self._render_highlight_html(text, token_weights),
            n_samples=self.num_samples,
        )

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_highlight_html(
        text: str,
        token_weights: Sequence[Tuple[str, float]],
    ) -> str:
        """Render input text as HTML with each token colored by its weight.

        * Positive weight → green tint (pushes prediction toward the class).
        * Negative weight → red tint (pushes the prediction away).
        * Magnitude → opacity (0 → transparent, max → fully saturated).
        """
        # Build a lookup from token → weight for fast styling.
        lookup: Dict[str, float] = {tok: w for tok, w in token_weights}

        # Normalize magnitudes so the strongest token reaches full opacity.
        max_abs = max((abs(w) for w in lookup.values()), default=1.0) or 1.0

        spans: List[str] = []
        # Whitespace-tokenize the original text the same way LIME did, so
        # token boundaries match the weights we just received.
        for tok in text.split():
            w = lookup.get(tok, 0.0)
            opacity = abs(w) / max_abs
            if w > 0:
                # Green tint — supports the chosen class.
                bg = f"rgba(34, 197, 94, {opacity:.2f})"
            elif w < 0:
                # Red tint — opposes the chosen class.
                bg = f"rgba(239, 68, 68, {opacity:.2f})"
            else:
                bg = "transparent"
            # Escape so user-supplied "<script>" can't break the page.
            spans.append(
                f'<span style="background:{bg}; padding:2px 4px; '
                f'border-radius:4px; margin:1px;">{html.escape(tok)}</span>'
            )

        return (
            '<div style="line-height:2.0; font-size:1.05rem; '
            'font-family:system-ui, sans-serif;">' + " ".join(spans) + "</div>"
        )
