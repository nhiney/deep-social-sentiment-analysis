"""Inference wrapper around :class:`~src.models.LateFusionModel`.

Encapsulates everything an interactive UI needs to call the model:

* model + tokenizer + teencode normalizer + tabular preprocessor
* device resolution (cuda / cpu / mps)
* a low-level :meth:`predict_proba` consumed by both the UI and the LIME
  explainer (LIME calls it ~hundreds of times per explanation)

Designed to be **constructed once at app startup** (cache via
``@st.cache_resource``) — re-entrant calls are cheap.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.models import LateFusionModel
from src.preprocessing import TabularPreprocessor, TeencodeNormalizer

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

# Surface "behavior" features derived from the text — mirrors the training-time
# extractor used by ``scripts/run_ablation.py`` so inference matches training.
_RE_EMOJI_TOKEN = re.compile(r"\[[A-Z_]+\]")
_RE_LATIN_WORD = re.compile(r"\b[A-Za-z]{3,}\b")
_RE_HASHTAG = re.compile(r"#\w+")


def make_text_derived_features(text_series: pd.Series) -> pd.DataFrame:
    """Re-export of the training-time feature extractor (kept in-sync).

    Column names must exactly match what TabularPreprocessor was fitted on
    during training (see scripts/prepare_data.py).
    """
    s = text_series.astype(str)
    out = pd.DataFrame(index=s.index)
    out["text_length"]    = s.str.len().astype(np.float32)
    out["n_words"]        = s.str.split().apply(len).astype(np.float32)
    out["n_exclamation"]  = s.str.count("!").astype(np.float32)
    out["n_question"]     = s.str.count(r"\?").astype(np.float32)
    out["n_emoji_token"]  = s.apply(lambda t: len(_RE_EMOJI_TOKEN.findall(t))).astype(np.float32)
    out["n_hashtag"]      = s.str.count("#").astype(np.float32)
    out["n_latin_words"]  = s.apply(lambda t: len(_RE_LATIN_WORD.findall(t))).astype(np.float32)
    # Engagement signals — unavailable at inference; filled with NaN so that
    # TabularPreprocessor imputes them using training-set means.
    out["likes"]          = np.nan
    out["comments"]       = np.nan
    out["shares"]         = np.nan
    # Categorical
    out["has_emoji"]      = (out["n_emoji_token"] > 0).map({True: "yes", False: "no"})
    out["has_codeswitch"] = (out["n_latin_words"] >= 2).map({True: "yes", False: "no"})
    out["has_hashtag"]    = s.apply(lambda t: "yes" if _RE_HASHTAG.search(t) else "no")
    out["is_crawled"]     = "0"   # new texts are not from the crawled dataset
    return out


# Column lists must mirror TabularPreprocessor.numerical_cols / categorical_cols.
DEFAULT_NUM_COLS: List[str] = [
    "text_length", "n_words", "n_exclamation", "n_question",
    "n_emoji_token", "n_hashtag", "n_latin_words", "likes", "comments", "shares",
]
DEFAULT_CAT_COLS: List[str] = ["has_emoji", "has_codeswitch", "has_hashtag", "is_crawled"]


# =========================================================================== #
# Predictor
# =========================================================================== #
class LateFusionPredictor:
    """High-level inference façade for the demo app.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory containing ``pytorch_model.bin`` + ``config.json``
        (output of :meth:`LateFusionModel.save_pretrained`). If the directory
        also contains ``tab_preprocessor.joblib``, it is loaded automatically.
    class_names : sequence of str
        Class names in id order — controls label decoding & UI display.
    device : {"auto", "cuda", "cpu", "cuda:0", ...}, default="auto"
    max_length : int, default=128
        Tokenizer truncation length.
    apply_normalizer : bool, default=True
        If True, every text is passed through :class:`TeencodeNormalizer`
        before tokenization (matching the training-time normalization).

    Attributes
    ----------
    model : LateFusionModel
    tokenizer : transformers.PreTrainedTokenizerBase
    tab_pp : TabularPreprocessor or None
    normalizer : TeencodeNormalizer or None
    device : torch.device
    """

    def __init__(
        self,
        checkpoint_dir: PathLike,
        class_names: Sequence[str],
        device: str = "auto",
        max_length: int = 128,
        apply_normalizer: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.class_names = list(class_names)
        self.label_to_id = {n: i for i, n in enumerate(self.class_names)}
        self.max_length = int(max_length)

        self.device = self._resolve_device(device)
        logger.info("LateFusionPredictor: loading from %s on %s",
                    self.checkpoint_dir, self.device)

        # -------- Model + tokenizer --------
        self.model = (
            LateFusionModel
            .from_pretrained(str(self.checkpoint_dir))
            .to(self.device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_model_name)

        # -------- Optional teencode normalizer --------
        self.normalizer: Optional[TeencodeNormalizer] = (
            TeencodeNormalizer() if apply_normalizer else None
        )

        # -------- Tabular preprocessor (optional) --------
        # If the model has zero tabular features we don't need a tab_pp.
        # Otherwise try to load the joblib saved next to the checkpoint.
        self.tab_pp: Optional[TabularPreprocessor] = None
        self._tab_required = (
            self.model.config.n_num_features > 0
            or len(self.model.config.cat_cardinalities) > 0
        )
        if self._tab_required:
            tab_path = self.checkpoint_dir / "tab_preprocessor.joblib"
            if tab_path.exists():
                self.tab_pp = TabularPreprocessor.load(tab_path)
                logger.info("Loaded tab_preprocessor from %s", tab_path)
            else:
                logger.warning(
                    "Model expects tabular features but %s is missing. "
                    "Falling back to a fresh extractor — predictions will be "
                    "less accurate. Re-run training to regenerate it.",
                    tab_path,
                )
                # Best-effort fallback: build an empty-vocab preprocessor so
                # categorical lookups all map to UNK and numericals are zero
                # after standardization. Predictions stay sensible-ish.
                self.tab_pp = self._build_fallback_tab_pp()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_proba(
        self,
        texts: Sequence[str],
        tabular_overrides: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Return softmax probabilities of shape ``(N, n_classes)``.

        Parameters
        ----------
        texts : sequence of str
            Raw social-media texts. Normalization is applied internally if
            ``apply_normalizer=True``.
        tabular_overrides : dict, optional
            Per-column override values applied *after* feature derivation.
            Useful when the UI lets users adjust behavior signals manually.
        """
        # Normalize text first (the model was trained on normalized input).
        if self.normalizer is not None:
            texts = [self.normalizer(t) for t in texts]

        # Build the tabular block — derived features + optional overrides.
        if self._tab_required and self.tab_pp is not None:
            df = pd.DataFrame({"text": texts})
            df = pd.concat([df, make_text_derived_features(df["text"])], axis=1)
            if tabular_overrides:
                # Broadcast overrides across the whole batch.
                for col, val in tabular_overrides.items():
                    if col in df.columns:
                        df[col] = val
            tab = self.tab_pp.transform(df)
            num_t = torch.from_numpy(tab["num_features"]).float().to(self.device)
            cat_t = torch.from_numpy(tab["cat_features"]).long().to(self.device)
        else:
            num_t = cat_t = None

        # Tokenize everything in one batched pass.
        enc = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_features=num_t,
            cat_features=cat_t,
        )
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()

    def predict(
        self,
        texts: Sequence[str],
        tabular_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return per-sample labelled predictions: ``[{label, confidence, probs}]``."""
        probs = self.predict_proba(texts, tabular_overrides=tabular_overrides)
        out: List[Dict[str, Any]] = []
        for row in probs:
            top_id = int(np.argmax(row))
            out.append({
                "label":      self.class_names[top_id],
                "confidence": float(row[top_id]),
                "probs":      {self.class_names[i]: float(row[i])
                               for i in range(len(self.class_names))},
            })
        return out

    def predict_proba_for_lime(self, texts: Sequence[str]) -> np.ndarray:
        """Variant tailored for ``lime.lime_text.LimeTextExplainer``."""
        return self.predict_proba(texts, tabular_overrides=None)

    @torch.no_grad()
    def predict_text_branch_only(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """XLM-R branch only — tabular embedding zeroed out.

        Simulates a text-only model on the fusion model's weights.
        The tabular slot in the concatenation is replaced with zeros so
        the fusion head receives [h_text | 0...0] instead of [h_text | h_tab].
        """
        if self.normalizer is not None:
            texts = [self.normalizer(t) for t in texts]

        enc = self.tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        h_text = self.model.text_branch(enc["input_ids"], enc["attention_mask"])

        if self.model.has_tabular:
            h_tab = torch.zeros(h_text.shape[0], self.model.tab_dim, device=self.device)
            h_fusion = torch.cat([h_text, h_tab], dim=-1)
        else:
            h_fusion = h_text

        logits = self.model.fusion_head(h_fusion)
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()

    @torch.no_grad()
    def predict_tabular_branch_only(
        self,
        texts: Sequence[str],
        tabular_overrides: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """FT-Transformer branch only — text embedding zeroed out.

        Simulates a tabular-only model: text-derived features (length,
        n_words, emoji count …) and optional engagement overrides are
        preserved; the XLM-R embedding slot is replaced with zeros.
        """
        if self.normalizer is not None:
            texts = [self.normalizer(t) for t in texts]

        if self._tab_required and self.tab_pp is not None:
            df = pd.DataFrame({"text": texts})
            df = pd.concat([df, make_text_derived_features(df["text"])], axis=1)
            if tabular_overrides:
                for col, val in tabular_overrides.items():
                    if col in df.columns:
                        df[col] = val
            tab = self.tab_pp.transform(df)
            num_t = torch.from_numpy(tab["num_features"]).float().to(self.device)
            cat_t = torch.from_numpy(tab["cat_features"]).long().to(self.device)
            h_tab = self.model.tabular_branch(num_t, cat_t)
        else:
            raise RuntimeError("Model has no tabular branch.")

        h_text = torch.zeros(h_tab.shape[0], self.model.text_dim, device=self.device)
        h_fusion = torch.cat([h_text, h_tab], dim=-1)
        logits = self.model.fusion_head(h_fusion)
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()

    def _probs_to_results(self, probs: np.ndarray) -> List[Dict[str, Any]]:
        """Convert (N, n_classes) probs array to list of result dicts."""
        out = []
        for row in probs:
            top_id = int(np.argmax(row))
            out.append({
                "label":      self.class_names[top_id],
                "confidence": float(row[top_id]),
                "probs":      {self.class_names[i]: float(row[i])
                               for i in range(len(self.class_names))},
            })
        return out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(spec)

    def _build_fallback_tab_pp(self) -> TabularPreprocessor:
        """Build a TabularPreprocessor with the default schema, fit on noise.

        Used when ``tab_preprocessor.joblib`` is missing — predictions will
        be noticeably worse than with the real fitted statistics, but at
        least the inference path stays functional.
        """
        # Synthesize a one-row DataFrame with all expected columns so .fit
        # has something to operate on.
        sample = pd.DataFrame({
            **{c: [0.0] for c in DEFAULT_NUM_COLS},
            **{c: ["no"] for c in DEFAULT_CAT_COLS},
        })
        return TabularPreprocessor(
            numerical_cols=DEFAULT_NUM_COLS,
            categorical_cols=DEFAULT_CAT_COLS,
        ).fit(sample)

    @property
    def num_cols(self) -> List[str]:
        """Numerical feature names (empty for text-only checkpoints)."""
        return list(self.tab_pp.numerical_cols) if self.tab_pp else []

    @property
    def cat_cols(self) -> List[str]:
        """Categorical feature names (empty for text-only checkpoints)."""
        return list(self.tab_pp.categorical_cols) if self.tab_pp else []
