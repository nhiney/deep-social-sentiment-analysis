"""Modeling stack for the social-sentiment / emotion classifier.

This module ships **four** model families so the dissertation can present a
clean story of progressive sophistication:

============================  ====================================================
Model                          Purpose
============================  ====================================================
:class:`TfidfBaseline`         Classical TF-IDF + LogisticRegression (or LinearSVC).
:class:`DnnBaseline`           Small token-bag MLP — same input as TF-IDF baseline.
:class:`TextBranch`            XLM-R feature extractor →  ``h_text  ∈ R^{d_text}``.
:class:`TabularBranch`         FT-Transformer →           ``h_tab   ∈ R^{d_tab}``.
:class:`LateFusionModel`       ``[h_text ⊕ h_tab] → MLP → softmax``.
============================  ====================================================

Architecture
------------
::

    Text  ─►  TextBranch (XLM-R)            ─┐
                                              ├─►  Concat ─► MLP head ─► logits
    Tab.  ─►  TabularBranch (FT-Transformer) ─┘

All branches expose a uniform ``forward`` returning a fixed-size embedding so
:class:`LateFusionModel` can swap them out (ablations, gating, attention fusion).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel

PathLike = Union[str, Path]


# =========================================================================== #
# 0. Config
# =========================================================================== #
@dataclass
class LateFusionConfig:
    """Hyperparameters for the Late Fusion sentiment model.

    Notes
    -----
    Defaults target ``xlm-roberta-base`` (12-layer, ``hidden_size=768``)
    paired with a small FT-Transformer (``d_token=192, n_blocks=3``). All
    sizes are configurable from YAML so ablation studies stay declarative.
    """

    # ---- Text branch ----
    text_model_name: str = "xlm-roberta-base"
    text_pooling: str = "cls"              # cls | mean
    text_hidden_dim: int = -1              # -1 = auto-detect from HF config
    freeze_text_encoder: bool = False

    # ---- Tabular branch ----
    n_num_features: int = 0
    cat_cardinalities: Sequence[int] = field(default_factory=list)
    ft_d_token: int = 192
    ft_n_blocks: int = 3
    ft_attention_n_heads: int = 8
    ft_ffn_d_hidden: int = 256
    ft_dropout: float = 0.1

    # ---- Fusion head ----
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.2
    n_classes: int = 7


# =========================================================================== #
# 1. Baselines (Phase-3 Step 1)
# =========================================================================== #
class TfidfBaseline:
    """TF-IDF (1-2 grams) + LogisticRegression / LinearSVC baseline.

    A deliberately *classical* model — included to demonstrate that bag-of-
    words features hit a ceiling on Vietnamese social text and motivate the
    contextual XLM-R encoder used by the main model.

    Parameters
    ----------
    classifier : {"logreg", "svm"}, default="logreg"
        ``"logreg"`` → ``LogisticRegression`` (predicts probabilities).
        ``"svm"``    → ``LinearSVC`` (decision-function only).
    max_features : int, default=50_000
        Vocab cap for the TF-IDF vectorizer.
    ngram_range : tuple of int, default=(1, 2)
        n-gram range for the TF-IDF vectorizer.
    class_weight : {"balanced", None}, default="balanced"
        Pass-through to the underlying scikit-learn classifier — important
        because the crawled dataset is class-imbalanced.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        ``TfidfVectorizer → classifier`` pipeline (lazy-built in :meth:`fit`).
    """

    def __init__(
        self,
        classifier: str = "logreg",
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        class_weight: Optional[str] = "balanced",
    ) -> None:
        # Defer sklearn imports so the file remains importable even when only
        # the deep-learning subset of requirements is installed.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        if classifier == "logreg":
            # ``n_jobs`` was deprecated for the default solver in sklearn 1.8;
            # ``saga`` is the recommended multinomial solver and uses one core.
            clf = LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
                solver="lbfgs",
            )
        elif classifier == "svm":
            clf = LinearSVC(class_weight=class_weight)
        else:
            raise ValueError(f"Unknown classifier {classifier!r}")

        self.classifier_name = classifier
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,    # log-scale TF — robust to outlier docs
                strip_accents=None,   # KEEP Vietnamese diacritics
                lowercase=False,      # already lowercased by TeencodeNormalizer
            )),
            ("clf", clf),
        ])

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "TfidfBaseline":
        """Fit the TF-IDF + classifier pipeline on training data."""
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Hard label predictions, shape ``(N,)``."""
        return self.pipeline.predict(list(texts))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Softmax-style probabilities, shape ``(N, n_classes)``.

        Falls back to a softmax over decision-function scores for ``LinearSVC``,
        which doesn't expose ``predict_proba`` natively.
        """
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            return self.pipeline.predict_proba(list(texts))

        scores = self.pipeline.decision_function(list(texts))
        # Numerically-stable softmax over the SVM margins — interpretable as
        # pseudo-probabilities (NOT calibrated, but fine for ranking).
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)


class DnnBaseline(nn.Module):
    """Tiny token-bag MLP baseline: ``mean-embedding → MLP → logits``.

    A second baseline that *is* a neural network but ignores word order, to
    isolate the contribution of contextual modeling vs. plain embedding lookup.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary (use the XLM-R tokenizer's vocab).
    embed_dim : int, default=128
        Token embedding dimension.
    hidden_dim : int, default=256
        MLP hidden size.
    n_classes : int
        Output dimension (= number of emotion classes).
    dropout : float, default=0.3
        Dropout applied between hidden layer and classifier.
    pad_token_id : int, default=1
        Token id of ``<pad>`` (XLM-R uses ``1``). Excluded from the mean.

    Notes
    -----
    Tensor flow (B = batch size, L = sequence length):

        input_ids  : LongTensor[B, L]
        ────────►  embedding lookup        →  Tensor[B, L, embed_dim]
        ────────►  attention-mask mean     →  Tensor[B, embed_dim]
        ────────►  Linear → ReLU → Dropout →  Tensor[B, hidden_dim]
        ────────►  Linear                  →  Tensor[B, n_classes]   (logits)
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        pad_token_id: int = 1,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        # padding_idx zeroes the embedding row for <pad> and excludes it from
        # gradient updates — clean way to preserve mask semantics.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ids : LongTensor[B, L]
        attention_mask : LongTensor[B, L]

        Returns
        -------
        FloatTensor[B, n_classes]
            Unnormalized class logits.
        """
        # (B, L)            →  (B, L, embed_dim)
        embeds = self.embedding(input_ids)

        # Mean-pool over real tokens only (mask out padding).
        mask = attention_mask.unsqueeze(-1).float()                # (B, L, 1)
        summed = (embeds * mask).sum(dim=1)                        # (B, embed_dim)
        denom = mask.sum(dim=1).clamp(min=1.0)                     # (B, 1)
        pooled = summed / denom                                    # (B, embed_dim)

        # (B, embed_dim)    →  (B, n_classes)
        return self.mlp(pooled)


# =========================================================================== #
# 2. Text branch — XLM-R (Phase-3 Step 2)
# =========================================================================== #
class TextBranch(nn.Module):
    """XLM-R encoder producing a single ``(B, hidden_dim)`` sentence embedding.

    Parameters
    ----------
    model_name : str
        HuggingFace identifier (``"xlm-roberta-base"``, ``"xlm-roberta-large"``).
    pooling : {"cls", "mean"}, default="cls"
        Pooling strategy applied over token embeddings.
    freeze : bool, default=False
        If ``True`` all encoder parameters are frozen (tabular branch + head
        still train — useful for fast prototyping or low-data regimes).

    Attributes
    ----------
    encoder : transformers.PreTrainedModel
        The underlying XLM-R model.
    hidden_dim : int
        Dimensionality of the produced sentence embedding.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        pooling: str = "cls",
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}.")
        self.model_name = model_name
        self.pooling = pooling

        # Load encoder + cache its config so we can expose ``hidden_dim``.
        # ``output_hidden_states=False`` keeps memory low — we only need the
        # last-layer states for pooling.
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_dim: int = config.hidden_size

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Eval mode disables dropout in the frozen encoder; we still
            # toggle .train() on the wrapper class for the head/tabular layers.
            self.encoder.eval()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode a tokenized text batch.

        Parameters
        ----------
        input_ids : LongTensor[B, L]
            Subword token ids.
        attention_mask : LongTensor[B, L]
            1 for real tokens, 0 for padding.

        Returns
        -------
        FloatTensor[B, hidden_dim]
            Pooled sentence embedding ``h_text``.
        """
        # XLM-R returns a ModelOutput; ``last_hidden_state`` shape: (B, L, H).
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state                    # (B, L, H)

        if self.pooling == "cls":
            # Convention: token at index 0 is the BOS / <s> token for XLM-R,
            # which RoBERTa-style models use as the sequence representation.
            return last_hidden[:, 0, :]                            # (B, H)

        # "mean" pooling — mask-aware average over real tokens.
        return self._mean_pool(last_hidden, attention_mask)        # (B, H)

    @staticmethod
    def _mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        """Mask-aware mean pooling over the token dimension.

        Parameters
        ----------
        last_hidden_state : FloatTensor[B, L, H]
        attention_mask : LongTensor[B, L]

        Returns
        -------
        FloatTensor[B, H]
        """
        mask = attention_mask.unsqueeze(-1).float()                # (B, L, 1)
        summed = (last_hidden_state * mask).sum(dim=1)             # (B, H)
        # ``clamp(min=1)`` guards against degenerate all-zero masks (very
        # rare but possible if a row got fully truncated to padding).
        denom = mask.sum(dim=1).clamp(min=1.0)                     # (B, 1)
        return summed / denom                                      # (B, H)


# =========================================================================== #
# 3. Tabular branch — FT-Transformer (Phase-3 Step 3)
# =========================================================================== #
class TabularBranch(nn.Module):
    """FT-Transformer encoder for numerical + categorical behavior features.

    Internally relies on `rtdl_revisiting_models.FTTransformer
    <https://github.com/yandex-research/rtdl-revisiting-models>`_, which
    tokenizes each feature into a ``d_token``-dim embedding before applying
    a stack of Transformer blocks. We strip its built-in classification head
    and expose only the CLS-token (``d_out=None``) so the fusion layer
    receives a clean per-sample embedding.

    Parameters
    ----------
    n_num_features : int
        Number of continuous features.
    cat_cardinalities : sequence of int
        Cardinality of each categorical feature column.
    d_token : int, default=192
        Token embedding dimension.
    n_blocks : int, default=3
        Number of Transformer blocks.
    attention_n_heads : int, default=8
        Number of attention heads.
    ffn_d_hidden : int, default=256
        FFN hidden size.
    dropout : float, default=0.1
        Dropout used uniformly across attention/FFN/residual.

    Attributes
    ----------
    encoder : nn.Module
        The underlying FT-Transformer (without classification head).
    output_dim : int
        Dimensionality of the produced tabular embedding (= ``d_token``).
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: Sequence[int],
        d_token: int = 192,
        n_blocks: int = 3,
        attention_n_heads: int = 8,
        ffn_d_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Defer the rtdl import so the module file stays importable in
        # environments that haven't installed the optional dependency yet
        # (e.g. CI for the baselines-only branch).
        try:
            from rtdl_revisiting_models import FTTransformer
        except ImportError as e:
            raise ImportError(
                "TabularBranch requires `rtdl-revisiting-models`. "
                "Install with `pip install rtdl-revisiting-models`."
            ) from e

        self.n_num_features = n_num_features
        self.cat_cardinalities: List[int] = list(cat_cardinalities)
        self.output_dim = d_token

        # rtdl-revisiting-models uses ``n_cont_features`` (not n_num_features)
        # and accepts backbone hparams as flat kwargs. ``d_out=None`` skips the
        # internal head so we get the CLS-token (B, d_token) directly — fusion
        # happens later. We pass ``ffn_d_hidden_multiplier=None`` because we
        # specify ``ffn_d_hidden`` absolutely (the two are mutually exclusive).
        self.encoder = FTTransformer(
            n_cont_features=n_num_features,
            cat_cardinalities=self.cat_cardinalities,
            d_out=None,
            n_blocks=n_blocks,
            d_block=d_token,
            attention_n_heads=attention_n_heads,
            attention_dropout=dropout,
            ffn_d_hidden=ffn_d_hidden,
            ffn_d_hidden_multiplier=None,
            ffn_dropout=dropout,
            residual_dropout=dropout,
        )

    def forward(
        self,
        num_features: Optional[Tensor],
        cat_features: Optional[Tensor],
    ) -> Tensor:
        """Encode the tabular part of one batch.

        Parameters
        ----------
        num_features : FloatTensor[B, n_num] or ``None``
            Standardized numerical features. Pass ``None`` only if
            ``n_num_features == 0``.
        cat_features : LongTensor[B, n_cat] or ``None``
            Integer-encoded categorical features. Pass ``None`` only if
            ``len(cat_cardinalities) == 0``.

        Returns
        -------
        FloatTensor[B, output_dim]
            Tabular embedding ``h_tab`` (CLS-token of FT-Transformer).
        """
        # rtdl's FTTransformer expects ``None`` (not zero-width tensors)
        # when a modality is absent — pass through accordingly.
        # NB: positional args are (x_cont, x_cat).
        x_cont = num_features if self.n_num_features > 0 else None
        x_cat = cat_features if self.cat_cardinalities else None

        # Output shape: (B, d_token) — already the CLS-token representation.
        return self.encoder(x_cont, x_cat)


# =========================================================================== #
# 4. Late Fusion model (Phase-3 Step 4)
# =========================================================================== #
class LateFusionModel(nn.Module):
    """Concatenate text + tabular embeddings and classify via an MLP head.

    Tensor flow (B = batch, L = sequence length, H_t = text dim, H_b = tab dim):

        input_ids, attention_mask     ─►  TextBranch     →  (B, H_t)
        num_features, cat_features    ─►  TabularBranch  →  (B, H_b)
        ──────── concat along dim=1 ────────►              (B, H_t + H_b)
        ──────── Linear → ReLU → Dropout ────►            (B, fusion_hidden_dim)
        ──────── Linear                ────►              (B, n_classes)  [logits]

    Notes
    -----
    The forward signature is compatible with the dictionary returned by
    :class:`~src.dataset.SocialSentimentDataset`, so a training loop can do::

        logits = model(**batch)

    Softmax is applied **only** in :meth:`predict_proba` — the training
    objective (``CrossEntropyLoss``) operates on raw logits.
    """

    def __init__(self, config: LateFusionConfig) -> None:
        super().__init__()
        self.config = config

        # ----- Text branch -----
        self.text_branch = TextBranch(
            model_name=config.text_model_name,
            pooling=config.text_pooling,
            freeze=config.freeze_text_encoder,
        )
        # Auto-detect text hidden dim if user left it as the sentinel -1.
        text_dim = (
            config.text_hidden_dim if config.text_hidden_dim > 0
            else self.text_branch.hidden_dim
        )

        # ----- Tabular branch (skip if no tabular features configured) -----
        self.has_tabular = (
            config.n_num_features > 0 or len(config.cat_cardinalities) > 0
        )
        if self.has_tabular:
            self.tabular_branch = TabularBranch(
                n_num_features=config.n_num_features,
                cat_cardinalities=config.cat_cardinalities,
                d_token=config.ft_d_token,
                n_blocks=config.ft_n_blocks,
                attention_n_heads=config.ft_attention_n_heads,
                ffn_d_hidden=config.ft_ffn_d_hidden,
                dropout=config.ft_dropout,
            )
            tab_dim = self.tabular_branch.output_dim
        else:
            self.tabular_branch = None
            tab_dim = 0

        # ----- Fusion head -----
        # h_fusion = [h_text ⊕ h_tab]  ∈  R^(text_dim + tab_dim)
        fusion_in_dim = text_dim + tab_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, config.n_classes),
        )

        # Cache dims for introspection / unit tests.
        self.text_dim = text_dim
        self.tab_dim = tab_dim

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        num_features: Optional[Tensor] = None,
        cat_features: Optional[Tensor] = None,
        label: Optional[Tensor] = None,  # noqa: ARG002 — accepted for **batch unpack
    ) -> Tensor:
        """Run the full text + tabular → logits pipeline.

        Parameters
        ----------
        input_ids : LongTensor[B, L]
        attention_mask : LongTensor[B, L]
        num_features : FloatTensor[B, n_num] or ``None``
        cat_features : LongTensor[B, n_cat] or ``None``
        label
            Ignored — accepted only so callers can do ``model(**batch)``.

        Returns
        -------
        FloatTensor[B, n_classes]
            Unnormalized class logits.
        """
        # 1) Text encoder:        (B, L) → (B, H_t)
        h_text = self.text_branch(input_ids, attention_mask)

        # 2) Tabular encoder:     (B, n_num/n_cat) → (B, H_b)   [optional]
        if self.has_tabular and self.tabular_branch is not None:
            h_tab = self.tabular_branch(num_features, cat_features)
            # 3) Concatenate ⊕:    (B, H_t) ⊕ (B, H_b) → (B, H_t + H_b)
            h_fusion = torch.cat([h_text, h_tab], dim=-1)
        else:
            # Text-only fallback (still useful for ablation studies).
            h_fusion = h_text

        # 4) Classification head:  (B, H_t + H_b) → (B, n_classes)
        logits = self.fusion_head(h_fusion)
        return logits

    # ------------------------------------------------------------------ #
    # Inference helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_proba(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        num_features: Optional[Tensor] = None,
        cat_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Softmax probabilities over classes, shape ``(B, n_classes)``."""
        self.eval()
        logits = self.forward(
            input_ids, attention_mask, num_features, cat_features,
        )
        # Softmax along the class dim — produces a proper probability simplex.
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        num_features: Optional[Tensor] = None,
        cat_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Argmax class predictions, shape ``(B,)`` of ``LongTensor``."""
        probs = self.predict_proba(
            input_ids, attention_mask, num_features, cat_features,
        )
        return probs.argmax(dim=-1)

    # ------------------------------------------------------------------ #
    # Persistence (HuggingFace-style layout)
    # ------------------------------------------------------------------ #
    def save_pretrained(self, save_dir: PathLike) -> None:
        """Save weights + config to ``save_dir``.

        Layout::

            save_dir/
            ├── pytorch_model.bin   # state_dict
            └── config.json         # LateFusionConfig as JSON
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Serialize the dataclass config — converting any tuple to list for
        # safe JSON round-tripping.
        cfg_dict = asdict(self.config) if is_dataclass(self.config) else dict(self.config)
        cfg_dict["cat_cardinalities"] = list(cfg_dict.get("cat_cardinalities", []))
        with (save_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(cfg_dict, fh, ensure_ascii=False, indent=2)

        # Save just the state_dict — keeps file size minimal vs torch.save(model).
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, save_dir: PathLike) -> "LateFusionModel":
        """Reconstruct a :class:`LateFusionModel` from a directory."""
        save_dir = Path(save_dir)
        with (save_dir / "config.json").open("r", encoding="utf-8") as fh:
            cfg_dict = json.load(fh)
        config = LateFusionConfig(**cfg_dict)

        model = cls(config)
        state = torch.load(
            save_dir / "pytorch_model.bin", map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state)
        return model
