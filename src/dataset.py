"""PyTorch Dataset wrapping text + tabular features for the Late Fusion model.

A single sample is a dictionary that :class:`~src.models.LateFusionModel`
can consume directly:

.. code-block:: python

    {
        "input_ids":      LongTensor[seq_len],
        "attention_mask": LongTensor[seq_len],
        "num_features":   FloatTensor[n_num],     # may be shape (0,)
        "cat_features":   LongTensor[n_cat],      # may be shape (0,)
        "label":          LongTensor[],           # scalar (omitted at inference)
    }

The collate function :func:`sentiment_collate_fn` stacks samples into a
batched dict the model can consume via ``model(**batch)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.preprocessing import TabularPreprocessor, TeencodeNormalizer

PathLike = Union[str, Path]


class SocialSentimentDataset(Dataset):
    """Dataset coupling normalized text with tabular behavior features.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Source dataframe holding text, tabular features and (optionally) labels.
    text_column : str
        Column name containing the raw text.
    label_column : str
        Column name holding labels — string class names (will be encoded via
        ``label_to_id``) **or** already-encoded integer ids.
    tokenizer : transformers.PreTrainedTokenizerBase
        XLM-R compatible tokenizer (``XLMRobertaTokenizerFast`` recommended).
    tabular_preprocessor : TabularPreprocessor
        **Already fitted** tabular preprocessor. Pass one with empty column
        lists for text-only experiments — its ``transform`` will return
        zero-width arrays which the LateFusion model handles.
    teencode_normalizer : TeencodeNormalizer, optional
        If provided, normalizes text *before* tokenization. Pass ``None``
        when the text is already normalized (e.g. loaded from a parquet
        produced by ``scripts/prepare_data.py``).
    label_to_id : mapping, optional
        Mapping from class name to integer id. If the label column contains
        strings, this is **required**.
    max_length : int, default=128
        Max number of subword tokens after truncation.
    return_label : bool, default=True
        If ``False`` (e.g. inference), the ``"label"`` key is omitted.

    Attributes
    ----------
    df : pandas.DataFrame
        Reset-index copy of the input dataframe.
    encoded_tabular : dict[str, np.ndarray]
        Cached output of ``tabular_preprocessor.transform(df)``.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_column: str,
        label_column: str,
        tokenizer: PreTrainedTokenizerBase,
        tabular_preprocessor: TabularPreprocessor,
        teencode_normalizer: Optional[TeencodeNormalizer] = None,
        label_to_id: Optional[Mapping[Any, int]] = None,
        max_length: int = 128,
        return_label: bool = True,
    ) -> None:
        # Defensive copy so downstream mutations of the source df don't
        # corrupt the cached arrays computed here.
        self.df = dataframe.reset_index(drop=True).copy()
        self.text_column = text_column
        self.label_column = label_column
        self.tokenizer = tokenizer
        self.normalizer = teencode_normalizer
        self.max_length = int(max_length)
        self.return_label = return_label
        self.label_to_id = dict(label_to_id) if label_to_id is not None else None

        # ---- Pre-encode the tabular block ONCE at construction time ----
        # Avoids per-sample re-encoding inside the DataLoader worker hot loop.
        self.encoded_tabular: Dict[str, np.ndarray] = (
            tabular_preprocessor.transform(self.df)
        )
        # Cache shapes so __getitem__ can return correctly-shaped tensors
        # even when n_num/n_cat is 0 (text-only ablation experiment).
        self._n_num: int = self.encoded_tabular["num_features"].shape[1]
        self._n_cat: int = self.encoded_tabular["cat_features"].shape[1]

        # ---- Pre-encode labels to integer ids if labels are strings ----
        if self.return_label:
            raw_labels = self.df[self.label_column]
            if pd.api.types.is_integer_dtype(raw_labels):
                self._label_ids = raw_labels.to_numpy(dtype=np.int64)
            else:
                if self.label_to_id is None:
                    raise ValueError(
                        "label_to_id mapping is required when label column "
                        "is not already integer-encoded."
                    )
                self._label_ids = (
                    raw_labels.map(self.label_to_id).to_numpy(dtype=np.int64)
                )

    # ------------------------------------------------------------------ #
    # torch.utils.data.Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one (text + tabular [+ label]) sample as a tensor dict."""
        # 1) Pull and (optionally) normalize the raw text.
        text = self.df.iloc[index][self.text_column]
        if self.normalizer is not None:
            text = self.normalizer(text)

        # 2) Tokenize. ``return_tensors="pt"`` adds a batch dim of 1 that we
        #    immediately squeeze — the DataLoader collate adds the real one.
        encoded = self._encode_text(text)

        # 3) Slice the pre-computed tabular arrays. Even when shape[1] is 0,
        #    these stay valid 1-D float/long tensors.
        sample: Dict[str, torch.Tensor] = {
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "num_features":   torch.from_numpy(
                self.encoded_tabular["num_features"][index]
            ).float(),
            "cat_features":   torch.from_numpy(
                self.encoded_tabular["cat_features"][index]
            ).long(),
        }

        # 4) Append the label if we're in train/val mode.
        if self.return_label:
            sample["label"] = torch.tensor(
                int(self._label_ids[index]), dtype=torch.long
            )

        return sample

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text into ``input_ids`` and ``attention_mask``.

        Both tensors are returned with shape ``(max_length,)`` — the DataLoader
        collate function adds the leading batch dimension.
        """
        # ``padding="max_length"`` so all samples in a batch have identical
        # lengths without needing dynamic padding logic in the collate fn.
        out = self.tokenizer(
            text if isinstance(text, str) else "",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "input_ids":      out["input_ids"].squeeze(0),       # (L,)
            "attention_mask": out["attention_mask"].squeeze(0),  # (L,)
        }


# --------------------------------------------------------------------------- #
# Collate
# --------------------------------------------------------------------------- #
def sentiment_collate_fn(
    batch: Sequence[Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Stack a list of samples into a batched tensor dictionary.

    Handles the ``"label"`` key gracefully (some datasets, e.g. inference,
    don't include it).

    Parameters
    ----------
    batch : sequence of mappings
        List of dicts as produced by :meth:`SocialSentimentDataset.__getitem__`.

    Returns
    -------
    dict[str, torch.Tensor]
        Batched tensors with a leading batch dimension.
    """
    # All samples share the same key set; use the first one to drive the loop.
    keys = list(batch[0].keys())
    return {k: torch.stack([s[k] for s in batch], dim=0) for k in keys}
