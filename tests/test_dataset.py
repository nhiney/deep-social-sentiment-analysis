"""Unit tests for src/dataset.py.

Tests run without a real tokenizer checkpoint or GPU — a stub tokenizer returns
zero tensors of the correct shape, and TabularPreprocessor is fitted on a tiny
in-memory DataFrame.

Run::

    pytest tests/test_dataset.py -v
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import SocialSentimentDataset, sentiment_collate_fn
from src.preprocessing import TabularPreprocessor, TeencodeNormalizer


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
LABEL_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
MAX_LENGTH = 16
N_ROWS = 14  # 2 complete cycles over 7 classes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokenizer(max_length: int = MAX_LENGTH) -> MagicMock:
    """Stub tokenizer: returns zero input_ids + ones attention_mask."""
    tok = MagicMock()

    def _call(*args, **kwargs):
        ml = kwargs.get("max_length", max_length)
        return {
            "input_ids":      torch.zeros(1, ml, dtype=torch.long),
            "attention_mask": torch.ones(1, ml, dtype=torch.long),
        }

    tok.side_effect = _call
    return tok


def _make_df(n: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "text":         [f"Hôm nay tôi cảm thấy {CLASS_NAMES[i % 7]}!" for i in range(n)],
        "label":        [CLASS_NAMES[i % 7] for i in range(n)],
        "text_length":  rng.integers(10, 200, size=n).astype(float),
        "n_words":      rng.integers(2, 20, size=n).astype(float),
    })


def _make_tab_pp_empty(df: pd.DataFrame) -> TabularPreprocessor:
    return TabularPreprocessor(numerical_cols=[], categorical_cols=[]).fit(df)


def _make_tab_pp_numeric(df: pd.DataFrame) -> TabularPreprocessor:
    return TabularPreprocessor(
        numerical_cols=["text_length", "n_words"],
        categorical_cols=[],
    ).fit(df)


def _make_dataset(
    df: pd.DataFrame,
    tab_pp: TabularPreprocessor,
    return_label: bool = True,
    normalizer: Optional[TeencodeNormalizer] = None,
    label_col: str = "label",
) -> SocialSentimentDataset:
    return SocialSentimentDataset(
        dataframe=df,
        text_column="text",
        label_column=label_col,
        tokenizer=_make_tokenizer(),
        tabular_preprocessor=tab_pp,
        teencode_normalizer=normalizer,
        label_to_id=LABEL_TO_ID,
        max_length=MAX_LENGTH,
        return_label=return_label,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def df():
    return _make_df()


@pytest.fixture(scope="module")
def empty_tab_pp(df):
    return _make_tab_pp_empty(df)


@pytest.fixture(scope="module")
def numeric_tab_pp(df):
    return _make_tab_pp_numeric(df)


@pytest.fixture(scope="module")
def dataset(df, empty_tab_pp):
    return _make_dataset(df, empty_tab_pp)


@pytest.fixture(scope="module")
def dataset_with_tabular(df, numeric_tab_pp):
    return _make_dataset(df, numeric_tab_pp)


# ---------------------------------------------------------------------------
# Tests: __len__
# ---------------------------------------------------------------------------
class TestLen:
    def test_len_matches_dataframe(self, dataset, df):
        assert len(dataset) == len(df)

    def test_len_after_subset(self, df, empty_tab_pp):
        sub = df.iloc[:5].copy()
        ds = _make_dataset(sub, _make_tab_pp_empty(sub))
        assert len(ds) == 5


# ---------------------------------------------------------------------------
# Tests: __getitem__ keys
# ---------------------------------------------------------------------------
class TestGetitemKeys:
    def test_keys_with_label(self, dataset):
        sample = dataset[0]
        expected = {"input_ids", "attention_mask", "num_features", "cat_features", "label"}
        assert set(sample.keys()) == expected

    def test_keys_inference_mode(self, df, empty_tab_pp):
        ds = _make_dataset(df, empty_tab_pp, return_label=False)
        sample = ds[0]
        assert "label" not in sample
        assert "input_ids" in sample
        assert "attention_mask" in sample


# ---------------------------------------------------------------------------
# Tests: __getitem__ shapes
# ---------------------------------------------------------------------------
class TestGetitemShapes:
    def test_input_ids_shape(self, dataset):
        sample = dataset[0]
        assert sample["input_ids"].shape == (MAX_LENGTH,)

    def test_attention_mask_shape(self, dataset):
        sample = dataset[0]
        assert sample["attention_mask"].shape == (MAX_LENGTH,)

    def test_num_features_empty_for_text_only(self, dataset):
        sample = dataset[0]
        assert sample["num_features"].shape == (0,)

    def test_cat_features_empty_for_text_only(self, dataset):
        sample = dataset[0]
        assert sample["cat_features"].shape == (0,)

    def test_label_is_scalar(self, dataset):
        sample = dataset[0]
        assert sample["label"].ndim == 0

    def test_num_features_shape_with_tabular(self, dataset_with_tabular):
        sample = dataset_with_tabular[0]
        assert sample["num_features"].shape == (2,)  # text_length, n_words

    def test_num_features_shape_all_samples(self, dataset_with_tabular):
        for i in range(len(dataset_with_tabular)):
            assert dataset_with_tabular[i]["num_features"].shape == (2,)


# ---------------------------------------------------------------------------
# Tests: __getitem__ dtypes
# ---------------------------------------------------------------------------
class TestGetitemDtypes:
    def test_input_ids_dtype(self, dataset):
        assert dataset[0]["input_ids"].dtype == torch.long

    def test_attention_mask_dtype(self, dataset):
        assert dataset[0]["attention_mask"].dtype == torch.long

    def test_num_features_dtype(self, dataset_with_tabular):
        assert dataset_with_tabular[0]["num_features"].dtype == torch.float

    def test_cat_features_dtype_empty(self, dataset):
        assert dataset[0]["cat_features"].dtype == torch.long

    def test_label_dtype(self, dataset):
        assert dataset[0]["label"].dtype == torch.long


# ---------------------------------------------------------------------------
# Tests: label encoding
# ---------------------------------------------------------------------------
class TestLabelEncoding:
    def test_string_labels_encoded_to_int(self, dataset, df):
        for i, row in df.iterrows():
            expected_id = LABEL_TO_ID[row["label"]]
            assert dataset[i]["label"].item() == expected_id

    def test_integer_labels_work_without_label_to_id(self, df, empty_tab_pp):
        df_int = df.copy()
        df_int["label_id"] = df_int["label"].map(LABEL_TO_ID)
        ds = SocialSentimentDataset(
            dataframe=df_int,
            text_column="text",
            label_column="label_id",
            tokenizer=_make_tokenizer(),
            tabular_preprocessor=empty_tab_pp,
            label_to_id=None,
            max_length=MAX_LENGTH,
            return_label=True,
        )
        for i in range(len(ds)):
            assert 0 <= ds[i]["label"].item() < len(CLASS_NAMES)

    def test_missing_label_to_id_raises_for_string_labels(self, df, empty_tab_pp):
        with pytest.raises(ValueError, match="label_to_id"):
            SocialSentimentDataset(
                dataframe=df,
                text_column="text",
                label_column="label",
                tokenizer=_make_tokenizer(),
                tabular_preprocessor=empty_tab_pp,
                label_to_id=None,
                max_length=MAX_LENGTH,
                return_label=True,
            )

    def test_all_label_ids_in_valid_range(self, dataset):
        for i in range(len(dataset)):
            label = dataset[i]["label"].item()
            assert 0 <= label < len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Tests: TeencodeNormalizer applied
# ---------------------------------------------------------------------------
class TestNormalizer:
    def test_normalizer_called_for_each_item(self, df, empty_tab_pp):
        normalizer = MagicMock(side_effect=lambda t: t)
        ds = _make_dataset(df, empty_tab_pp, normalizer=normalizer)
        _ = ds[0]
        _ = ds[1]
        assert normalizer.call_count == 2

    def test_no_normalizer_does_not_crash(self, dataset):
        _ = dataset[0]  # dataset fixture has no normalizer


# ---------------------------------------------------------------------------
# Tests: sentiment_collate_fn
# ---------------------------------------------------------------------------
class TestCollate:
    def _get_batch(self, dataset, n: int = 4):
        return [dataset[i] for i in range(n)]

    def test_collate_adds_batch_dim(self, dataset):
        batch = self._get_batch(dataset, n=4)
        out = sentiment_collate_fn(batch)
        assert out["input_ids"].shape == (4, MAX_LENGTH)
        assert out["attention_mask"].shape == (4, MAX_LENGTH)

    def test_collate_all_keys_present(self, dataset):
        batch = self._get_batch(dataset, n=3)
        out = sentiment_collate_fn(batch)
        assert set(out.keys()) == {"input_ids", "attention_mask", "num_features", "cat_features", "label"}

    def test_collate_label_shape(self, dataset):
        batch = self._get_batch(dataset, n=5)
        out = sentiment_collate_fn(batch)
        assert out["label"].shape == (5,)

    def test_collate_without_label_key(self, df, empty_tab_pp):
        ds = _make_dataset(df, empty_tab_pp, return_label=False)
        batch = [ds[i] for i in range(3)]
        out = sentiment_collate_fn(batch)
        assert "label" not in out

    def test_collate_num_features_shape(self, dataset_with_tabular):
        batch = [dataset_with_tabular[i] for i in range(4)]
        out = sentiment_collate_fn(batch)
        assert out["num_features"].shape == (4, 2)

    def test_collate_input_ids_dtype(self, dataset):
        batch = self._get_batch(dataset, n=2)
        out = sentiment_collate_fn(batch)
        assert out["input_ids"].dtype == torch.long

    def test_collate_single_sample(self, dataset):
        batch = [dataset[0]]
        out = sentiment_collate_fn(batch)
        assert out["input_ids"].shape == (1, MAX_LENGTH)

    def test_collate_label_values_match_items(self, dataset):
        indices = [0, 2, 4, 6]
        batch = [dataset[i] for i in indices]
        out = sentiment_collate_fn(batch)
        for j, i in enumerate(indices):
            assert out["label"][j].item() == dataset[i]["label"].item()
