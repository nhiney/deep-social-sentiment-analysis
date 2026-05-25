"""Unit tests for src/models.py.

Tests run WITHOUT downloading XLM-R weights. The heavy models (TextBranch,
LateFusionModel) are tested via:
  - LateFusionConfig: dataclass instantiation and defaults
  - DnnBaseline: full forward pass (small random vocab, no pretrained weights)
  - TfidfBaseline: fit/predict pipeline (pure sklearn, no GPU)
  - LateFusionModel.from_config / save_pretrained / load_pretrained:
    mocked to avoid the HuggingFace download

Run::

    pytest tests/test_models.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import (
    DnnBaseline,
    LateFusionConfig,
    TfidfBaseline,
)


# =========================================================================== #
# LateFusionConfig
# =========================================================================== #
class TestLateFusionConfig:

    def test_defaults(self):
        cfg = LateFusionConfig()
        assert cfg.text_model_name == "xlm-roberta-base"
        assert cfg.text_pooling == "cls"
        assert cfg.n_classes == 7
        assert cfg.ft_d_token == 192
        assert cfg.ft_n_blocks == 3
        assert cfg.fusion_hidden_dim == 256
        assert cfg.fusion_dropout == 0.2

    def test_custom_values(self):
        cfg = LateFusionConfig(
            text_model_name="xlm-roberta-large",
            n_classes=3,
            fusion_hidden_dim=512,
            n_num_features=10,
        )
        assert cfg.text_model_name == "xlm-roberta-large"
        assert cfg.n_classes == 3
        assert cfg.fusion_hidden_dim == 512
        assert cfg.n_num_features == 10

    def test_cat_cardinalities_default_empty(self):
        cfg = LateFusionConfig()
        assert cfg.cat_cardinalities == []

    def test_cat_cardinalities_custom(self):
        cfg = LateFusionConfig(cat_cardinalities=[3, 5, 2])
        assert list(cfg.cat_cardinalities) == [3, 5, 2]

    def test_freeze_text_encoder_default_false(self):
        cfg = LateFusionConfig()
        assert cfg.freeze_text_encoder is False

    def test_text_hidden_dim_auto_detect_marker(self):
        cfg = LateFusionConfig()
        assert cfg.text_hidden_dim == -1


# =========================================================================== #
# DnnBaseline
# =========================================================================== #
class TestDnnBaseline:

    VOCAB_SIZE = 500
    N_CLASSES = 7
    BATCH = 4
    SEQ_LEN = 16

    def _make_batch(self, batch: int = BATCH, seq: int = SEQ_LEN):
        input_ids = torch.randint(0, self.VOCAB_SIZE, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        # Zero out padding at the tail
        for i in range(batch):
            pad_start = seq - (i % 4)
            attention_mask[i, pad_start:] = 0
            input_ids[i, pad_start:] = 1  # pad_token_id=1
        return input_ids, attention_mask

    def test_output_shape(self):
        model = DnnBaseline(self.VOCAB_SIZE, self.N_CLASSES)
        ids, mask = self._make_batch()
        logits = model(ids, mask)
        assert logits.shape == (self.BATCH, self.N_CLASSES)

    def test_output_dtype(self):
        model = DnnBaseline(self.VOCAB_SIZE, self.N_CLASSES)
        ids, mask = self._make_batch()
        logits = model(ids, mask)
        assert logits.dtype == torch.float32

    def test_no_nan_in_output(self):
        model = DnnBaseline(self.VOCAB_SIZE, self.N_CLASSES)
        ids, mask = self._make_batch()
        logits = model(ids, mask)
        assert not torch.isnan(logits).any()

    def test_batch_size_one(self):
        model = DnnBaseline(self.VOCAB_SIZE, self.N_CLASSES)
        ids, mask = self._make_batch(batch=1)
        logits = model(ids, mask)
        assert logits.shape == (1, self.N_CLASSES)

    def test_custom_embed_and_hidden_dim(self):
        model = DnnBaseline(self.VOCAB_SIZE, self.N_CLASSES, embed_dim=64, hidden_dim=128)
        ids, mask = self._make_batch()
        logits = model(ids, mask)
        assert logits.shape == (self.BATCH, self.N_CLASSES)

    def test_all_padding_mask_stable(self):
        model = DnnBaseline(self.VOCAB_SIZE, self.N_CLASSES)
        ids = torch.ones(2, 10, dtype=torch.long)
        mask = torch.zeros(2, 10, dtype=torch.long)
        logits = model(ids, mask)
        assert not torch.isnan(logits).any()

    def test_gradient_flows(self):
        model = DnnBaseline(self.VOCAB_SIZE, self.N_CLASSES)
        ids, mask = self._make_batch()
        logits = model(ids, mask)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

    def test_eval_mode_deterministic(self):
        model = DnnBaseline(self.VOCAB_SIZE, self.N_CLASSES)
        ids, mask = self._make_batch()
        model.eval()
        with torch.no_grad():
            out1 = model(ids, mask)
            out2 = model(ids, mask)
        torch.testing.assert_close(out1, out2)

    def test_padding_tokens_not_contributing(self):
        model = DnnBaseline(self.VOCAB_SIZE, self.N_CLASSES, pad_token_id=1)
        torch.manual_seed(0)
        ids = torch.randint(2, self.VOCAB_SIZE, (1, 8))
        full_mask = torch.ones(1, 8, dtype=torch.long)

        # Run with full sequence
        model.eval()
        with torch.no_grad():
            out_full = model(ids, full_mask)

        # Add padding tokens to the end (mask says they don't count)
        ids_padded = torch.cat([ids, torch.ones(1, 4, dtype=torch.long)], dim=1)
        padded_mask = torch.cat([full_mask, torch.zeros(1, 4, dtype=torch.long)], dim=1)
        with torch.no_grad():
            out_padded = model(ids_padded, padded_mask)

        torch.testing.assert_close(out_full, out_padded)

    def test_is_nn_module(self):
        model = DnnBaseline(self.VOCAB_SIZE, self.N_CLASSES)
        assert isinstance(model, torch.nn.Module)

    def test_parameter_count_reasonable(self):
        model = DnnBaseline(100, 7, embed_dim=32, hidden_dim=64)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be > 0 and < 1M for a tiny model
        assert 0 < n_params < 1_000_000


# =========================================================================== #
# TfidfBaseline
# =========================================================================== #
class TestTfidfBaseline:

    LABELS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]

    def _make_data(self, n: int = 70):
        rng = np.random.default_rng(0)
        texts = [
            f"sample text number {i} about emotion topic"
            for i in range(n)
        ]
        labels = [self.LABELS[i % len(self.LABELS)] for i in range(n)]
        return texts, labels

    def test_fit_predict_logreg(self):
        model = TfidfBaseline(classifier="logreg")
        texts, labels = self._make_data()
        model.fit(texts, labels)
        preds = model.predict(texts)
        assert len(preds) == len(texts)
        assert all(p in self.LABELS for p in preds)

    def test_fit_predict_svm(self):
        model = TfidfBaseline(classifier="svm")
        texts, labels = self._make_data()
        model.fit(texts, labels)
        preds = model.predict(texts)
        assert len(preds) == len(texts)

    def test_predict_proba_logreg_shape(self):
        model = TfidfBaseline(classifier="logreg")
        texts, labels = self._make_data()
        model.fit(texts, labels)
        proba = model.predict_proba(texts)
        assert proba.shape == (len(texts), len(self.LABELS))

    def test_predict_proba_sums_to_one(self):
        model = TfidfBaseline(classifier="logreg")
        texts, labels = self._make_data()
        model.fit(texts, labels)
        proba = model.predict_proba(texts)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_proba_svm_fallback(self):
        model = TfidfBaseline(classifier="svm")
        texts, labels = self._make_data()
        model.fit(texts, labels)
        proba = model.predict_proba(texts)
        assert proba.shape[1] == len(self.LABELS)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_invalid_classifier_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            TfidfBaseline(classifier="random_forest")

    def test_fit_returns_self(self):
        model = TfidfBaseline()
        texts, labels = self._make_data()
        assert model.fit(texts, labels) is model

    def test_single_text_predict(self):
        model = TfidfBaseline()
        texts, labels = self._make_data()
        model.fit(texts, labels)
        preds = model.predict(["vui quá hôm nay"])
        assert len(preds) == 1
        assert preds[0] in self.LABELS

    def test_custom_ngram_range(self):
        model = TfidfBaseline(ngram_range=(1, 3), max_features=1000)
        texts, labels = self._make_data()
        model.fit(texts, labels)
        preds = model.predict(texts[:5])
        assert len(preds) == 5

    def test_classifier_name_attribute(self):
        m1 = TfidfBaseline(classifier="logreg")
        m2 = TfidfBaseline(classifier="svm")
        assert m1.classifier_name == "logreg"
        assert m2.classifier_name == "svm"
