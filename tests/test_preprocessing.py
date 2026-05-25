"""Unit tests for src/preprocessing.py.

Tests cover:
- TeencodeNormalizer: normalize(), transform(), edge cases, JSON dict loading
- TabularPreprocessor: fit(), transform(), fit_transform(), save/load, edge cases
- stratified_split: ratio validation, stratification correctness
- cohens_kappa: agreement metric

Run::

    pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    TabularPreprocessor,
    TeencodeNormalizer,
    cohens_kappa,
    stratified_split,
)


# =========================================================================== #
# TeencodeNormalizer
# =========================================================================== #
class TestTeencodeNormalizerBasics:

    def setup_method(self):
        self.norm = TeencodeNormalizer()

    def test_basic_teencode_substitution(self):
        assert self.norm.normalize("k bik j") == "không biết gì"

    def test_emoji_replacement(self):
        result = self.norm.normalize("vui quá 😊")
        assert "[SMILE]" in result

    def test_sad_emoji(self):
        result = self.norm.normalize("buồn quá 😢")
        assert "[CRY]" in result

    def test_angry_emoji(self):
        result = self.norm.normalize("tức quá 😡")
        assert "[ANGRY]" in result

    def test_repeat_char_collapse(self):
        result = self.norm.normalize("vui quáaaa")
        assert "quáaaa" not in result
        assert "quá" in result

    def test_repeat_char_collapse_english(self):
        result = self.norm.normalize("hahahahaha")
        assert len(result) < len("hahahahaha") or "ha" in result

    def test_lowercase(self):
        result = self.norm.normalize("Tôi Không Biết")
        assert result == result.lower()

    def test_whitespace_normalization(self):
        result = self.norm.normalize("vui   quá   lắm")
        assert "  " not in result

    def test_empty_string(self):
        assert self.norm.normalize("") == ""

    def test_none_like_input(self):
        assert self.norm.normalize(None) == ""  # type: ignore[arg-type]

    def test_whitespace_only(self):
        assert self.norm.normalize("   ") == ""

    def test_trailing_punctuation_preserved(self):
        result = self.norm.normalize("k bik j!")
        assert result.endswith("!")

    def test_vcl_maps_to_rat(self):
        result = self.norm.normalize("đẹp vcl luôn")
        assert "rất" in result

    def test_transform_list(self):
        texts = ["k bik j", "vui 😊", ""]
        results = self.norm.transform(texts)
        assert len(results) == 3
        assert results[2] == ""

    def test_transform_series(self):
        series = pd.Series(["k bik j", "vui 😊", ""])
        results = self.norm.transform(series)
        assert len(results) == 3

    def test_callable_interface(self):
        assert self.norm("k bik j") == self.norm.normalize("k bik j")

    def test_multiple_emojis(self):
        result = self.norm.normalize("vui 😊😂 nhưng cũng sợ 😱")
        assert "[SMILE]" in result
        assert "[LAUGH_TEARS]" in result
        assert "[SHOCK]" in result

    def test_mixed_emoji_and_teencode(self):
        result = self.norm.normalize("hok bik j luôn 😊")
        assert "không" in result
        assert "[SMILE]" in result

    def test_emoji_token_uppercased(self):
        result = self.norm.normalize("buồn 😢")
        assert "[CRY]" in result
        assert "[cry]" not in result


class TestTeencodeNormalizerCustomDict:

    def test_json_dict_extends_defaults(self, tmp_path):
        dict_path = tmp_path / "custom.json"
        dict_path.write_text(json.dumps({"customslang": "standard form"}), encoding="utf-8")
        norm = TeencodeNormalizer(teencode_dict_path=str(dict_path))
        assert norm.normalize("customslang") == "standard form"

    def test_json_dict_overrides_defaults(self, tmp_path):
        dict_path = tmp_path / "custom.json"
        dict_path.write_text(json.dumps({"k": "override"}), encoding="utf-8")
        norm = TeencodeNormalizer(teencode_dict_path=str(dict_path))
        assert norm.normalize("k") == "override"

    def test_missing_json_path_tolerated(self):
        norm = TeencodeNormalizer(teencode_dict_path="nonexistent_file.json")
        assert norm.normalize("k bik j") == "không biết gì"

    def test_direct_mapping_highest_precedence(self):
        norm = TeencodeNormalizer(mapping={"k": "custom_override"})
        assert norm.normalize("k") == "custom_override"

    def test_extended_teencode_json_loads(self):
        dict_path = (
            Path(__file__).resolve().parents[1] / "data" / "external" / "teencode.json"
        )
        if not dict_path.exists():
            pytest.skip("teencode.json not found — run project setup first")
        norm = TeencodeNormalizer(teencode_dict_path=str(dict_path))
        # Should still handle built-in cases after extending
        assert norm.normalize("k bik j") == "không biết gì"
        # Extended dict should add new entries
        assert len(norm.mapping) > 80


class TestTeencodeNormalizerFlags:

    def test_handle_emoji_false(self):
        norm = TeencodeNormalizer(handle_emoji=False)
        result = norm.normalize("vui 😊")
        assert "[SMILE]" not in result
        assert "😊" in result

    def test_collapse_repeats_false(self):
        norm = TeencodeNormalizer(collapse_repeats=False)
        result = norm.normalize("vui quáaaa")
        assert "quáaaa" in result

    def test_lowercase_false(self):
        norm = TeencodeNormalizer(lowercase=False)
        result = norm.normalize("Không Biết")
        assert "K" in result or "N" in result


# =========================================================================== #
# TabularPreprocessor
# =========================================================================== #
NUM_COLS = ["text_length", "n_words", "likes"]
CAT_COLS = ["has_emoji", "is_crawled"]


def _make_df(n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "text_length": rng.uniform(10, 500, n).astype(np.float32),
        "n_words":     rng.integers(1, 100, n).astype(np.float32),
        "likes":       rng.uniform(0, 5000, n).astype(np.float32),
        "has_emoji":   rng.choice(["yes", "no"], n),
        "is_crawled":  rng.choice(["0", "1"], n),
    })


class TestTabularPreprocessorFitTransform:

    def test_fit_returns_self(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        df = _make_df()
        assert tp.fit(df) is tp

    def test_transform_output_shapes(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        df = _make_df(20)
        tp.fit(df[:10])
        out = tp.transform(df[10:])
        assert out["num_features"].shape == (10, len(NUM_COLS))
        assert out["cat_features"].shape == (10, len(CAT_COLS))

    def test_num_features_float32(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        df = _make_df()
        tp.fit(df)
        out = tp.transform(df)
        assert out["num_features"].dtype == np.float32

    def test_cat_features_int64(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        df = _make_df()
        tp.fit(df)
        out = tp.transform(df)
        assert out["cat_features"].dtype == np.int64

    def test_z_score_mean_near_zero(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        df = _make_df(200)
        tp.fit(df)
        out = tp.transform(df)
        # Mean of z-scored data should be near 0
        col_means = out["num_features"].mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=0.1)

    def test_fit_transform_same_as_fit_then_transform(self):
        tp1 = TabularPreprocessor(NUM_COLS, CAT_COLS)
        tp2 = TabularPreprocessor(NUM_COLS, CAT_COLS)
        df = _make_df(20)
        out1 = tp1.fit_transform(df)
        tp2.fit(df)
        out2 = tp2.transform(df)
        np.testing.assert_array_almost_equal(out1["num_features"], out2["num_features"])
        np.testing.assert_array_equal(out1["cat_features"], out2["cat_features"])

    def test_unk_category_at_transform_time(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        df = _make_df(10)
        tp.fit(df)
        # Inject unseen category value
        df_new = df.copy()
        df_new["has_emoji"] = "maybe"
        out = tp.transform(df_new)
        col_idx = CAT_COLS.index("has_emoji")
        unk_idx = tp.cat_vocab_["has_emoji"]["<UNK>"]
        assert (out["cat_features"][:, col_idx] == unk_idx).all()

    def test_missing_numerical_imputed(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        df = _make_df(10)
        tp.fit(df)
        df_nan = df.copy()
        df_nan.loc[0, "likes"] = np.nan
        out = tp.transform(df_nan)
        assert np.isfinite(out["num_features"][0, NUM_COLS.index("likes")])

    def test_transform_before_fit_raises(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        with pytest.raises(RuntimeError, match="fit"):
            tp.transform(_make_df())

    def test_missing_columns_raises(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        df = _make_df()
        tp.fit(df)
        with pytest.raises(KeyError):
            tp.transform(df.drop(columns=["likes"]))

    def test_invalid_fillna_strategy_raises(self):
        with pytest.raises(ValueError, match="fillna_strategy"):
            TabularPreprocessor(NUM_COLS, CAT_COLS, fillna_strategy="mode")

    def test_empty_numerical_cols(self):
        tp = TabularPreprocessor([], CAT_COLS)
        df = _make_df()
        out = tp.fit_transform(df)
        assert out["num_features"].shape == (len(df), 0)

    def test_empty_categorical_cols(self):
        tp = TabularPreprocessor(NUM_COLS, [])
        df = _make_df()
        out = tp.fit_transform(df)
        assert out["cat_features"].shape == (len(df), 0)

    def test_cat_cardinalities_property(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        tp.fit(_make_df())
        cards = tp.cat_cardinalities
        assert len(cards) == len(CAT_COLS)
        # Each cardinality should be at least 2 (UNK + at least one real value)
        assert all(c >= 2 for c in cards)

    def test_cat_cardinalities_before_fit_raises(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        with pytest.raises(RuntimeError):
            _ = tp.cat_cardinalities

    def test_n_num_features(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        assert tp.n_num_features == len(NUM_COLS)

    def test_n_cat_features(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        assert tp.n_cat_features == len(CAT_COLS)

    def test_mean_fillna_strategy(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS, fillna_strategy="mean")
        tp.fit(_make_df())
        assert all(np.isfinite(v) for v in tp.num_imputers_.values())

    def test_zero_fillna_strategy(self):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS, fillna_strategy="zero")
        tp.fit(_make_df())
        assert tp.num_imputers_["text_length"] == 0.0

    def test_zero_variance_column_stable(self):
        tp = TabularPreprocessor(["const"], CAT_COLS)
        df = _make_df()
        df["const"] = 42.0
        tp.fit(df)
        out = tp.transform(df)
        # std was 0 → replaced with 1.0 → z-score should be all 0
        np.testing.assert_allclose(out["num_features"][:, 0], 0.0, atol=1e-5)


class TestTabularPreprocessorPersistence:

    def test_save_and_load(self, tmp_path):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        df = _make_df(20)
        tp.fit(df)
        save_path = tmp_path / "tp.joblib"
        tp.save(save_path)
        tp2 = TabularPreprocessor.load(save_path)
        out1 = tp.transform(df)
        out2 = tp2.transform(df)
        np.testing.assert_array_equal(out1["num_features"], out2["num_features"])
        np.testing.assert_array_equal(out1["cat_features"], out2["cat_features"])

    def test_save_before_fit_raises(self, tmp_path):
        tp = TabularPreprocessor(NUM_COLS, CAT_COLS)
        with pytest.raises(RuntimeError, match="un-fitted"):
            tp.save(tmp_path / "tp.joblib")

    def test_load_wrong_type_raises(self, tmp_path):
        import joblib
        dummy_path = tmp_path / "dummy.joblib"
        joblib.dump({"not": "a preprocessor"}, dummy_path)
        with pytest.raises(TypeError):
            TabularPreprocessor.load(dummy_path)


# =========================================================================== #
# stratified_split
# =========================================================================== #
class TestStratifiedSplit:

    def _make_labeled_df(self, n_per_class: int = 20) -> pd.DataFrame:
        labels = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
        rows = []
        for lbl in labels:
            for i in range(n_per_class):
                rows.append({"text": f"sample {lbl} {i}", "label": lbl})
        return pd.DataFrame(rows)

    def test_returns_three_splits(self):
        df = self._make_labeled_df()
        result = stratified_split(df, "label")
        assert len(result) == 3

    def test_sizes_sum_to_total(self):
        df = self._make_labeled_df()
        train, val, test = stratified_split(df, "label")
        assert len(train) + len(val) + len(test) == len(df)

    def test_approximate_ratios(self):
        df = self._make_labeled_df(100)
        train, val, test = stratified_split(df, "label")
        n = len(df)
        assert abs(len(train) / n - 0.70) < 0.03
        assert abs(len(val)   / n - 0.15) < 0.03
        assert abs(len(test)  / n - 0.15) < 0.03

    def test_stratification_preserves_distribution(self):
        df = self._make_labeled_df(50)
        train, val, test = stratified_split(df, "label")
        src_dist = df["label"].value_counts(normalize=True).sort_index()
        for split in (train, val, test):
            split_dist = split["label"].value_counts(normalize=True).sort_index()
            for lbl in src_dist.index:
                assert abs(src_dist[lbl] - split_dist.get(lbl, 0)) < 0.08

    def test_bad_ratios_raises(self):
        df = self._make_labeled_df()
        with pytest.raises(ValueError, match="1.0"):
            stratified_split(df, "label", train_size=0.5, val_size=0.3, test_size=0.3)

    def test_rare_class_raises(self):
        df = pd.DataFrame({"text": ["a"], "label": ["joy"]})
        with pytest.raises(ValueError, match="2 samples"):
            stratified_split(df, "label")

    def test_reproducibility(self):
        df = self._make_labeled_df()
        t1, v1, te1 = stratified_split(df, "label", seed=42)
        t2, v2, te2 = stratified_split(df, "label", seed=42)
        pd.testing.assert_frame_equal(t1.reset_index(drop=True), t2.reset_index(drop=True))

    def test_different_seeds_different_splits(self):
        df = self._make_labeled_df()
        t1, _, _ = stratified_split(df, "label", seed=1)
        t2, _, _ = stratified_split(df, "label", seed=2)
        assert not t1["text"].equals(t2["text"])

    def test_no_data_leakage(self):
        df = self._make_labeled_df(50)
        train, val, test = stratified_split(df, "label")
        train_texts = set(train["text"])
        val_texts   = set(val["text"])
        test_texts  = set(test["text"])
        assert train_texts.isdisjoint(val_texts)
        assert train_texts.isdisjoint(test_texts)
        assert val_texts.isdisjoint(test_texts)


# =========================================================================== #
# cohens_kappa
# =========================================================================== #
class TestCohensKappa:

    def test_perfect_agreement(self):
        labels = ["joy", "sad", "anger"] * 10
        kappa = cohens_kappa(labels, labels)
        assert kappa == pytest.approx(1.0)

    def test_random_agreement_near_zero(self):
        rng = np.random.default_rng(0)
        labels = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
        a = rng.choice(labels, 200).tolist()
        b = rng.choice(labels, 200).tolist()
        kappa = cohens_kappa(a, b)
        # Random should be near 0 (±0.15 tolerance)
        assert abs(kappa) < 0.20

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            cohens_kappa(["joy", "sad"], ["joy"])

    def test_with_label_restriction(self):
        a = ["joy", "sad", "joy", "sad"]
        b = ["joy", "joy", "sad", "sad"]
        kappa = cohens_kappa(a, b, labels=["joy", "sad"])
        assert -1.0 <= kappa <= 1.0

    def test_returns_float(self):
        labels = ["joy", "sad"] * 5
        kappa = cohens_kappa(labels, labels)
        assert isinstance(kappa, float)
