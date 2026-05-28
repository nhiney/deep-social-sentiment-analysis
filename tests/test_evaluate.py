"""Unit tests for src/evaluate.py.

All tests run on CPU with synthetic label arrays — no model checkpoint or GPU
required.

Run::

    pytest tests/test_evaluate.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluate import (
    _to_numpy,
    classification_report_str,
    compute_classification_metrics,
    confusion_matrix_array,
    per_class_metrics,
)


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------
CLASS_NAMES = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
N_CLASSES = len(CLASS_NAMES)

# Perfect predictions — every label matches.
_Y_PERFECT = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6])

# Deliberately imperfect: class 0 always predicted as 1.
_Y_TRUE_NOISY = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6])
_Y_PRED_NOISY = np.array([1, 1, 1, 1, 0, 2, 2, 3, 4, 5, 6])


# ---------------------------------------------------------------------------
# Tests: _to_numpy
# ---------------------------------------------------------------------------
class TestToNumpy:
    def test_list_int(self):
        result = _to_numpy([0, 1, 2])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_numpy_array_passthrough(self):
        arr = np.array([3, 4, 5])
        result = _to_numpy(arr)
        assert result is arr  # exact same object — no copy

    def test_torch_tensor_cpu(self):
        t = torch.tensor([1, 2, 3], dtype=torch.long)
        result = _to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_torch_tensor_requires_grad(self):
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = _to_numpy(t)
        assert isinstance(result, np.ndarray)

    def test_empty_list(self):
        result = _to_numpy([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_single_element_list(self):
        result = _to_numpy([42])
        assert result[0] == 42

    def test_2d_tensor(self):
        t = torch.zeros(3, 4, dtype=torch.long)
        result = _to_numpy(t)
        assert result.shape == (3, 4)


# ---------------------------------------------------------------------------
# Tests: compute_classification_metrics
# ---------------------------------------------------------------------------
class TestComputeClassificationMetrics:
    def test_perfect_predictions_accuracy(self):
        m = compute_classification_metrics(_Y_PERFECT, _Y_PERFECT)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_perfect_predictions_f1_macro(self):
        m = compute_classification_metrics(_Y_PERFECT, _Y_PERFECT)
        assert m["f1_macro"] == pytest.approx(1.0)

    def test_perfect_predictions_precision_macro(self):
        m = compute_classification_metrics(_Y_PERFECT, _Y_PERFECT)
        assert m["precision_macro"] == pytest.approx(1.0)

    def test_perfect_predictions_recall_macro(self):
        m = compute_classification_metrics(_Y_PERFECT, _Y_PERFECT)
        assert m["recall_macro"] == pytest.approx(1.0)

    def test_output_keys_macro(self):
        m = compute_classification_metrics(_Y_PERFECT, _Y_PERFECT)
        assert set(m.keys()) == {"accuracy", "f1_macro", "precision_macro", "recall_macro"}

    def test_output_keys_weighted(self):
        m = compute_classification_metrics(_Y_PERFECT, _Y_PERFECT, average="weighted")
        assert set(m.keys()) == {"accuracy", "f1_weighted", "precision_weighted", "recall_weighted"}

    def test_all_values_in_unit_interval(self):
        m = compute_classification_metrics(_Y_TRUE_NOISY, _Y_PRED_NOISY)
        for v in m.values():
            assert 0.0 <= v <= 1.0

    def test_noisy_accuracy_below_one(self):
        m = compute_classification_metrics(_Y_TRUE_NOISY, _Y_PRED_NOISY)
        assert m["accuracy"] < 1.0

    def test_accepts_torch_tensors(self):
        y_t = torch.from_numpy(_Y_PERFECT)
        m = compute_classification_metrics(y_t, y_t)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_accepts_python_lists(self):
        m = compute_classification_metrics(list(_Y_PERFECT), list(_Y_PERFECT))
        assert m["accuracy"] == pytest.approx(1.0)

    def test_labels_subset_param(self):
        # Restrict to class ids 0 and 1 only.
        m = compute_classification_metrics(_Y_TRUE_NOISY, _Y_PRED_NOISY, labels=[0, 1])
        assert 0.0 <= m["f1_macro"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: per_class_metrics
# ---------------------------------------------------------------------------
class TestPerClassMetrics:
    @pytest.fixture(scope="class")
    def result_perfect(self):
        return per_class_metrics(_Y_PERFECT, _Y_PERFECT, CLASS_NAMES)

    @pytest.fixture(scope="class")
    def result_noisy(self):
        return per_class_metrics(_Y_TRUE_NOISY, _Y_PRED_NOISY, CLASS_NAMES)

    def test_keys_are_class_names(self, result_perfect):
        assert set(result_perfect.keys()) == set(CLASS_NAMES)

    def test_each_class_has_required_subkeys(self, result_perfect):
        for cls in CLASS_NAMES:
            assert set(result_perfect[cls].keys()) == {"precision", "recall", "f1", "support"}

    def test_perfect_f1_per_class(self, result_perfect):
        for cls in CLASS_NAMES:
            assert result_perfect[cls]["f1"] == pytest.approx(1.0)

    def test_support_is_int(self, result_perfect):
        for cls in CLASS_NAMES:
            assert isinstance(result_perfect[cls]["support"], int)

    def test_support_sums_to_n(self, result_perfect):
        total_support = sum(v["support"] for v in result_perfect.values())
        assert total_support == len(_Y_PERFECT)

    def test_noisy_class_f1_below_one(self, result_noisy):
        # Class "joy" (id=0) was always predicted wrong in _Y_PRED_NOISY.
        assert result_noisy["joy"]["f1"] < 1.0

    def test_correct_class_has_f1_one(self, result_noisy):
        # Classes 3–6 were predicted perfectly in _Y_PRED_NOISY.
        for cls in ["fear", "disgust", "surprise", "neutral"]:
            assert result_noisy[cls]["f1"] == pytest.approx(1.0)

    def test_accepts_torch_tensors(self):
        y_t = torch.from_numpy(_Y_PERFECT)
        result = per_class_metrics(y_t, y_t, CLASS_NAMES)
        assert "joy" in result


# ---------------------------------------------------------------------------
# Tests: confusion_matrix_array
# ---------------------------------------------------------------------------
class TestConfusionMatrixArray:
    def test_shape(self):
        cm = confusion_matrix_array(_Y_PERFECT, _Y_PERFECT, n_classes=N_CLASSES)
        assert cm.shape == (N_CLASSES, N_CLASSES)

    def test_perfect_diagonal(self):
        cm = confusion_matrix_array(_Y_PERFECT, _Y_PERFECT, n_classes=N_CLASSES)
        off_diag = cm - np.diag(np.diag(cm))
        assert off_diag.sum() == 0

    def test_normalize_true(self):
        cm = confusion_matrix_array(
            _Y_PERFECT, _Y_PERFECT, n_classes=N_CLASSES, normalize="true"
        )
        # Every row should sum to 1 (or 0 if class has no true samples).
        row_sums = cm.sum(axis=1)
        for rs in row_sums:
            assert rs == pytest.approx(1.0) or rs == pytest.approx(0.0)

    def test_normalize_pred(self):
        cm = confusion_matrix_array(
            _Y_PERFECT, _Y_PERFECT, n_classes=N_CLASSES, normalize="pred"
        )
        col_sums = cm.sum(axis=0)
        for cs in col_sums:
            assert cs == pytest.approx(1.0) or cs == pytest.approx(0.0)

    def test_no_normalize_integer_counts(self):
        cm = confusion_matrix_array(_Y_PERFECT, _Y_PERFECT, n_classes=N_CLASSES)
        # Raw counts are integers.
        assert np.issubdtype(cm.dtype, np.integer)

    def test_total_count_matches_n_samples(self):
        cm = confusion_matrix_array(_Y_TRUE_NOISY, _Y_PRED_NOISY, n_classes=N_CLASSES)
        assert cm.sum() == len(_Y_TRUE_NOISY)

    def test_noisy_off_diagonal_nonzero(self):
        cm = confusion_matrix_array(_Y_TRUE_NOISY, _Y_PRED_NOISY, n_classes=N_CLASSES)
        off_diag = cm - np.diag(np.diag(cm))
        assert off_diag.sum() > 0

    def test_accepts_lists(self):
        cm = confusion_matrix_array(
            list(_Y_PERFECT), list(_Y_PERFECT), n_classes=N_CLASSES
        )
        assert cm.shape == (N_CLASSES, N_CLASSES)


# ---------------------------------------------------------------------------
# Tests: classification_report_str
# ---------------------------------------------------------------------------
class TestClassificationReportStr:
    def test_returns_string(self):
        s = classification_report_str(_Y_PERFECT, _Y_PERFECT, CLASS_NAMES)
        assert isinstance(s, str)

    def test_contains_all_class_names(self):
        s = classification_report_str(_Y_PERFECT, _Y_PERFECT, CLASS_NAMES)
        for cls in CLASS_NAMES:
            assert cls in s

    def test_contains_macro_avg(self):
        s = classification_report_str(_Y_PERFECT, _Y_PERFECT, CLASS_NAMES)
        assert "macro avg" in s

    def test_digits_parameter(self):
        s = classification_report_str(_Y_PERFECT, _Y_PERFECT, CLASS_NAMES, digits=4)
        assert isinstance(s, str)

    def test_noisy_predictions_still_produces_report(self):
        s = classification_report_str(_Y_TRUE_NOISY, _Y_PRED_NOISY, CLASS_NAMES)
        assert len(s) > 0
