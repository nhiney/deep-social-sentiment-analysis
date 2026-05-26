"""Unit tests for the FastAPI inference service (app/main.py).

Tests run WITHOUT a real model checkpoint by monkey-patching _load_artifacts
and injecting a stub predictor. This keeps the test suite fast and dependency-
free (no GPU, no HuggingFace download).

Run::

    pytest tests/test_api.py -v
"""

from __future__ import annotations

import types
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Stub predictor — mimics LateFusionPredictor without loading any weights
# ---------------------------------------------------------------------------
CLASS_NAMES = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]


class _StubPredictor:
    """Returns deterministic fake predictions for any input text."""

    class_names = CLASS_NAMES

    def predict(
        self,
        texts: Sequence[str],
        tabular_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for _ in texts:
            probs = {n: 1.0 / len(CLASS_NAMES) for n in CLASS_NAMES}
            probs["joy"] = 0.5
            probs["sadness"] = 0.1
            # Renormalize
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            results.append(
                {"label": "joy", "confidence": 0.5, "probs": probs}
            )
        return results

    def predict_proba_for_lime(self, texts: Sequence[str]) -> np.ndarray:
        n = len(texts)
        arr = np.full((n, len(CLASS_NAMES)), 1.0 / len(CLASS_NAMES))
        arr[:, 0] = 0.5
        arr = arr / arr.sum(axis=1, keepdims=True)
        return arr


class _StubExplainer:
    """Returns a deterministic stub ExplanationResult."""

    num_samples = 200

    def explain(self, text: str, target_label=None):
        result = types.SimpleNamespace(
            label="joy",
            label_id=0,
            confidence=0.5,
            tokens=[("vui", 0.12), ("quá", 0.08)],
            highlight_html="<div>stub html</div>",
            n_samples=200,
        )
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """TestClient with stub predictor injected."""
    import app.main as main_module

    # Patch before the app processes any request.
    main_module._predictor = _StubPredictor()
    main_module._explainer = _StubExplainer()

    with TestClient(main_module.app) as c:
        yield c

    # Clean up module state so other test modules are not affected.
    main_module._predictor = None
    main_module._explainer = None


# ---------------------------------------------------------------------------
# Tests: /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_model_loaded_true(self, client):
        data = resp = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_class_names_present(self, client):
        names = client.get("/health").json()["class_names"]
        assert set(names) == set(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Tests: /predict
# ---------------------------------------------------------------------------
class TestPredict:
    def test_basic_prediction(self, client):
        resp = client.post("/predict", json={"text": "Hôm nay vui quá!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in CLASS_NAMES
        assert 0.0 <= data["confidence"] <= 1.0
        assert set(data["probs"].keys()) == set(CLASS_NAMES)

    def test_probs_sum_to_one(self, client):
        resp = client.post("/predict", json={"text": "Buồn quá đi"})
        probs = list(resp.json()["probs"].values())
        assert abs(sum(probs) - 1.0) < 1e-4

    def test_with_num_feature_overrides(self, client):
        payload = {
            "text": "Tức quá!",
            "num_features": {"likes": 500, "comments": 120},
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_with_cat_feature_overrides(self, client):
        payload = {
            "text": "Sợ lắm!",
            "cat_features": {"has_hashtag": "yes"},
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_blank_text_rejected(self, client):
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 422

    def test_missing_text_rejected(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_explanation_field_absent_by_default(self, client):
        data = client.post("/predict", json={"text": "test"}).json()
        assert data.get("explanation") is None


# ---------------------------------------------------------------------------
# Tests: /predict/batch
# ---------------------------------------------------------------------------
class TestBatchPredict:
    def test_basic_batch(self, client):
        texts = ["Vui lắm", "Buồn quá", "Tức giận"]
        resp = client.post("/predict/batch", json={"texts": texts})
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_texts"] == len(texts)
        assert len(data["predictions"]) == len(texts)

    def test_each_prediction_has_probs(self, client):
        resp = client.post(
            "/predict/batch",
            json={"texts": ["text1", "text2"]},
        )
        for pred in resp.json()["predictions"]:
            assert set(pred["probs"].keys()) == set(CLASS_NAMES)

    def test_max_batch_exceeded(self, client):
        texts = [f"text{i}" for i in range(65)]
        resp = client.post("/predict/batch", json={"texts": texts})
        assert resp.status_code == 422

    def test_empty_texts_list_rejected(self, client):
        resp = client.post("/predict/batch", json={"texts": []})
        assert resp.status_code == 422

    def test_custom_batch_size(self, client):
        texts = ["a", "b", "c", "d", "e"]
        resp = client.post(
            "/predict/batch",
            json={"texts": texts, "batch_size": 2},
        )
        assert resp.status_code == 200
        assert resp.json()["n_texts"] == 5


# ---------------------------------------------------------------------------
# Tests: /predict/explain
# ---------------------------------------------------------------------------
class TestPredictExplain:
    def test_basic_explain(self, client):
        resp = client.post("/predict/explain", json={"text": "Vui vẻ lắm!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["explanation"] is not None

    def test_explanation_has_required_keys(self, client):
        resp = client.post("/predict/explain", json={"text": "Test text"})
        exp = resp.json()["explanation"]
        for key in ("label", "label_id", "confidence", "tokens", "highlight_html", "n_samples"):
            assert key in exp, f"Missing key: {key}"

    def test_tokens_are_pairs(self, client):
        resp = client.post("/predict/explain", json={"text": "abc"})
        tokens = resp.json()["explanation"]["tokens"]
        for pair in tokens:
            assert len(pair) == 2
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], float)

    def test_num_samples_respected(self, client):
        resp = client.post(
            "/predict/explain",
            json={"text": "Text", "num_samples": 100},
        )
        assert resp.status_code == 200

    def test_target_label_accepted(self, client):
        resp = client.post(
            "/predict/explain",
            json={"text": "Tức giận lắm", "target_label": "anger"},
        )
        assert resp.status_code == 200

    def test_num_samples_out_of_range(self, client):
        resp = client.post(
            "/predict/explain",
            json={"text": "text", "num_samples": 10},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests: degraded mode (no model loaded)
# ---------------------------------------------------------------------------
class TestDegradedMode:
    @pytest.fixture
    def degraded_client(self, monkeypatch):
        import app.main as main_module

        # Point to a non-existent checkpoint so the lifespan handler fails and
        # leaves _predictor = None regardless of whether a real checkpoint exists.
        monkeypatch.setenv("MODEL_CHECKPOINT", "/nonexistent/path/for/testing")
        with TestClient(main_module.app) as c:
            yield c

    def test_health_degraded(self, degraded_client):
        data = degraded_client.get("/health").json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_predict_returns_503(self, degraded_client):
        resp = degraded_client.post("/predict", json={"text": "test"})
        assert resp.status_code == 503

    def test_batch_returns_503(self, degraded_client):
        resp = degraded_client.post(
            "/predict/batch", json={"texts": ["test"]}
        )
        assert resp.status_code == 503
