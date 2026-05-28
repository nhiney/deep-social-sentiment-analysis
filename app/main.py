"""FastAPI inference service for the Late Fusion sentiment model.

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
GET  /health        → liveness probe + model status
POST /predict       → single-text inference
POST /predict/batch → batch inference (up to 64 texts)
POST /predict/explain → single-text inference + LIME token attribution
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast when the model dir doesn't exist yet.
# ---------------------------------------------------------------------------
_predictor = None   # type: Optional[Any]   # LateFusionPredictor
_explainer = None   # type: Optional[Any]   # TextExplainer

CLASS_NAMES = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
_CHECKPOINT_ENV = "MODEL_CHECKPOINT"
_DEFAULT_CHECKPOINT = "models/best_model"


# --------------------------------------------------------------------------- #
# I/O schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    """Request payload for single-text inference."""

    text: str = Field(..., min_length=1, description="Raw social-media post (Vietnamese / code-switched).")
    num_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional behavioral feature overrides (e.g. {'likes': 120}).",
    )
    cat_features: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional categorical feature overrides (e.g. {'has_hashtag': 'yes'}).",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank.")
        return v


class BatchPredictRequest(BaseModel):
    """Request payload for batch inference."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of raw posts (max 64).",
    )
    batch_size: int = Field(default=16, ge=1, le=64)

    @field_validator("texts")
    @classmethod
    def max_batch(cls, v: List[str]) -> List[str]:
        if len(v) > 64:
            raise ValueError("Maximum batch size is 64 texts.")
        return v


class ExplainRequest(BaseModel):
    """Request payload for LIME explanation."""

    text: str = Field(..., min_length=1)
    target_label: Optional[str] = Field(
        default=None,
        description="Class to attribute against. Defaults to the top prediction.",
    )
    num_samples: int = Field(
        default=200, ge=50, le=1000,
        description="Number of LIME perturbation samples.",
    )


class PredictResponse(BaseModel):
    """Single-text prediction response."""

    label: str
    confidence: float
    probs: Dict[str, float]
    explanation: Optional[Dict[str, Any]] = None


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictResponse]
    n_texts: int


class HealthResponse(BaseModel):
    """Liveness / readiness probe response."""

    status: str
    model_loaded: bool
    checkpoint: str
    class_names: List[str]


# --------------------------------------------------------------------------- #
# App lifecycle
# --------------------------------------------------------------------------- #
@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI):
    """FastAPI lifespan handler: load model at startup, release at shutdown."""
    global _predictor, _explainer

    checkpoint = os.environ.get(_CHECKPOINT_ENV, _DEFAULT_CHECKPOINT)
    logger.info("Loading model from: %s", checkpoint)

    try:
        # Deferred import so the module loads even without torch installed.
        from app.explainer import TextExplainer
        from app.inference import LateFusionPredictor

        _predictor = LateFusionPredictor(
            checkpoint_dir=checkpoint,
            class_names=CLASS_NAMES,
            device="auto",
            max_length=128,
            apply_normalizer=True,
        )
        _explainer = TextExplainer(
            class_names=CLASS_NAMES,
            predict_proba_fn=_predictor.predict_proba_for_lime,
            num_samples=200,
        )
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.warning(
            "Model could not be loaded (%s). "
            "Server is running in degraded mode — inference endpoints will return 503.",
            exc,
        )
        _predictor = None
        _explainer = None

    yield  # application runs here

    # Shutdown: release GPU memory if the model is loaded.
    if _predictor is not None:
        try:
            import torch
            if hasattr(_predictor, "model"):
                _predictor.model.cpu()
                del _predictor.model
            torch.cuda.empty_cache()
        except Exception:
            pass
    _predictor = None
    _explainer = None


app = FastAPI(
    title="Deep Social Sentiment API",
    description=(
        "Late Fusion XLM-RoBERTa + FT-Transformer model for 7-class Vietnamese emotion classification. "
        "Supports single / batch inference and LIME token-level explanations."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

# Serve report figures at /figures/<filename>
_FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
if _FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(_FIGURES_DIR)), name="figures")

@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


def _require_predictor():
    """Raise 503 if the model is not loaded."""
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded. "
                f"Set the {_CHECKPOINT_ENV!r} env var to a valid checkpoint directory "
                f"and restart the server. Default path: {_DEFAULT_CHECKPOINT!r}."
            ),
        )
    return _predictor


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    """Liveness / readiness probe.

    Returns ``status: "ok"`` when the model is loaded, ``status: "degraded"``
    when the server is running but the checkpoint was not found at startup.
    The HTTP status code is always 200 — use ``model_loaded`` to decide
    whether to route traffic.
    """
    checkpoint = os.environ.get(_CHECKPOINT_ENV, _DEFAULT_CHECKPOINT)
    return HealthResponse(
        status="ok" if _predictor is not None else "degraded",
        model_loaded=_predictor is not None,
        checkpoint=checkpoint,
        class_names=CLASS_NAMES,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    """Run end-to-end inference on a single post.

    * ``text``         — raw Vietnamese post (teencode normalization applied automatically).
    * ``num_features`` — optional dict to override auto-derived behavioral features
      (e.g. supply real ``likes`` counts scraped from Facebook).
    * ``cat_features`` — optional categorical overrides.
    """
    predictor = _require_predictor()

    overrides = {**payload.num_features, **payload.cat_features}

    try:
        result = predictor.predict([payload.text], tabular_overrides=overrides or None)[0]
    except Exception as exc:
        logger.exception("Inference error for text: %.80s", payload.text)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    return PredictResponse(
        label=result["label"],
        confidence=round(result["confidence"], 6),
        probs={k: round(v, 6) for k, v in result["probs"].items()},
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    """Run inference on a list of posts (up to 64).

    Processes texts in mini-batches of ``batch_size`` (default 16) to
    avoid OOM on the GPU. Tabular features are auto-derived from each text.
    """
    predictor = _require_predictor()

    texts = [t for t in payload.texts if t and t.strip()]
    if not texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="All texts are blank after stripping whitespace.",
        )

    try:
        results = []
        bs = payload.batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            results.extend(predictor.predict(chunk))
    except Exception as exc:
        logger.exception("Batch inference error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {exc}",
        ) from exc

    predictions = [
        PredictResponse(
            label=r["label"],
            confidence=round(r["confidence"], 6),
            probs={k: round(v, 6) for k, v in r["probs"].items()},
        )
        for r in results
    ]
    return BatchPredictResponse(predictions=predictions, n_texts=len(predictions))


@app.post("/predict/explain", response_model=PredictResponse, tags=["explainability"])
def predict_with_explanation(payload: ExplainRequest) -> PredictResponse:
    """Run inference + LIME token-level attribution on a single post.

    LIME explanation is returned inside the ``explanation`` field:

    ```json
    {
      "label": "anger",
      "confidence": 0.83,
      "probs": {...},
      "explanation": {
        "label": "anger",
        "confidence": 0.83,
        "tokens": [["từ_mạnh", 0.12], ["chửi", 0.09], ...],
        "highlight_html": "<div>...</div>",
        "n_samples": 200
      }
    }
    ```

    **Note**: generating a LIME explanation takes ~2–5 s with 200 samples.
    Increase ``num_samples`` only when you need stable attributions for
    research-quality visualizations.
    """
    predictor = _require_predictor()
    if _explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LIME explainer is not initialized.",
        )

    # Update the explainer's sample count for this request.
    _explainer.num_samples = payload.num_samples

    try:
        result = predictor.predict([payload.text])[0]
        explanation = _explainer.explain(
            payload.text,
            target_label=payload.target_label,
        )
    except Exception as exc:
        logger.exception("Explain error for text: %.80s", payload.text)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {exc}",
        ) from exc

    return PredictResponse(
        label=result["label"],
        confidence=round(result["confidence"], 6),
        probs={k: round(v, 6) for k, v in result["probs"].items()},
        explanation={
            "label":          explanation.label,
            "label_id":       explanation.label_id,
            "confidence":     round(explanation.confidence, 6),
            "tokens":         explanation.tokens,
            "highlight_html": explanation.highlight_html,
            "n_samples":      explanation.n_samples,
        },
    )
