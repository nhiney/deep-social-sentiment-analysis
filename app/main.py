"""FastAPI inference service for the Late Fusion sentiment model.

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.models import LateFusionModel


# --------------------------------------------------------------------------- #
# I/O schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    """Request payload combining text and tabular behavior features."""

    text: str = Field(..., description="Raw social-media post.")
    num_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Numerical features keyed by column name.",
    )
    cat_features: Dict[str, str] = Field(
        default_factory=dict,
        description="Categorical features keyed by column name.",
    )


class PredictResponse(BaseModel):
    """Response payload with predicted label + per-class probabilities."""

    label: str
    probs: Dict[str, float]
    explanation: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="Deep Social Sentiment API", version="0.1.0")


@app.on_event("startup")
def _load_artifacts() -> None:
    """Load the model + preprocessors into module-level state at boot."""
    pass


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe."""
    pass


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Run end-to-end inference on a single example."""
    pass
