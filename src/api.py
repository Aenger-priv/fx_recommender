"""FastAPI service exposing prediction endpoints.

Run locally:
  uv run uvicorn src.api:app --reload --port 8000

Or via the helper:
  uv run fxrec-api

Environment variables:
  FXREC_MODEL_DIR: default model directory (default: models/latest)
  FXREC_HOST / FXREC_PORT: host/port for fxrec-api helper
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import load_artifacts, predict_context


DEFAULT_MODEL_DIR = os.environ.get("FXREC_MODEL_DIR", "models/latest")


class PredictRequest(BaseModel):
    attributes: Dict[str, Any] = Field(..., description="Trade context attributes")
    mc_passes: int = Field(30, ge=1, le=500, description="MC Dropout passes")
    model_dir: Optional[str] = Field(
        None, description="Optional override for model directory"
    )


class PredictResponse(BaseModel):
    strategy: str
    estimated_reward: float
    confidence: float
    top: list[dict]


app = FastAPI(title="FX Recommender API", version="0.1.0")


@lru_cache(maxsize=32)
def _cached_load(model_dir: str):
    return load_artifacts(model_dir)


@app.get("/health")
def health():
    return {"status": "ok", "model_dir": DEFAULT_MODEL_DIR}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model_dir = req.model_dir or DEFAULT_MODEL_DIR
    try:
        artifacts, meta = _cached_load(model_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")
    try:
        result = predict_context(artifacts, meta, req.attributes, mc_passes=req.mc_passes)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.post("/reload")
def reload(model_dir: Optional[str] = None):
    """Clear the cache and optionally set a new default model_dir via query."""
    global DEFAULT_MODEL_DIR
    if model_dir:
        DEFAULT_MODEL_DIR = model_dir
    _cached_load.cache_clear()
    return {"reloaded": True, "model_dir": DEFAULT_MODEL_DIR}


def run():
    """Run a development server via uvicorn (console script helper)."""
    import uvicorn

    host = os.environ.get("FXREC_HOST", "0.0.0.0")
    port = int(os.environ.get("FXREC_PORT", "8000"))
    uvicorn.run("src.api:app", host=host, port=port, reload=False)

