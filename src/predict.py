"""Prediction utilities: load artifacts and score a new context."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .data import Preprocessor
from .modeling import MLPRegressor


def load_artifacts(model_dir: str):
    """Load preprocessor, model, and metadata from a model directory.

    Returns:
        ((Preprocessor, MLPRegressor), metadata_dict)
    """
    with open(Path(model_dir) / "metadata.json", "r") as f:
        meta = json.load(f)
    with open(Path(model_dir) / "preprocessor.json", "r") as f:
        preproc_dict = json.load(f)
    preproc = Preprocessor.from_dict(preproc_dict)
    model = MLPRegressor.load(model_dir)
    return (preproc, model), meta


def predict_context(
    artifacts, meta: Dict[str, Any], context: Dict[str, Any], mc_passes: int = 30
) -> Dict[str, Any]:
    """Predict per-strategy rewards and choose the best strategy.

    Uses Monte Carlo dropout to produce mean estimates and uncertainty (std).

    Args:
        artifacts: Tuple of (Preprocessor, MLPRegressor) from load_artifacts.
        meta: Metadata dict containing the strategy mapping.
        context: Input context attributes.
        mc_passes: Number of stochastic forward passes for MC dropout.

    Returns:
        Dict with best strategy, estimated_reward, confidence, and a top list.
    """
    preproc, model = artifacts
    strat_map = meta["strategy_map"]
    inv_strats = {v: k for k, v in strat_map.items()}

    X = preproc.transform([context])
    mean, std = model.mc_predict(X, passes=mc_passes)
    mean_np = mean.reshape(-1)
    std_np = std.reshape(-1)

    best_idx = int(np.argmax(mean_np))
    best_strategy = inv_strats.get(best_idx, str(best_idx))
    best_reward = float(mean_np[best_idx])
    uncertainty = float(std_np[best_idx])
    confidence = float(1.0 / (1.0 + max(1e-8, uncertainty)))

    top_indices = list(np.argsort(-mean_np)[:3])
    top = [
        {
            "strategy": inv_strats.get(int(i), str(int(i))),
            "est_reward": float(mean_np[i]),
            "uncertainty": float(std_np[i]),
            "confidence": float(1.0 / (1.0 + float(std_np[i]))),
        }
        for i in top_indices
    ]

    return {
        "strategy": best_strategy,
        "estimated_reward": best_reward,
        "confidence": confidence,
        "top": top,
    }
