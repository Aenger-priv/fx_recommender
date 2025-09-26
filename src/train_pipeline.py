"""Training pipeline that ties data, model, and artifact saving together."""

import json
from pathlib import Path
from typing import List

import numpy as np

from .data import (
    Preprocessor,
    build_strategy_mapping,
    build_xyw,
    infer_feature_types,
    latest_date,
    load_dataset,
    restrict_to_last_n_days,
    split_by_time,
)
from .modeling import MLPConfig, MLPRegressor


def train_and_save(
    data_path: str,
    model_dir: str,
    epochs: int = 20,
    batch_size: int = 256,
    alpha: float = 3.0,
    val_split: float = 0.15,
    test_split: float = 0.15,
    hidden: List[int] = [128, 64],
    dropout: float = 0.2,
    window_days: int = 0,
):
    """Train the NumPy MLP and save artifacts to a directory.

    This function performs a time-based split, fits preprocessing, trains with
    masked, recency-weighted MSE, evaluates on val/test, and writes these files:
    - model.json, weights.npz, preprocessor.json, metadata.json

    Args:
        data_path: Path to the dataset JSON file.
        model_dir: Output directory for model artifacts.
        epochs: Max training epochs (early stopping applies).
        batch_size: Mini-batch size.
        alpha: Recency weighting strength (0 disables).
        val_split: Validation fraction in time order.
        test_split: Test fraction in time order.
        hidden: Hidden layer sizes for the MLP.
        dropout: Dropout rate for hidden layers.
        window_days: If >0, restrict strategy mapping and training window.
    """
    data = load_dataset(data_path)

    # Optional rolling window restriction for training only
    data_for_mapping = (
        restrict_to_last_n_days(data, max(window_days, 0)) if window_days > 0 else data
    )

    contexts = [ex.get("context", {}) for ex in data]
    num_features, cat_features = infer_feature_types(contexts)
    num_features = [f for f in num_features if f != "date"]
    cat_features = [f for f in cat_features if f != "date"]

    strategy_map, strategy_names = build_strategy_mapping(data_for_mapping)
    S = len(strategy_map)

    train_ex, val_ex, test_ex = split_by_time(data, val_split, test_split)

    latest = latest_date(data)

    # Fit preprocessor on train set contexts
    preproc = Preprocessor(num_features, cat_features)
    preproc.fit([ex.get("context", {}) for ex in train_ex])

    # Build XY and sample weights
    X_tr, Y_tr, W_tr = build_xyw(train_ex, strategy_map, preproc, latest, alpha)
    X_va, Y_va, W_va = build_xyw(val_ex, strategy_map, preproc, latest, alpha=0.0)
    X_te, Y_te, W_te = build_xyw(test_ex, strategy_map, preproc, latest, alpha=0.0)

    cfg = MLPConfig(
        input_dim=X_tr.shape[1], output_dim=S, hidden=list(hidden), dropout=dropout
    )
    model = MLPRegressor(cfg)

    hist = model.fit(
        X_tr,
        Y_tr,
        W_tr,
        X_va,
        Y_va,
        W_va,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        patience=5,
        verbose=True,
    )

    # Evaluate on validation and test
    val_pred = model.predict(X_va)
    test_pred = model.predict(X_te)

    def masked_mse(y_pred, y_true, w):
        diff = y_pred - y_true
        return float(np.sum((diff * diff) * w) / (np.sum(w) + 1e-8))

    eval_va = masked_mse(val_pred, Y_va, W_va)
    eval_te = masked_mse(test_pred, Y_te, W_te)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save(model_dir)
    with open(Path(model_dir) / "preprocessor.json", "w") as f:
        json.dump(preproc.to_dict(), f)

    meta = {
        "num_features": num_features,
        "cat_features": cat_features,
        "strategy_map": strategy_map,
        "strategy_names": strategy_names,
        "alpha": alpha,
        "val_loss": float(eval_va),
        "test_loss": float(eval_te),
    }
    with open(Path(model_dir) / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        json.dumps(
            {
                "message": "Training complete",
                "val_loss": meta["val_loss"],
                "test_loss": meta["test_loss"],
                "saved_to": model_dir,
            },
            indent=2,
        )
    )
