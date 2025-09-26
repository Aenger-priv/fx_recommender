"""Evaluation utilities for the FX strategy recommender.

Provides evaluation of a saved model on a held-out time-based test split.
Reports masked MSE for regression quality, coverage (fraction of decisions
where the predicted top strategy had an observed reward), top-1 accuracy among
observed strategies, and regret statistics based on realized rewards.
"""

import json
from typing import Dict, Any, List

import numpy as np

from .data import load_dataset, split_by_time, latest_date, Preprocessor, build_xyw
from .predict import load_artifacts


def _masked_mse(y_pred: np.ndarray, y_true: np.ndarray, w: np.ndarray) -> float:
    """Compute element-wise masked MSE for multi-target regression.

    Args:
        y_pred: Predicted targets with shape ``[N, D]``.
        y_true: Ground truth targets with shape ``[N, D]``.
        w: Element-wise mask/weights with shape ``[N, D]``; zeros indicate
            unobserved targets.

    Returns:
        Scalar masked mean squared error over all elements.
    """
    diff = y_pred - y_true
    return float(np.sum((diff * diff) * w) / (np.sum(w) + 1e-8))


def evaluate_and_report(
    data_path: str,
    model_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> Dict[str, Any]:
    """Evaluate a saved model on the test split and return a report.

    The dataset is split in chronological order into train/val/test using the
    same logic as training. The test set is transformed with the saved
    preprocessor and evaluated using masked metrics that respect the set of
    observed strategies per example.

    Args:
        data_path: Path to the dataset JSON file.
        model_dir: Directory containing saved model and preprocessor artifacts.
        val_split: Fraction of data assigned to validation (from the end).
        test_split: Fraction of data assigned to test (from the end).

    Returns:
        Dict with summary metrics, including keys:
        - ``samples_test``: Number of test examples.
        - ``masked_mse``: Masked mean squared error on test.
        - ``coverage``: Fraction where predicted top strategy had an observed
          reward in the example.
        - ``top1_accuracy``: Accuracy among covered examples where the
          predicted top strategy equals the true best observed strategy.
        - ``mean_regret``/``p50_regret``/``p90_regret``: Regret statistics in
          realized reward units, computed only for covered examples.
        - ``avg_true_best_reward``: Average of the best observed rewards.
        - ``avg_predicted_strategy_actual_reward``: Average realized reward of
          the predicted top strategy among covered examples.
    """
    (preproc, model), meta = load_artifacts(model_dir)
    strat_map: Dict[str, int] = meta["strategy_map"]
    inv_strats = {v: k for k, v in strat_map.items()}

    data = load_dataset(data_path)
    train_ex, val_ex, test_ex = split_by_time(data, val_split, test_split)

    # Build XYW for test (alpha=0.0 -> weights reduce to mask)
    latest = latest_date(data)
    X_te, Y_te, W_te = build_xyw(test_ex, strat_map, preproc, latest, alpha=0.0)

    # Predictions
    Y_pred = model.predict(X_te)

    # Regression metric
    masked_mse = _masked_mse(Y_pred, Y_te, W_te)

    # Strategy selection metrics
    n = len(test_ex)
    covered = 0
    correct = 0
    regrets: List[float] = []
    avg_best_rewards: List[float] = []
    avg_pred_rewards: List[float] = []

    for i, ex in enumerate(test_ex):
        rewards = ex.get("rewards", {})
        if not rewards:
            continue
        # True best among available
        items = [(strat_map[k], float(v)) for k, v in rewards.items() if k in strat_map]
        if not items:
            continue
        true_best_idx, true_best_val = max(items, key=lambda t: t[1])
        avg_best_rewards.append(true_best_val)

        # Predicted best
        pred_vec = Y_pred[i]
        pred_best_idx = int(np.argmax(pred_vec))
        # Only evaluate top-1/ret if the predicted strategy has an observed reward
        if any(idx == pred_best_idx for idx, _ in items):
            covered += 1
            pred_actual_val = dict(items)[pred_best_idx]
            avg_pred_rewards.append(pred_actual_val)
            if pred_best_idx == true_best_idx:
                correct += 1
            regrets.append(true_best_val - pred_actual_val)

    coverage = covered / max(1, n)
    top1_acc = correct / max(1, covered)
    mean_regret = float(np.mean(regrets)) if regrets else None
    p50_regret = float(np.percentile(regrets, 50)) if regrets else None
    p90_regret = float(np.percentile(regrets, 90)) if regrets else None
    avg_true_best = float(np.mean(avg_best_rewards)) if avg_best_rewards else None
    avg_pred_reward = float(np.mean(avg_pred_rewards)) if avg_pred_rewards else None

    report = {
        "samples_test": n,
        "masked_mse": masked_mse,
        "coverage": coverage,
        "top1_accuracy": top1_acc,
        "mean_regret": mean_regret,
        "p50_regret": p50_regret,
        "p90_regret": p90_regret,
        "avg_true_best_reward": avg_true_best,
        "avg_predicted_strategy_actual_reward": avg_pred_reward,
    }
    return report
