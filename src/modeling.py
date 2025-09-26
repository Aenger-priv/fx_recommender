"""NumPy-based MLP for multi-target regression with dropout and Adam.

This module implements a lightweight neural network for learning per-strategy
rewards using masked mean squared error. It supports Monte Carlo (MC) dropout
for uncertainty estimation and provides utilities to save and load trained
artifacts. Designed to be dependency-light (NumPy only) and compatible with
non-stationary settings via time-aware training in the surrounding pipeline.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator):
    """He (Kaiming) normal initializer for ReLU networks.

    Args:
        fan_in: Number of input units to the layer.
        fan_out: Number of output units from the layer.
        rng: NumPy random generator used for reproducibility.

    Returns:
        Tuple of weight matrix ``W`` with shape ``[fan_in, fan_out]`` and bias
        row ``b`` with shape ``[1, fan_out]`` in ``float32``.
    """
    std = np.sqrt(2.0 / max(1, fan_in))
    W = rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float32)
    b = np.zeros((1, fan_out), dtype=np.float32)
    return W, b


@dataclass
class MLPConfig:
    """Configuration for the MLPRegressor.

    Attributes:
        input_dim: Number of input features.
        output_dim: Number of targets (e.g., strategies).
        hidden: Sizes of hidden layers.
        dropout: Dropout rate applied to hidden activations.
        seed: Random seed for reproducibility.
    """

    input_dim: int
    output_dim: int
    hidden: List[int]
    dropout: float = 0.2
    seed: int = 42


class MLPRegressor:
    """Minimal MLP for masked multi-target regression with MC dropout.

    The model uses ReLU hidden layers, optional dropout on hidden activations,
    a linear output layer sized to the number of strategies, and Adam for
    optimization. Loss is a masked MSE where the mask selects observed targets
    (e.g., observed rewards for specific strategies).
    """

    def __init__(self, cfg: MLPConfig):
        """Initialize parameters and optimizer state from config.

        Args:
            cfg: Model configuration specifying layer sizes, dropout, and seed.
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        dims = [cfg.input_dim] + cfg.hidden + [cfg.output_dim]
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(len(dims) - 1):
            W, b = he_init(dims[i], dims[i + 1], self.rng)
            self.W.append(W)
            self.b.append(b)
        # Adam states
        self.mW = [np.zeros_like(W) for W in self.W]
        self.vW = [np.zeros_like(W) for W in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(
        self, X: np.ndarray, training: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward pass through all layers.

        Args:
            X: Input matrix with shape ``[batch, input_dim]``.
            training: If True, apply dropout to hidden activations.

        Returns:
            - List of activations per layer (including input and final output).
            - List of dropout masks per layer (ones for non-dropout layers).
        """
        a = X
        activations = [a]
        masks = []
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                a = np.maximum(z, 0.0)
                if training and self.cfg.dropout > 0.0:
                    keep_prob = 1.0 - self.cfg.dropout
                    mask = (self.rng.random(a.shape) < keep_prob).astype(
                        np.float32
                    ) / keep_prob
                    a = a * mask
                    masks.append(mask)
                else:
                    masks.append(np.ones_like(a))
            else:
                a = z
                masks.append(np.ones_like(a))
            activations.append(a)
        return activations, masks

    @staticmethod
    def mse_masked(
        y_pred: np.ndarray, y_true: np.ndarray, sample_weight: np.ndarray
    ) -> float:
        """Compute masked MSE.

        Args:
            y_pred: Predicted targets with shape ``[batch, output_dim]``.
            y_true: Ground-truth targets with shape ``[batch, output_dim]``.
            sample_weight: Element-wise mask/weights with same shape as
                ``y_true``; zeros indicate missing targets.

        Returns:
            Scalar masked mean squared error.
        """
        diff = y_pred - y_true
        se = diff * diff
        wse = se * sample_weight
        denom = np.sum(sample_weight) + 1e-8
        return float(np.sum(wse) / denom)

    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        activations: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backpropagate masked MSE loss and compute gradients.

        Args:
            X: Input batch with shape ``[batch, input_dim]``.
            y: Target batch with shape ``[batch, output_dim]``.
            w: Mask/weights with shape ``[batch, output_dim]``.
            activations: Activations from ``forward`` (including output layer).
            masks: Dropout masks from ``forward`` aligned to hidden layers.

        Returns:
            Tuple ``(dW, dB)`` where each list item matches the parameter
            shapes for the corresponding layer.
        """
        y_pred = activations[-1]
        # dL/dy_pred for masked MSE: 2 * (y_pred - y) * w / sum(w)
        denom = np.sum(w) + 1e-8
        dA = 2.0 * (y_pred - y) * w / denom
        dW = [np.zeros_like(W) for W in self.W]
        dB = [np.zeros_like(b) for b in self.b]

        for i in reversed(range(len(self.W))):
            A_prev = activations[i]
            dW[i] = A_prev.T @ dA
            dB[i] = np.sum(dA, axis=0, keepdims=True)
            if i > 0:
                dA = dA @ self.W[i].T
                Z_prev = activations[i]  # post-activation of previous layer
                relu_grad = (Z_prev > 0).astype(np.float32)
                dA = dA * relu_grad
                dA = dA * masks[i - 1]
        return dW, dB

    def adam_step(
        self,
        dW: List[np.ndarray],
        dB: List[np.ndarray],
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        """Apply one Adam optimizer step to parameters.

        Args:
            dW: Gradients for weights per layer.
            dB: Gradients for biases per layer.
            lr: Learning rate.
            beta1: Exponential decay rate for the first moment estimates.
            beta2: Exponential decay rate for the second moment estimates.
            eps: Numerical stability epsilon.
        """
        self.t += 1
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (dW[i] ** 2)
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * dB[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (dB[i] ** 2)

            mW_hat = self.mW[i] / (1 - beta1**self.t)
            vW_hat = self.vW[i] / (1 - beta2**self.t)
            mb_hat = self.mb[i] / (1 - beta1**self.t)
            vb_hat = self.vb[i] / (1 - beta2**self.t)

            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def fit(
        self,
        X_tr: np.ndarray,
        Y_tr: np.ndarray,
        W_tr: np.ndarray,
        X_va: np.ndarray,
        Y_va: np.ndarray,
        W_va: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train with mini-batch Adam and early stopping on validation loss.

        Args:
            X_tr: Training inputs ``[N, input_dim]``.
            Y_tr: Training targets ``[N, output_dim]``.
            W_tr: Training masks/weights ``[N, output_dim]``.
            X_va: Validation inputs ``[M, input_dim]``.
            Y_va: Validation targets ``[M, output_dim]``.
            W_va: Validation masks/weights ``[M, output_dim]``.
            epochs: Maximum epochs.
            batch_size: Mini-batch size.
            lr: Learning rate for Adam.
            patience: Early stopping patience on validation loss.
            verbose: If True, prints training/validation losses per epoch.

        Returns:
            Dict with keys:
            - ``best_val``: Best validation loss achieved.
            - ``history``: Dict containing per-epoch ``train_loss`` and
              ``val_loss`` lists.
        """
        n = X_tr.shape[0]
        idx = np.arange(n)
        best_val = float("inf")
        best_weights = None
        since_improve = 0
        hist = {"val_loss": [], "train_loss": []}
        for ep in range(1, epochs + 1):
            self.rng.shuffle(idx)
            X_tr_sh = X_tr[idx]
            Y_tr_sh = Y_tr[idx]
            W_tr_sh = W_tr[idx]
            # mini-batch
            tot_loss = 0.0
            tot_w = 0.0
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                Xb = X_tr_sh[start:end]
                Yb = Y_tr_sh[start:end]
                Wb = W_tr_sh[start:end]
                A, masks = self.forward(Xb, training=True)
                loss = self.mse_masked(A[-1], Yb, Wb)
                dW, dB = self.backward(Xb, Yb, Wb, A, masks)
                self.adam_step(dW, dB, lr=lr)
                tot_loss += loss * np.sum(Wb)
                tot_w += np.sum(Wb)
            train_loss = float(tot_loss / (tot_w + 1e-8))

            # Validation
            A_va, _ = self.forward(X_va, training=False)
            val_loss = self.mse_masked(A_va[-1], Y_va, W_va)
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)
            if verbose:
                print(
                    f"Epoch {ep}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                )

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_weights = (
                    [(w.copy()) for w in self.W],
                    [(b.copy()) for b in self.b],
                )
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= patience:
                    if verbose:
                        print("Early stopping")
                    break

        if best_weights is not None:
            self.W, self.b = best_weights[0], best_weights[1]
        return {"best_val": best_val, "history": hist}

    def predict(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Produce predictions with an optional dropout-enabled forward pass.

        Args:
            X: Input matrix ``[batch, input_dim]``.
            training: If True, keeps dropout active (useful for MC dropout).

        Returns:
            Predicted targets with shape ``[batch, output_dim]``.
        """
        A, _ = self.forward(X, training=training)
        return A[-1]

    def mc_predict(
        self, X: np.ndarray, passes: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Monte Carlo dropout predictions.

        Runs multiple stochastic forward passes with dropout enabled and
        aggregates the predictions.

        Args:
            X: Input matrix ``[batch, input_dim]``.
            passes: Number of stochastic passes to average.

        Returns:
            Tuple ``(mean, std)`` where each has shape ``[batch, output_dim]``.
        """
        preds = []
        for _ in range(max(1, passes)):
            preds.append(self.predict(X, training=True))
        stacked = np.stack(preds, axis=0)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        return mean, std

    def save(self, model_dir: str):
        """Persist weights and config to a directory.

        Saves parameters to ``weights.npz`` and configuration to ``model.json``
        under ``model_dir``. Creates the directory if it does not exist.

        Args:
            model_dir: Target directory for serialized artifacts.
        """
        import json
        from pathlib import Path

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        np.savez(
            Path(model_dir) / "weights.npz",
            **{f"W{i}": W for i, W in enumerate(self.W)},
            **{f"b{i}": b for i, b in enumerate(self.b)},
        )
        with open(Path(model_dir) / "model.json", "w") as f:
            json.dump(
                {
                    "input_dim": self.cfg.input_dim,
                    "output_dim": self.cfg.output_dim,
                    "hidden": self.cfg.hidden,
                    "dropout": self.cfg.dropout,
                    "seed": self.cfg.seed,
                },
                f,
            )

    @staticmethod
    def load(model_dir: str) -> "MLPRegressor":
        """Load weights and config from a directory and construct a model.

        Args:
            model_dir: Directory containing ``weights.npz`` and ``model.json``.

        Returns:
            An ``MLPRegressor`` instance with restored parameters.
        """
        import json
        from pathlib import Path

        with open(Path(model_dir) / "model.json", "r") as f:
            cfg_d = json.load(f)
        cfg = MLPConfig(**cfg_d)
        model = MLPRegressor(cfg)
        data = np.load(Path(model_dir) / "weights.npz")
        for i in range(len(model.W)):
            model.W[i] = data[f"W{i}"]
            model.b[i] = data[f"b{i}"]
        return model
