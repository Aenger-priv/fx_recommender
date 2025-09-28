"""Data utilities: loading, feature typing, preprocessing, and dataset splits.

Dataset format
- List[dict] where each item has:
  - context: mapping of feature name -> value (numeric or categorical). May contain 'date' (YYYY-MM-DD).
  - rewards: mapping of strategy_id (string) -> float reward.

This module provides a simple NumPy preprocessor (standardize numeric, one-hot categorical),
time-based splitting, and helpers to build training arrays with recency-aware weights.
"""

import datetime as dt
import json
from typing import Any, Dict, List, Tuple

import numpy as np


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load the trading dataset from JSON.

    Args:
        path: Path to a JSON file with a list of examples.

    Returns:
        List of examples, each with 'context' and 'rewards' keys.
    """
    with open(path, "r") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0, "Dataset must be a non-empty list"
    return data


def parse_date(s: str) -> dt.date:
    """Parse a YYYY-MM-DD string into a date."""
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def infer_feature_types(contexts: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Infer numeric vs. categorical context features from example contexts.

    Args:
        contexts: List of context dicts (feature -> value).

    Returns:
        (numeric_feature_names, categorical_feature_names)
    """
    keys = set()
    for c in contexts:
        keys.update(c.keys())
    num_keys, cat_keys = [], []
    for k in sorted(keys):
        # date is used for splitting/weights; keep as categorical if present as a feature
        example_value = next((c[k] for c in contexts if k in c), None)
        if isinstance(example_value, (int, float)):
            num_keys.append(k)
        else:
            cat_keys.append(k)
    return num_keys, cat_keys


def build_strategy_mapping(
    examples: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], List[str]]:
    """Build a mapping from strategy-id string to a contiguous index.

    Args:
        examples: Dataset examples with 'rewards' mappings.

    Returns:
        (mapping dict: strategy->index, list of strategy names ordered by index)
    """
    strategies = set()
    for ex in examples:
        strategies.update(list((ex.get("rewards") or {}).keys()))
    names = sorted(strategies, key=lambda x: int(x) if str(x).isdigit() else str(x))
    mapping = {s: i for i, s in enumerate(names)}
    return mapping, names


def split_by_time(
    examples: List[Dict[str, Any]], val_ratio: float, test_ratio: float
) -> Tuple[List, List, List]:
    """Sort by date and split into train/val/test by ratios.

    If 'context.date' is missing, a default epoch date is used.

    Args:
        examples: Full dataset.
        val_ratio: Fraction assigned to validation.
        test_ratio: Fraction assigned to test.

    Returns:
        (train_examples, val_examples, test_examples)
    """

    def get_date(e):
        d = e.get("context", {}).get("date")
        return parse_date(d) if d else dt.date(1970, 1, 1)

    sorted_ex = sorted(examples, key=get_date)
    n = len(sorted_ex)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    train = sorted_ex[:n_train]
    val = sorted_ex[n_train : n_train + n_val]
    test = sorted_ex[n_train + n_val :]
    return train, val, test


class Preprocessor:
    """NumPy preprocessor: standardize numeric and one-hot encode categoricals.

    Enhancements:
    - Optional per-feature numeric transforms (e.g., log1p for heavy-tailed 'size').
    - Clip standardized numeric z-scores to reduce blow-ups for out-of-range inputs.

    Stores fitted statistics and vocabularies for deterministic transformation
    of new contexts.
    """

    def __init__(
        self,
        num_features: List[str],
        cat_features: List[str],
        z_clip: float = 5.0,
        numeric_transforms: Dict[str, str] | None = None,
    ):
        """Initialize the preprocessor with feature name lists.

        Args:
            num_features: Names of numeric features to standardize.
            cat_features: Names of categorical features to one-hot encode.
            z_clip: Clip standardized numeric values to [-z_clip, z_clip].
            numeric_transforms: Mapping feature -> transform name; supported:
                "standard" (default) or "log1p" (applies log1p(max(0, x))).
        """
        self.num_features = list(num_features)
        self.cat_features = list(cat_features)
        self.z_clip = float(z_clip)
        self.numeric_transforms: Dict[str, str] = dict(numeric_transforms or {})
        self.num_stats: Dict[str, Tuple[float, float]] = {}
        self.cat_vocab: Dict[str, List[str]] = {}
        self.cat_index: Dict[str, Dict[str, int]] = {}

    def fit(self, contexts: List[Dict[str, Any]]):
        """Fit normalization stats and categorical vocabularies from contexts.

        Numeric stats (mean/std) are computed after applying any configured
        per-feature transform (e.g., log1p for heavy-tailed features).
        """
        for f in self.num_features:
            t = self.numeric_transforms.get(f, "standard")
            vals: List[float] = []
            for c in contexts:
                x = float(c.get(f, 0.0))
                if t == "log1p":
                    x = float(np.log1p(max(0.0, x)))
                vals.append(x)
            arr = np.array(vals, dtype=np.float32)
            mu = float(np.mean(arr))
            sigma = float(np.std(arr) + 1e-8)
            if sigma == 0.0:
                sigma = 1.0
            self.num_stats[f] = (mu, sigma)
        for f in self.cat_features:
            vals = [str(c.get(f, "")) for c in contexts]
            uniq = sorted(set(vals))
            self.cat_vocab[f] = uniq
            self.cat_index[f] = {v: i for i, v in enumerate(uniq)}

    def transform(self, contexts: List[Dict[str, Any]]) -> np.ndarray:
        """Transform contexts into a dense design matrix.

        Returns a concatenation of standardized numeric features and one-hot
        categorical features with a fixed column order as per fitted vocab.

        Args:
            contexts: List of input context dicts.

        Returns:
            np.ndarray of shape [n_examples, input_dim].
        """
        n = len(contexts)
        # Numeric block
        num_block = []
        for f in self.num_features:
            mu, sigma = self.num_stats[f]
            t = self.numeric_transforms.get(f, "standard")
            vals: List[float] = []
            for c in contexts:
                x = float(c.get(f, 0.0))
                if t == "log1p":
                    x = float(np.log1p(max(0.0, x)))
                vals.append(x)
            arr = np.array(vals, dtype=np.float32)
            arr = (arr - mu) / sigma
            if self.z_clip > 0:
                arr = np.clip(arr, -self.z_clip, self.z_clip, out=arr)
            num_block.append(arr.reshape(n, 1))
        X_num = (
            np.concatenate(num_block, axis=1)
            if num_block
            else np.zeros((n, 0), dtype=np.float32)
        )

        # Categorical block (one-hot)
        cat_block = []
        for f in self.cat_features:
            vocab = self.cat_vocab[f]
            idx_map = self.cat_index[f]
            onehot = np.zeros((n, len(vocab)), dtype=np.float32)
            for i, c in enumerate(contexts):
                val = str(c.get(f, ""))
                j = idx_map.get(val, None)
                if j is not None:
                    onehot[i, j] = 1.0
            cat_block.append(onehot)
        X_cat = (
            np.concatenate(cat_block, axis=1)
            if cat_block
            else np.zeros((n, 0), dtype=np.float32)
        )

        X = (
            np.concatenate([X_num, X_cat], axis=1)
            if X_num.size + X_cat.size > 0
            else np.zeros((n, 0), dtype=np.float32)
        )
        return X

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the fitted preprocessor to a JSON-compatible dict."""
        return {
            "num_features": self.num_features,
            "cat_features": self.cat_features,
            "num_stats": self.num_stats,
            "cat_vocab": self.cat_vocab,
            "z_clip": self.z_clip,
            "numeric_transforms": self.numeric_transforms,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]):
        """Reconstruct a fitted preprocessor from a dict produced by to_dict()."""
        p = Preprocessor(
            d["num_features"],
            d["cat_features"],
            z_clip=float(d.get("z_clip", 5.0)),
            numeric_transforms=d.get("numeric_transforms", {}),
        )
        p.num_stats = {
            k: (float(v[0]), float(v[1])) for k, v in d.get("num_stats", {}).items()
        }
        p.cat_vocab = {k: list(v) for k, v in d.get("cat_vocab", {}).items()}
        p.cat_index = {
            f: {v: i for i, v in enumerate(p.cat_vocab[f])} for f in p.cat_vocab
        }
        return p


def contexts_to_feature_dict(
    contexts: List[Dict[str, Any]], num_features: List[str], cat_features: List[str]
) -> Dict[str, np.ndarray]:
    """Legacy helper returning per-feature arrays (not used by NumPy model)."""
    feat = {}
    for f in num_features:
        feat[f] = np.array(
            [float(c.get(f, 0.0)) for c in contexts], dtype=np.float32
        ).reshape(-1, 1)
    for f in cat_features:
        feat[f] = np.array([str(c.get(f, "")) for c in contexts], dtype=object).reshape(
            -1, 1
        )
    return feat


def build_xyw(
    examples: List[Dict[str, Any]],
    strategy_map: Dict[str, int],
    preproc: "Preprocessor",
    ref_latest_date: dt.date,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct design matrix X, target matrix Y, and sample weights W.

    - X: transformed contexts via preprocessor.
    - Y: per-strategy rewards, missing entries set to 0 but masked by W.
    - W: element-wise weights = mask(M) * recency_weight(row), shape matches Y.

    Args:
        examples: Dataset slice.
        strategy_map: Mapping strategy id -> output index.
        preproc: Fitted Preprocessor.
        ref_latest_date: Date used to compute recency ages.
        alpha: Recency weighting strength (0 disables).

    Returns:
        (X, Y, W): arrays suitable for training.
    """
    contexts = [ex.get("context", {}) for ex in examples]
    X = preproc.transform(contexts)
    S = len(strategy_map)
    Y = np.full((len(examples), S), np.nan, dtype=np.float32)
    M = np.zeros((len(examples), S), dtype=np.float32)
    for i, ex in enumerate(examples):
        rewards = ex.get("rewards", {})
        for k, v in rewards.items():
            if k in strategy_map:
                j = strategy_map[k]
                Y[i, j] = float(v)
                M[i, j] = 1.0

    dates = []
    for c in contexts:
        d = c.get("date")
        dates.append(parse_date(d) if d else dt.date(1970, 1, 1))
    days_from_latest = np.array(
        [(ref_latest_date - d).days for d in dates], dtype=np.float32
    )
    if alpha > 0:
        days_from_latest = days_from_latest - days_from_latest.min()
        days_from_latest = days_from_latest / (days_from_latest.max() + 1e-8)
        w = np.exp(-alpha * days_from_latest).astype(np.float32)
    else:
        w = np.ones_like(days_from_latest, dtype=np.float32)
    sample_weight = (M * w.reshape(-1, 1)).astype(np.float32)
    Y = np.nan_to_num(Y, nan=0.0)
    return X, Y, sample_weight


def latest_date(examples: List[Dict[str, Any]]) -> dt.date:
    """Return the maximum available date across examples (or epoch if none)."""
    dates = []
    for ex in examples:
        d = ex.get("context", {}).get("date")
        dates.append(parse_date(d) if d else dt.date(1970, 1, 1))
    return max(dates) if dates else dt.date(1970, 1, 1)


def restrict_to_last_n_days(
    examples: List[Dict[str, Any]], n_days: int
) -> List[Dict[str, Any]]:
    """Filter examples to those within the last N days from the dataset max date."""
    if n_days <= 0:
        return examples
    max_d = latest_date(examples)
    cutoff = max_d - dt.timedelta(days=n_days)
    out = []
    for ex in examples:
        d = ex.get("context", {}).get("date")
        dd = parse_date(d) if d else dt.date(1970, 1, 1)
        if dd >= cutoff:
            out.append(ex)
    return out
