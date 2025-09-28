"""Streamlit UI for the FX strategy recommender.

This app provides three main sections:
- Predict: enter trade attributes and get a recommended strategy.
- Train: train a new model on the bundled dataset or a user-uploaded JSON.
- Evaluate: run evaluation on a dataset and view summary metrics.

Run locally:
  uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src.evaluate import evaluate_and_report
from src.predict import load_artifacts, predict_context
from src.train_pipeline import train_and_save


APP_TITLE = "FX Strategy Recommender"
DEFAULT_MODEL_DIR = "models/latest"
DEFAULT_DATA_PATH = "fx_trading_dataset.json"


def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner=False)
def _cached_load_artifacts(model_dir: str):
    return load_artifacts(model_dir)


def _write_uploaded_json(file_bytes: bytes, target_dir: Path) -> Path:
    _ensure_dir(target_dir)
    # Validate JSON before writing
    try:
        _ = json.loads(file_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Uploaded file is not valid JSON: {e}") from e
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = target_dir / f"uploaded_{ts}.json"
    out_path.write_bytes(file_bytes)
    return out_path


def _parse_hidden_layers(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return [128, 64]
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except Exception as e:
        raise ValueError("Hidden layers must be a comma-separated list of ints, e.g. '128,64'.") from e


def _default_context_example() -> Dict[str, Any]:
    return {
        "currency_pair": "EUR/USD",
        "volatility": 0.12,
        "size": 250,
        "time_of_day": "open",
        "date": "2023-05-10",
    }


def page_predict():
    st.header("Predict")
    st.caption("Enter trade attributes and get a recommended strategy.")

    model_dir = st.text_input(
        "Model directory",
        value=st.session_state.get("model_dir", DEFAULT_MODEL_DIR),
        key="predict_model_dir",
    )
    with st.expander("Attributes (JSON)", expanded=True):
        default_json = json.dumps(_default_context_example(), indent=2)
        attr_text = st.text_area(
            "Trade attributes JSON", value=default_json, height=180, key="predict_attrs"
        )

    with st.expander("Advanced settings"):
        mc_passes = st.number_input(
            "MC Dropout passes",
            min_value=1,
            max_value=500,
            value=30,
            step=1,
            key="predict_mc_passes",
        )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        run = st.button("Predict", type="primary", key="btn_predict")
    with col_b:
        reload = st.button("Reload model cache", key="btn_reload_cache")

    if reload:
        _cached_load_artifacts.clear()
        st.info("Model cache cleared.")

    if run:
        try:
            context = json.loads(attr_text)
            artifacts, meta = _cached_load_artifacts(model_dir)
            with st.spinner("Running prediction..."):
                result = predict_context(artifacts, meta, context, mc_passes=int(mc_passes))
            st.success("Prediction complete")
            st.subheader("Result")
            st.write({k: v for k, v in result.items() if k != "top"})
            st.subheader("Top candidates")
            st.dataframe(result.get("top", []), use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.session_state["model_dir"] = model_dir


def page_train():
    st.header("Train")
    st.caption("Train a model on the bundled dataset or an uploaded JSON file.")

    data_source = st.radio(
        "Dataset source", ["Project dataset", "Upload JSON"], horizontal=True, key="train_data_source"
    )
    uploaded = None
    if data_source == "Upload JSON":
        uploaded = st.file_uploader("Upload a JSON file", type=["json"], key="train_uploader")

    model_dir = st.text_input(
        "Output model directory",
        value=st.session_state.get("model_dir", DEFAULT_MODEL_DIR),
        key="train_model_dir",
        help="Folder to save artifacts (weights, preprocessor, metadata).",
    )

    st.markdown("### Hyperparameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=1000,
            value=20,
            key="train_epochs",
            help="Max training epochs (early stopping may stop sooner).",
        )
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=10000,
            value=256,
            key="train_batch_size",
            help="Mini-batch size for Adam updates (larger = fewer steps per epoch).",
        )
        dropout = st.number_input(
            "Dropout",
            min_value=0.0,
            max_value=0.9,
            value=0.2,
            step=0.05,
            key="train_dropout",
            help="Dropout rate on hidden layers (regularization; 0 disables).",
        )
    with col2:
        alpha = st.number_input(
            "Recency weighting alpha",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.5,
            key="train_alpha",
            help="Strength of recency weighting for samples (higher = more recent emphasis; 0 disables).",
        )
        val_split = st.number_input(
            "Val split (0-1)",
            min_value=0.0,
            max_value=0.9,
            value=0.15,
            step=0.01,
            key="train_val_split",
            help="Fraction of latest data reserved for validation (time-ordered).",
        )
        test_split = st.number_input(
            "Test split (0-1)",
            min_value=0.0,
            max_value=0.9,
            value=0.15,
            step=0.01,
            key="train_test_split",
            help="Fraction of latest data reserved for test (time-ordered).",
        )
    with col3:
        hidden_text = st.text_input(
            "Hidden layers (comma-separated)",
            value="128,64",
            key="train_hidden",
            help="Hidden layer sizes, e.g., '128,64' for two layers of 128 and 64 units.",
        )
        window_days = st.number_input(
            "Window days (0=all)",
            min_value=0,
            max_value=3650,
            value=0,
            key="train_window",
            help="Restrict training to the last N days (0 uses the full dataset).",
        )

    run = st.button("Start training", type="primary", key="btn_train")

    if run:
        try:
            if data_source == "Project dataset":
                data_path = DEFAULT_DATA_PATH
            else:
                if not uploaded:
                    st.error("Please upload a JSON dataset.")
                    return
                file_bytes = uploaded.read()
                data_path = _write_uploaded_json(file_bytes, Path("data/uploads"))

            hidden = _parse_hidden_layers(hidden_text)

            with st.spinner("Training model..."):
                train_and_save(
                    data_path=str(data_path),
                    model_dir=model_dir,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    alpha=float(alpha),
                    val_split=float(val_split),
                    test_split=float(test_split),
                    hidden=hidden,
                    dropout=float(dropout),
                    window_days=int(window_days),
                )

            # Read training metadata for summary
            meta_path = Path(model_dir) / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                st.success("Training complete")
                st.subheader("Summary")
                st.json(meta)
            else:
                st.warning("Training finished, but no metadata.json was found.")

            st.session_state["model_dir"] = model_dir
            _cached_load_artifacts.clear()
        except Exception as e:
            st.error(f"Training failed: {e}")


def page_eval():
    st.header("Evaluate")
    st.caption("Run evaluation on a dataset and view summary metrics.")

    data_source = st.radio(
        "Dataset source", ["Project dataset", "Upload JSON"], horizontal=True, key="eval_data_source"
    )
    uploaded = None
    if data_source == "Upload JSON":
        uploaded = st.file_uploader(
            "Upload a JSON file", type=["json"], key="eval_uploader"
        )

    model_dir = st.text_input("Model directory", value=st.session_state.get("model_dir", DEFAULT_MODEL_DIR), key="eval_model_dir")
    col1, col2 = st.columns(2)
    with col1:
        val_split = st.number_input("Val split (0-1)", min_value=0.0, max_value=0.9, value=0.15, step=0.01, key="eval_val")
    with col2:
        test_split = st.number_input("Test split (0-1)", min_value=0.0, max_value=0.9, value=0.15, step=0.01, key="eval_test")

    run = st.button("Run evaluation", type="primary", key="btn_eval")

    if run:
        try:
            if data_source == "Project dataset":
                data_path = DEFAULT_DATA_PATH
            else:
                if not uploaded:
                    st.error("Please upload a JSON dataset.")
                    return
                file_bytes = uploaded.read()
                data_path = _write_uploaded_json(file_bytes, Path("data/uploads"))

            with st.spinner("Evaluating model..."):
                report = evaluate_and_report(
                    data_path=str(data_path),
                    model_dir=model_dir,
                    val_split=float(val_split),
                    test_split=float(test_split),
                )
            st.success("Evaluation complete")
            st.subheader("Report")
            st.json(report)

            # Quick explanations for metrics
            metric_help = {
                "samples_test": "Number of test examples evaluated.",
                "masked_mse": "Masked mean squared error between predicted and true rewards over observed targets. Lower is better.",
                "coverage": "Fraction of test examples where the predicted top strategy had an observed reward (was in the example's labels). Higher is better.",
                "top1_accuracy": "Accuracy among covered examples where the predicted top strategy equals the true best observed strategy. Higher is better.",
                "mean_regret": "Average (true best reward − reward of predicted strategy) over covered examples. Lower is better.",
                "p50_regret": "Median regret across covered examples. Lower is better.",
                "p90_regret": "90th percentile of regret (tail risk). Lower is better.",
                "avg_true_best_reward": "Average of the best observed rewards; context-dependent upper baseline.",
                "avg_predicted_strategy_actual_reward": "Average realized reward of the chosen strategy on covered examples; compare to avg_true_best_reward.",
            }
            with st.expander("What do these metrics mean?", expanded=False):
                lines = []
                for k, desc in metric_help.items():
                    val = report.get(k, None)
                    if isinstance(val, float):
                        try:
                            val_str = f"{val:.6f}"
                        except Exception:
                            val_str = str(val)
                    else:
                        val_str = str(val)
                    lines.append(f"- `{k}` = {val_str} — {desc}")
                st.markdown("\n".join(lines))
        except Exception as e:
            st.error(f"Evaluation failed: {e}")


def _mlp_svg(input_dim: int, hidden: List[int], output_dim: int, dropout: float) -> str:
    """Generate a simple inline SVG showing the MLP architecture."""
    layers = [f"Input\nD={input_dim}"] + [f"Hidden {i+1}\n{h} units\nReLU + Dropout {dropout:.2f}" for i, h in enumerate(hidden)] + [f"Output\nS={output_dim}"]
    box_w, box_h = 180, 90
    gap = 40
    n = len(layers)
    svg_w = n * box_w + (n - 1) * gap + 40
    svg_h = 180
    start_x = 20
    y = (svg_h - box_h) // 2

    def rect(x, label):
        lines = label.split("\n")
        t = "".join(
            f"<tspan x='{x + box_w/2}' dy='{14 if i==0 else 16}'>{line}</tspan>" for i, line in enumerate(lines)
        )
        return (
            f"<rect x='{x}' y='{y}' rx='8' ry='8' width='{box_w}' height='{box_h}' fill='#EEF2FF' stroke='#3B82F6' stroke-width='2'/>"
            f"<text x='{x + box_w/2}' y='{y + 30}' font-family='Inter, sans-serif' font-size='13' text-anchor='middle' fill='#1F2937'>{t}</text>"
        )

    def arrow(x1, x2):
        ay = y + box_h / 2
        return (
            f"<line x1='{x1}' y1='{ay}' x2='{x2}' y2='{ay}' stroke='#6B7280' stroke-width='2' marker-end='url(#arrow)'/>"
        )

    # Build SVG
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{svg_w}' height='{svg_h}' viewBox='0 0 {svg_w} {svg_h}'>",
        "<defs><marker id='arrow' viewBox='0 0 10 10' refX='10' refY='5' markerWidth='8' markerHeight='8' orient='auto-start-reverse'>",
        "<path d='M 0 0 L 10 5 L 0 10 z' fill='#6B7280'/></marker></defs>",
    ]
    xs = []
    x = start_x
    for label in layers:
        parts.append(rect(x, label))
        xs.append(x)
        x += box_w + gap
    for i in range(len(xs) - 1):
        parts.append(arrow(xs[i] + box_w, xs[i + 1]))
    parts.append("</svg>")
    return "".join(parts)


def page_model():
    st.header("Model Architecture")
    st.caption("Visual overview and notation of the MLP scoring pipeline.")

    model_dir = st.text_input(
        "Model directory",
        value=st.session_state.get("model_dir", DEFAULT_MODEL_DIR),
        key="model_model_dir",
        help="Directory containing model.json, weights.npz, preprocessor.json, metadata.json.",
    )
    show = st.button("Load model and show diagram", type="primary", key="btn_model_show")

    if show:
        try:
            (preproc, model), meta = _cached_load_artifacts(model_dir)
            hidden = list(model.cfg.hidden)
            dropout = float(model.cfg.dropout)
            input_dim = int(model.cfg.input_dim)
            output_dim = int(model.cfg.output_dim)

            st.subheader("Architecture")
            svg = _mlp_svg(input_dim=input_dim, hidden=hidden, output_dim=output_dim, dropout=dropout)
            components.html(svg, height=220)

            st.subheader("Notation and flow")
            st.markdown(
                """
                - Inputs: `X ∈ R^{N×D}` from standardized numerics + one‑hot categoricals.
                - Layer 1: `Z₁ = X W₁ + b₁`, `H₁ = ReLU(Z₁)`, `Ĥ₁ = H₁ ⊙ M₁` (dropout mask during training/MC).
                - Layer 2..k: repeat `Zᵢ = Ĥᵢ₋₁ Wᵢ + bᵢ`, `Hᵢ = ReLU(Zᵢ)`, apply dropout.
                - Output: `Ŷ = Ĥ_k W_out + b_out` with `S` strategy scores (higher = better).
                - Uncertainty: MC Dropout → run T stochastic passes: `μ = mean(Ŷ)`, `σ = std(Ŷ)`.
                - Loss (training): masked MSE with recency weights `w`: `L = Σ (w ⊙ (Ŷ − Y)²) / Σ w`.
                """
            )

            with st.expander("Feature breakdown"):
                num_feats = preproc.num_features
                cat_feats = preproc.cat_features
                cat_dims = {f: len(preproc.cat_vocab.get(f, [])) for f in cat_feats}
                st.write(
                    {
                        "input_dim": input_dim,
                        "numeric_features": num_feats,
                        "categorical_features": cat_feats,
                        "categorical_dims": cat_dims,
                        "output_strategies": meta.get("strategy_names", []),
                    }
                )
        except Exception as e:
            st.error(f"Failed to load model: {e}")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.markdown("Use the tabs below to predict, train, and evaluate.")

    tabs = st.tabs(["Predict", "Train", "Evaluate", "Model"])
    with tabs[0]:
        page_predict()
    with tabs[1]:
        page_train()
    with tabs[2]:
        page_eval()
    with tabs[3]:
        page_model()


if __name__ == "__main__":
    main()
