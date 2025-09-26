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


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.markdown("Use the tabs below to predict, train, and evaluate.")

    tabs = st.tabs(["Predict", "Train", "Evaluate"])
    with tabs[0]:
        page_predict()
    with tabs[1]:
        page_train()
    with tabs[2]:
        page_eval()


if __name__ == "__main__":
    main()
