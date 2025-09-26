FX Strategy Recommender (NumPy)

Minimal, dependency-light neural network (NumPy MLP) that learns per‑strategy rewards with masked regression, time‑aware splits, recency weights, and MC Dropout for uncertainty. Includes a CLI, a Streamlit app, and a small FastAPI service.

Quickstart
- Create venv and install:
  - `uv venv && source .venv/bin/activate`
  - `uv sync`
- Train: `uv run python main.py train --data fx_trading_dataset.json --model_dir models/latest`
- Predict: `uv run python main.py predict --attributes '{"currency_pair":"EUR/USD","volatility":0.1,"size":250,"time_of_day":"open","date":"2023-05-10"}'`
- Evaluate: `uv run python main.py eval --data fx_trading_dataset.json --model_dir models/latest`

Streamlit App
- Install UI deps: `uv sync --group app`
- Run: `uv run streamlit run streamlit_app.py`
- Features:
  - Predict: Enter trade attributes (JSON), adjust MC Dropout passes, see top candidates.
  - Train: Use bundled dataset or upload a JSON file; tweak hyperparameters; view summary.
  - Evaluate: Run evaluation and view metrics.

HTTP API
- Install server deps: `uv sync --group serve`
- Run: `uv run fxrec-api` (starts FastAPI via uvicorn on port 8000)
- Endpoints:
  - `GET /health`
  - `POST /predict` with body `{ "attributes": {...}, "mc_passes": 30, "model_dir": "models/latest" }`

Files
- `main.py` – CLI for training, prediction, evaluation.
- `streamlit_app.py` – Streamlit frontend.
- `src/data.py` – Dataset loading, feature typing, preprocessing, time splits.
- `src/modeling.py` – NumPy MLP (ReLU + dropout, Adam, MC dropout, save/load).
- `src/train_pipeline.py` – Training orchestration and artifact saving.
- `src/predict.py` – Model/preprocessor loading and single‑context prediction.
- `src/evaluate.py` – Test‑split evaluation and report.
- `src/api.py` – FastAPI service for predictions.

Notes
- Expects dataset items shaped like: `{ "context": {...}, "rewards": {"strategy_id": reward, ...} }`.
- Artifacts saved under `models/<run>/` (default `models/latest/`).
