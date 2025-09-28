# Repository Guidelines

This document is the source of truth for how agents (and humans) should work in this repo. Its scope applies to the entire repository unless otherwise stated.

## Project Structure & Module Organization
- `main.py` – CLI with `train`, `predict`, and `eval` subcommands.
- `streamlit_app.py` – Streamlit UI for predict/train/eval.
- `src/data.py` – Dataset loading, feature typing, preprocessing, time splits.
- `src/modeling.py` – NumPy MLP (ReLU + dropout, Adam, MC dropout, save/load).
- `src/train_pipeline.py` – Training orchestration and artifact saving.
- `src/predict.py` – Model/preprocessor loading and single‑context prediction.
- `src/evaluate.py` – Test‑split evaluation and report.
- `src/api.py` – FastAPI service exposing `/predict`.
- `fx_trading_dataset.json` – Input dataset (context + per‑strategy rewards).
- `models/` – Saved artifacts (do not rely on these for reviews; regenerate).

### Current State
- Model: pure NumPy MLP with masked MSE, MC Dropout, Adam, early stopping. Robust save/load.
- Data: time‑based splits; recency weighting (`alpha`); optional rolling window (`window_days`).
- CLI: training, prediction, and evaluation fully functional.
- UI: Streamlit app (3 tabs) for predict/train/eval, with tooltips and metric explanations.
- API: FastAPI service with `/predict`, simple cache and reload endpoint.
- Packaging: `pyproject.toml` with `[project]`, dev/app/serve dependency groups, scripts `fxrec` and `fxrec-api`.
- Docs: module/class/function docstrings in `src/modeling.py` and `src/evaluate.py`.

## Build, Test, and Development Commands
- Create venv (uv): `uv venv && source .venv/bin/activate`
- Install runtime deps: `uv sync`
- Also install dev tools: `uv sync --all-groups`
- Train: `uv run python main.py train --data fx_trading_dataset.json --epochs 30 --batch_size 512 --alpha 6.0 --val_split 0.05 --test_split 0.05 --hidden 128 64 --dropout 0.2 --model_dir models/latest`
- Predict: `uv run python main.py predict --attributes '{"currency_pair":"GBP/USD","volatility":0.1,"size":300,"time_of_day":"open","date":"2023-05-10"}'`
- Evaluate: `uv run python main.py eval --data fx_trading_dataset.json --model_dir models/latest`

Notes on console scripts
- You can run `uv run fxrec ...` and `uv run fxrec-api` if your uv version recognizes local project scripts.
- If not, install the project in editable mode first: `uv pip install -e .` then run `uv run fxrec ...`.

### Frontend & API
- Streamlit app: `uv sync --group app && uv run streamlit run streamlit_app.py`
- FastAPI server: `uv sync --group serve && uv run fxrec-api`

API details
- Default model dir via `FXREC_MODEL_DIR` env var. Helper runs on `FXREC_HOST`/`FXREC_PORT` (default `0.0.0.0:8000`).
- `POST /predict` body: `{ "attributes": {...}, "mc_passes": 30, "model_dir": "models/latest" }`.
- `POST /reload?model_dir=...` clears cache and optionally switches default model.

## Coding Style & Naming Conventions
- Python 3.13; PEP 8.
- Formatting: Black (line length 88). Linting: Ruff. See `pyproject.toml`.
- Names: modules `snake_case.py`; functions `snake_case`; classes `CapWords`.
- Add concise docstrings for public functions/classes; keep parameters typed.

Type checking and hooks
- Mypy configured in `pyproject.toml`. Run with `uv run mypy .` (types are partial; don’t force strict everywhere).
- Pre-commit (optional): `uv pip install pre-commit && pre-commit install`.

## Testing Guidelines
- No formal unit tests yet. Use `eval` as a smoke test after training.
- If adding tests, prefer `pytest` under `tests/` (e.g., `tests/test_predict.py`).
  - Name tests `test_*.py`; keep fixtures small and deterministic.
  - Avoid checking in large artifacts; synthesize minimal inputs in tests.

## Commit & Pull Request Guidelines
- Commits: small, focused; present‑tense imperative subject (e.g., "Add eval subcommand").
- PRs: include description, rationale, and CLI examples; link issues if applicable.
- Update docs (`README.md`, docstrings) when changing CLI, data schema, or outputs.
- Do not commit generated artifacts (`models/*`) unless required for a demo.

## Security & Configuration Tips
- Dataset may contain sensitive trading context; keep files local.
- Non‑stationarity handling is via time‑based splits, recency weighting (`--alpha`), and optional `--window_days`. Keep these flags visible in examples.

## Data & Modeling Details
- Dataset format: a list of items each with `context: {..}` and `rewards: {strategy_id: reward}`; `context.date` (YYYY-MM-DD) is used for ordering/weights and excluded from features by default.
- Preprocessing: standardize numeric features; one‑hot encode categoricals.
  - Numeric transforms: `size` uses `log1p` by default; standardized numerics are clipped to ±5σ to reduce OOD blow‑ups.
  - Unseen categorical values map to an all‑zeros vector (no learned fallback bucket yet). Consider adding an `__UNK__` bucket if you touch this area.
- Targets: multi‑output rewards (one per strategy). Missing rewards are masked via element‑wise weights.
- Loss: masked MSE; training uses recency weights row‑wise (controlled by `alpha`).
- Uncertainty: MC Dropout; confidence reported as `1/(1+uncertainty)` (heuristic).
- Evaluation metrics: masked MSE; coverage; top‑1 accuracy (covered only); mean/p50/p90 regret; avg true best reward; avg realized reward of predicted strategy.

## Agent‑Specific Instructions
- Keep dependencies minimal (NumPy only); avoid adding frameworks unless requested.
- Preserve CLI contracts and file structure; prefer incremental, surgical changes.
- When touching code, ensure docstrings remain accurate and run an `eval` sanity check.
- UI changes (Streamlit): give widgets explicit keys to avoid state collisions; keep defaults sensible and add `help` tooltips where useful.
- API changes: maintain backward compatibility for `/predict` request/response; keep model caching and `/reload` endpoint behavior.
- If you add exploration or online updates, gate them behind flags and persist any new state in a dedicated directory (e.g., `data/online/`).

## Roadmap / Nice‑to‑Haves
- Unknown category handling (`__UNK__`) with predict‑time warnings.
- Exploration policies (MC‑Dropout UCB / Thompson) with propensities and OPE metrics in `eval`.
- Online fine‑tuning with a replay buffer and drift triggers.
- GitHub Actions CI for lint/type/smoke‑train; optional Dockerfile.
