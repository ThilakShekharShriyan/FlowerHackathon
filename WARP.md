# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- Purpose: Run Flower federated learning experiments on the SOCOFing dataset, track runs/rounds/metrics in MongoDB, and drive iterations via a Google Gemini-powered terminal agent (`google-genai`).
- Main code lives under aicontroller/ (Python package), with Flower ServerApp/ClientApp and a Typer-based terminal agent.

Common commands
- Setup (from repo root)
  - cd aicontroller
  - python -m venv env && source env/bin/activate
  - python -m pip install -U pip && pip install -e .

- Health checks (optional)
  - flwr --version
  - python -c "import pymongo; print('pymongo ok')"
  - python -c "from google import genai; print('google-genai ok')"

- Run the agent REPL (recommended entrypoint)
  - cd aicontroller && source env/bin/activate
  - export GOOGLE_API_KEY=<your_key>
  - python -m aicontroller.llm_agent_cli --model "${AIC_GOOGLE_MODEL:-gemini-2.0-flash}"
  - At the agent> prompt, useful local commands (no LLM roundtrip):
    - status
    - list        (or: list runs)
    - run 1 rounds with FedAvg (lr 0.01, local-epochs 1, fraction-train 0.6)
    - show rounds <RUN_OBJECT_ID>
    - summarize latest
    - suggest next
    - compare strategies FedAvg, FedAdam (limit 10)

- Run Flower directly (bypass the agent)
  - cd aicontroller && source env/bin/activate
  - flwr run . --run-config 'num-server-rounds=3 local-epochs=1 fraction-train=0.5 lr=0.01 strategy="FedAvg" data-root="/ABS/PATH/TO/data" label-mode="binary" num-partitions=5'
  - Tip: add --stream to stream logs in-place.

- Federation (optional)
  - Use env var to target a federation when launching runs: export AIC_FEDERATION=local (or FLWR_FEDERATION)
  - From within the REPL you can also set: use federation local

- Data path
  - Default data root comes from pyproject.toml (tool.flwr.app.config.data-root). Ensure SOCOFing exists under that path (see README for expected layout), or override via AIC_DATA_ROOT.

- Logs and locks
  - Agent logs: ~/.cache/aicontroller/llm_agent.log (JSONL events also written nearby)
  - Run lock: aicontroller/.flwr_run.lock (use unlock in REPL if a prior run crashed)

- Linting / tests
  - No linting or test framework is configured in this repo at present.

Key environment variables
- MONGODB_URI (default: mongodb://localhost:27017)
- MONGODB_DB (default: flwr_runs)
- GOOGLE_API_KEY (required)
- AIC_GOOGLE_MODEL (default: gemini-2.0-flash)
- AIC_DATA_ROOT (default: aicontroller/data, or the value set in pyproject)
- AIC_LABEL_MODE (binary|fourclass, default: binary)
- AIC_NUM_PARTITIONS (default: 5)
- AIC_AGENT_LOG_LEVEL (default: INFO)
- AIC_AGENT_MONGO_LOCK_ENABLE (default: true)
- AIC_FEDERATION or FLWR_FEDERATION (optional; when present the agent adds the federation name to flwr run)

Architecture (big picture)
- Flower integration is defined in aicontroller/pyproject.toml
  - tool.flwr.app.components points to:
    - serverapp: aicontroller.server_app:app
    - clientapp: aicontroller.client_app:app
  - tool.flwr.app.config provides default run-config (num rounds, lr, fraction, data-root, label-mode, num-partitions, strategy).

- Server side: aicontroller/server_app.py
  - Builds a base Flower strategy (FedAvg/FedAdam/FedYogi/FedAdagrad) from run_config.strategy and wraps it in TrackingStrategy.
  - Handles ndarray/torch scalar compatibility for some strategies.
  - Starts training with global initial model (task.Net), saves final_model.pt at the end.

- Strategy wrapper: aicontroller/tracking_strategy.py
  - Normalizes aggregate_train/aggregate_evaluate signatures across Flower versions (tries official order first, then falls back).
  - Persists run lifecycle and per-round logs into Mongo: runs and rounds collections.
  - Accepts both dict and tuple-shaped metrics; stores client metrics best-effort.

- Client side: aicontroller/client_app.py
  - Implements train/evaluate using the SOCOFing task in aicontroller/task.py.
  - Reads run_config in Context (data-root, label-mode, num-partitions, local-epochs), chooses partition by node_id/partition-id.

- Task: aicontroller/task.py
  - Dataset scanner/loader for SOCOFing (binary vs fourclass modes), simple CNN (grayscale 96×96), IID partitioning, 80/20 split.
  - Exposes train() and test() used by the ClientApp.

- Agent REPL: aicontroller/llm_agent_cli.py
  - Typer CLI that opens an interactive prompt.
  - Tools exposed: run_flower, list_runs, show_rounds, summarize_run, suggest_next, compare_strategies.
  - Uses Google ADK (google-genai) via `aicontroller/adk_agent.py` for model I/O; shared tool logic in `aicontroller/agent_tools.py`; Rich UI in `aicontroller/ui_render.py`.
  - Local shortcuts avoid any LLM call for speed; otherwise, calls Google Gemini (default `gemini-2.0-flash`) and routes tool calls based on model output.
  - Run lock file prevents overlapping runs; optional Mongo-based active-run check.
  - Supports federation suffix (env or "use federation <name>") and passes --stream to flwr runs.

- Optional service: aicontroller/agent_runner.py
  - FastAPI microservice to trigger runs and query Mongo: POST /run, GET /runs, GET /runs/{id}/rounds, GET /runs/{id}/summary, GET /compare.
  - Uses a simple on-disk lock to avoid overlapping runs.

What matters when editing
- If you change run defaults or add new hyperparameters, keep tool.flwr.app.config in aicontroller/pyproject.toml in sync.
- If you add a new strategy, extend _make_strategy in server_app.py and ensure TrackingStrategy still captures metrics correctly.
- If you modify dataset layout, update task.py scanning paths and README notes accordingly.
- The agent assumes flwr CLI is on PATH and that pyproject.toml is discoverable from the CWD; prefer running commands from aicontroller/.
