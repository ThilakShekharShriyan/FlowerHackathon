# aicontroller/server_app.py
"""Flower ServerApp with strategy selection, scalar-compat shim, and Mongo tracking."""
from __future__ import annotations

import os
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdam, FedYogi, FedAdagrad

from aicontroller.task import Net
from aicontroller.tracking_strategy import TrackingStrategy

# --- Compat shim for Adam/Yogi/Adagrad expecting ndarray, not scalars/torch ---
try:
    from flwr.common.record.array import Array as _Array
    import flwr.serverapp.strategy.fedadam as _fedadam
    import flwr.serverapp.strategy.fedadagrad as _fedadagrad
    import flwr.serverapp.strategy.fedyogi as _fedyogi

    class _CompatArray(_Array):  # type: ignore[misc]
        def __init__(self, x=None, **kwargs):
            if x is not None:
                if hasattr(x, "detach"):  # torch.Tensor
                    x = x.detach().cpu().numpy()
                if np.isscalar(x):
                    x = np.asarray(x)
            super().__init__(x, **kwargs)

    _fedadam.Array = _CompatArray
    _fedadagrad.Array = _CompatArray
    _fedyogi.Array   = _CompatArray
except Exception:
    pass


def _make_strategy(name: str, fraction_train: float):
    n = (name or "FedAvg").strip()
    if n == "FedAvg":
        return FedAvg(fraction_train=fraction_train)
    if n == "FedAdam":
        return FedAdam(fraction_train=fraction_train)
    if n == "FedYogi":
        return FedYogi(fraction_train=fraction_train)
    if n == "FedAdagrad":
        return FedAdagrad(fraction_train=fraction_train)
    print(f"[ServerApp] Unknown strategy '{name}', falling back to FedAvg")
    return FedAvg(fraction_train=fraction_train)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = dict(context.run_config)
    fraction_train = float(rc["fraction-train"])
    num_rounds     = int(rc["num-server-rounds"])
    lr             = float(rc["lr"])
    strategy_name  = str(rc.get("strategy", "FedAvg")).strip().strip("'").strip('"')

    print(
        f"[ServerApp] Starting with strategy={strategy_name}, "
        f"rounds={num_rounds}, lr={lr}, fraction-train={fraction_train}"
    )

    base = _make_strategy(strategy_name, fraction_train=fraction_train)

    strategy = TrackingStrategy(
        base_strategy=base,
        mongo_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        mongo_db=os.getenv("MONGODB_DB", "flwr_runs"),
        run_meta=rc,
        app_name=os.getenv("AIC_APP_NAME", "aicontroller"),
    )

    # Global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Train
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model
    print("\n[ServerApp] Saving final model to disk (final_model.pt)...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    print("[ServerApp] Done.")
