"""Utilities for Drift-Sense V5 training, metrics and checkpoints."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    err = np.linalg.norm(pred - gt, axis=1)
    return {
        "mean_error": float(err.mean()),
        "median_error": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "acc@2px": float((err <= 2).mean() * 100.0),
        "acc@5px": float((err <= 5).mean() * 100.0),
        "acc@10px": float((err <= 10).mean() * 100.0),
        "acc@25px": float((err <= 25).mean() * 100.0),
        "max_error": float(err.max()),
    }


def save_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    epoch: int = 0, best_metric: float = float("inf"),
                    history: Optional[Dict[str, list]] = None, config: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "history": history or {},
        "config": config or {},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_metric", float("inf"))), ckpt.get("history", {}), ckpt.get("config", {})


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
