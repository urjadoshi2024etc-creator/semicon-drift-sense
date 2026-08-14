"""Drift-Sense V5 training script.

The existing generator and dram_dataset.py are intentionally untouched.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dram_dataset import DramPairDataset
from model_v6 import build_model
from loss_v5 import V5LocalizationLoss
from utils_v5 import compute_metrics, load_checkpoint, save_checkpoint, set_seed, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--lr_patience", type=int, default=40)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--early_stop_patience", type=int, default=15,
                    help="Stop if val mean_error doesn't improve by min_delta for this many epochs.")
    p.add_argument("--min_delta", type=float, default=0.5,
                    help="Minimum val mean_error (px) improvement to reset patience counter.")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False, grad_clip=5.0):
    training = optimizer is not None
    model.train(training)
    total_loss = cls_loss = off_loss = margin_loss = 0.0
    n = 0
    preds, targets = [], []
    peak_ratios = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, leave=False)
        for ref, search, gt, _pid in pbar:
            ref = ref.to(device, non_blocking=True)
            search = search.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            bs = ref.shape[0]
            if training:
                optimizer.zero_grad(set_to_none=True)

            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(ref, search)
                    loss, parts = criterion(out, gt)
            else:
                out = model(ref, search)
                loss, parts = criterion(out, gt)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += float(loss.detach()) * bs
            cls_loss += float(parts["loss_classification"]) * bs
            off_loss += float(parts["loss_offset"]) * bs
            margin_loss += float(parts["loss_margin"]) * bs
            n += bs
            preds.append(out.coords.detach().float().cpu().numpy())
            targets.append(gt.detach().float().cpu().numpy())
            peak_ratios.append(float(out.peak_to_uniform.mean().detach().cpu()))
            pbar.set_postfix(loss=f"{total_loss/n:.3f}", err=f"{np.linalg.norm(preds[-1]-targets[-1],axis=1).mean():.1f}")

    pred = np.concatenate(preds, axis=0)
    targ = np.concatenate(targets, axis=0)
    metrics = compute_metrics(pred, targ)
    metrics.update({
        "loss": total_loss / n,
        "loss_cls": cls_loss / n,
        "loss_offset": off_loss / n,
        "loss_margin": margin_loss / n,
        "peak_uniform": float(np.mean(peak_ratios)),
    })
    return metrics, pred, targ


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("V6: valid correlation + scale-aware separate encoders + "
          "context-aware candidate CE (3x3x2 score head) + local offsets")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = DramPairDataset(args.train_dir, normalize_labels=False)
    val_ds = DramPairDataset(args.val_dir, normalize_labels=False)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False,
                            pin_memory=device.type == "cuda")

    model = build_model().to(device)
    criterion = V5LocalizationLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda")) if device.type == "cuda" else None

    history = []
    best = float("inf")
    start = 1
    epochs_since_improve = 0
    if args.resume:
        start, best, old_hist, _ = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start += 1
        history = old_hist if isinstance(old_hist, list) else []
        print(f"Resumed checkpoint. Starting epoch {start}, best={best:.3f}px")

    config = vars(args).copy()
    save_json(os.path.join(args.output_dir, "v6_config.json"), config)

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        tr, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, args.amp, args.grad_clip)
        va, pred, targ = run_epoch(model, val_loader, criterion, device, None, None, False, args.grad_clip)
        scheduler.step(va["mean_error"])

        row = {"epoch": epoch, **{f"train_{k}": v for k,v in tr.items()}, **{f"val_{k}": v for k,v in va.items()}, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists: w.writeheader()
            w.writerow(row)

        improved = va["mean_error"] < (best - args.min_delta)
        if improved:
            best = va["mean_error"]
            epochs_since_improve = 0
            save_checkpoint(os.path.join(ckpt_dir, "best_model.pt"), model, optimizer, scheduler, epoch, best, history, config)
        else:
            epochs_since_improve += 1
        save_checkpoint(os.path.join(ckpt_dir, "last_checkpoint.pt"), model, optimizer, scheduler, epoch, best, history, config)

        print(f"[epoch {epoch}] train={tr['mean_error']:.2f}px val={va['mean_error']:.2f}px "
              f"median={va['median_error']:.2f}px acc@5={va['acc@5px']:.1f}% "
              f"acc@10={va['acc@10px']:.1f}% cls={va['loss_cls']:.3f} off={va['loss_offset']:.3f} "
              f"margin={va['loss_margin']:.3f} peak/uniform={va['peak_uniform']:.1f}x "
              f"lr={optimizer.param_groups[0]['lr']:.2e} ({time.time()-t0:.1f}s)")

        print(f"  (best val_mean_error={best:.2f}px, {epochs_since_improve}/{args.early_stop_patience} "
              f"epochs since last improvement)")

        if args.overfit and best < 10.0:
            print(f"\nOVERFIT GATE PASSED at epoch {epoch}: best val mean error={best:.2f}px < 10px")
            print("You may proceed to full-dataset training.")
            break

        if epochs_since_improve >= args.early_stop_patience:
            print(f"\nEARLY STOPPING at epoch {epoch}: no val improvement >= {args.min_delta}px "
                  f"for {args.early_stop_patience} epochs. Best val_mean_error={best:.2f}px.")
            print(f"Best checkpoint saved at: {os.path.join(ckpt_dir, 'best_model.pt')}")
            break

    print("\nFINAL")
    print(f"Best validation mean error: {best:.3f}px")
    if args.overfit and best >= 10.0:
        print("OVERFIT GATE NOT PASSED. Do NOT train the full dataset yet.")


if __name__ == "__main__":
    main()

