"""
network_i/train.py  —  Training entry point for Network I.

Usage:
    python -m network_i.train
    # or
    cd network_i && python train.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import (DEVICE, REAL_MAT_FILE, VIRTUAL_JSON_FILE,
                     NET1_EPOCHS, NET1_LR, NET1_WEIGHT_DECAY, NET1_GRAD_CLIP,
                     NET1_LR_MIN, NET1_PLOT_INTERVAL, NET1_CHECKPOINT,
                     REAL_DOMAIN_WEIGHT, VIRTUAL_DOMAIN_WEIGHT)
from network_i.model   import SoftRobotTrajectoryPlanner
from network_i.dataset import CotrainingDataset, load_real_trajectories, load_virtual_trajectories
from network_i.losses  import total_loss


def collate(batch): return batch[0]


def train():
    print("=" * 60)
    print("Network I — Sim & Real Co-Training (Trajectory Optimisation)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    real_trajs    = load_real_trajectories(REAL_MAT_FILE)
    virtual_trajs = load_virtual_trajectories(VIRTUAL_JSON_FILE)

    # Obstacle definition — matches the physical setup and Unity scene
    cubes = [
        ([0.352, 0.214, 1.762], 63.81, [2.01, 0.72, 0.49]),
    ]

    dataset = CotrainingDataset(real_trajs, virtual_trajs, cubes)
    loader  = DataLoader(dataset, batch_size=1, sampler=dataset.sampler,
                          collate_fn=collate, num_workers=0)

    model     = SoftRobotTrajectoryPlanner().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=NET1_LR, weight_decay=NET1_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NET1_EPOCHS, eta_min=NET1_LR_MIN)

    os.makedirs(os.path.dirname(NET1_CHECKPOINT), exist_ok=True)
    history = {k: [] for k in ("total", "position", "shape", "obstacle")}

    for epoch in range(NET1_EPOCHS):
        model.train()
        logs = []
        for batch in loader:
            graph    = batch["graph"].to(DEVICE)
            target   = batch["target"].to(DEVICE)
            cond     = batch["condition"].to(DEVICE)
            is_real  = batch["is_real"]
            start_np = batch["start_pos"]
            goal_np  = batch["goal_pos"]

            pred = model(graph, cond)
            loss, pos_l, shp_l, obs_l = total_loss(pred, target, cubes, start_np, goal_np)
            w    = REAL_DOMAIN_WEIGHT if is_real else VIRTUAL_DOMAIN_WEIGHT
            optimizer.zero_grad()
            (w * loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), NET1_GRAD_CLIP)
            optimizer.step()
            logs.append(dict(total=loss.item(), position=pos_l.item(),
                              shape=shp_l.item(), obstacle=obs_l.item()))
        scheduler.step()

        if logs:
            avgs = {k: np.mean([x[k] for x in logs]) for k in logs[0]}
            for k, v in avgs.items(): history[k].append(v)
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d} | total={avgs['total']:.4f}  "
                  f"pos={avgs['position']:.4f}  shp={avgs['shape']:.4f}  "
                  f"obs={avgs['obstacle']:.4f}  lr={lr:.2e}")

        if (epoch + 1) % NET1_PLOT_INTERVAL == 0:
            torch.save(model.state_dict(), NET1_CHECKPOINT)
            _plot(history, epoch)

    torch.save(model.state_dict(), NET1_CHECKPOINT)
    _plot(history, NET1_EPOCHS - 1, final=True)
    print(f"\nCheckpoint → {NET1_CHECKPOINT}")


def _plot(history, epoch, final=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, v in history.items(): ax.plot(v, label=k)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Network I Training Loss (epoch {epoch})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("network_i_final.png" if final else "network_i_progress.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    train()
