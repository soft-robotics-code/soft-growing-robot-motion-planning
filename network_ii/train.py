"""
network_ii/train.py  —  Training and evaluation entry point for Network II (CQL).

Usage:
    python -m network_ii.train
    # or
    cd network_ii && python train.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import scipy.io as sio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import (DEVICE, RL_SIM_JSON, RL_REAL_JSON, RL_EPOCHS, RL_BATCH_SIZE,
                     RL_SIM_RATIO, RL_CHECKPOINT, RL_CHECKPOINT_BEST, RL_VAL_INTERVAL,
                     RL_NONZERO_ACTION_THR)
from network_ii.model   import DistanceAdaptiveCQL
from network_ii.dataset import OfflineRLDataset


# ── Validation ────────────────────────────────────────────────────────────────

def validate(agent, dataset, num_trajs=None):
    """
    Evaluate the policy on the first non-zero-action step of each trajectory.

    Success criterion: model outputs a non-zero action AND the distance to
    the goal decreases (using the ground-truth next_end_point from the dataset).
    """
    trajs = dataset.get_trajectories()
    if num_trajs: trajs = trajs[:num_trajs]

    total_r, nz_count, succ_count = 0.0, 0, 0

    for traj in trajs:
        states, actions_gt, next_ends = (traj["states"], traj["actions"], traj["next_end_points"])

        first_nz = next((i for i, a in enumerate(actions_gt)
                          if abs(float(a[0])) >= RL_NONZERO_ACTION_THR), None)
        if first_nz is None:
            continue

        nz_count += 1
        s    = states[first_nz]
        nend = next_ends[first_nz]

        pred_a   = agent.select_action(s, deterministic=True)
        pred_mag = abs(float(pred_a[0]))
        d_before = np.linalg.norm(s[1:3] - s[4:6])
        d_after  = np.linalg.norm(nend - s[4:6])
        Δd       = d_before - d_after

        total_r += 10.0 * Δd if Δd > 0 else -5.0 * abs(Δd)
        if pred_mag >= RL_NONZERO_ACTION_THR and Δd > 0:
            succ_count += 1

    n = max(len(trajs), 1)
    return {
        "avg_reward":   total_r / n,
        "success_rate": succ_count / max(nz_count, 1),
        "nz_trajs":     nz_count,
        "total_trajs":  len(trajs),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def _build_batch(sim_ds, real_ds, batch_size, sim_ratio):
    n_sim  = int(batch_size * sim_ratio)
    n_real = batch_size - n_sim
    items  = []
    for ds, n in [(sim_ds, n_sim), (real_ds, n_real)]:
        idxs = np.random.randint(0, len(ds), n)
        for i in idxs: items.append(ds[i])
    batch = []
    for k in range(11):
        tensors = []
        for item in items:
            v = item[k]
            if isinstance(v, np.ndarray):
                tensors.append(torch.from_numpy(v).float())
            else:
                tensors.append(torch.tensor(float(v), dtype=torch.float32).unsqueeze(0))
        batch.append(torch.stack(tensors))
    return tuple(batch)


def train():
    print("=" * 60)
    print("Network II — CQL Offline RL (Morphology Control & Error Correction)")
    print(f"Device: {DEVICE}   Sim ratio: {RL_SIM_RATIO:.0%}")
    print("=" * 60)

    sim_ds  = OfflineRLDataset(RL_SIM_JSON,  data_source="sim")
    real_ds = OfflineRLDataset(RL_REAL_JSON, data_source="real")

    agent = DistanceAdaptiveCQL(device=str(DEVICE))

    os.makedirs(os.path.dirname(RL_CHECKPOINT), exist_ok=True)

    history = {k: [] for k in ("policy_loss", "q_loss", "avg_reward",
                                "train_success_rate", "val_reward", "val_success_rate")}
    best_score = -9e9

    steps_per_epoch = max(len(sim_ds), len(real_ds)) // RL_BATCH_SIZE

    print(f"\n{'Epoch':>6} | {'QLoss':>7} | {'PLoss':>7} | "
          f"{'Rwd':>7} | {'TrnSR':>6} | {'ValR':>7} | {'ValSR':>6}")
    print("=" * 60)

    for epoch in range(RL_EPOCHS):
        agent.step_schedulers()

        pl_acc, ql_acc, r_acc, sr_acc = 0.0, 0.0, 0.0, 0.0
        n_steps = 0

        for _ in range(steps_per_epoch):
            try:
                batch  = _build_batch(sim_ds, real_ds, RL_BATCH_SIZE, RL_SIM_RATIO)
                losses = agent.train_step(batch)
                if any(np.isnan(v) or np.isinf(v) for v in losses.values()):
                    continue
                pl_acc  += losses["policy_loss"]
                ql_acc  += losses["q_loss"]
                r_acc   += losses["avg_reward"]
                sr_acc  += losses["train_success_rate"]
                n_steps += 1
            except Exception as e:
                print(f"  [WARN] step error: {e}")
                continue

        if n_steps == 0: continue

        avg = lambda x: x / n_steps
        history["policy_loss"].append(avg(pl_acc))
        history["q_loss"].append(avg(ql_acc))
        history["avg_reward"].append(avg(r_acc))
        history["train_success_rate"].append(avg(sr_acc))

        # Validation
        val_r, val_sr = "", ""
        if (epoch + 1) % RL_VAL_INTERVAL == 0 or epoch == 0:
            vm = validate(agent, real_ds, num_trajs=100)
            history["val_reward"].append(vm["avg_reward"])
            history["val_success_rate"].append(vm["success_rate"])
            val_r  = f"{vm['avg_reward']:7.2f}"
            val_sr = f"{vm['success_rate']:6.1%}"

            score = vm["avg_reward"] + vm["success_rate"] * 10
            if score > best_score:
                best_score = score
                agent.save(RL_CHECKPOINT_BEST)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:6d} | {avg(ql_acc):7.3f} | {avg(pl_acc):7.3f} | "
                  f"{avg(r_acc):7.2f} | {avg(sr_acc):6.1%} | {val_r:>7} | {val_sr:>6}")

    agent.save(RL_CHECKPOINT)
    _save_plots(history)
    _save_mat(history)
    print(f"\nDone. Final checkpoint → {RL_CHECKPOINT}")


def _save_plots(history):
    keys = [("policy_loss", "Policy Loss"), ("q_loss", "Q-Network Loss"),
            ("avg_reward",  "Avg Reward"),  ("train_success_rate", "Train Success Rate")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (k, title) in zip(axes.flat, keys):
        v = history[k]
        if not v: continue
        ax.plot(v, alpha=0.4, color="steelblue")
        w = min(10, len(v))
        ema = []
        for i, x in enumerate(v):
            ema.append(np.mean(v[max(0, i-w+1):i+1]))
        ax.plot(ema, color="steelblue", lw=2, label="EMA")
        if k == "train_success_rate" and history["val_success_rate"]:
            val_ep = list(range(RL_VAL_INTERVAL, RL_EPOCHS + 1, RL_VAL_INTERVAL))[:len(history["val_success_rate"])]
            ax.plot(val_ep, history["val_success_rate"], "o-", color="crimson", lw=2, label="Val SR")
        if k == "avg_reward" and history["val_reward"]:
            val_ep = list(range(RL_VAL_INTERVAL, RL_EPOCHS + 1, RL_VAL_INTERVAL))[:len(history["val_reward"])]
            ax.plot(val_ep, history["val_reward"], "o-", color="forestgreen", lw=2, label="Val Reward")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("network_ii_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Training curves saved → network_ii_training.png")


def _save_mat(history):
    sio.savemat("network_ii_history.mat",
                {k: np.array(v, dtype=np.float64) for k, v in history.items()})
    print("Training history saved → network_ii_history.mat")


if __name__ == "__main__":
    train()
