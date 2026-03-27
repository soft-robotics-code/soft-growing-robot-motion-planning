"""
network_i/inference.py  —  Inference entry point for Network I.

Usage:
    python -m network_i.inference
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import scipy.io as sio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import DEVICE, NET1_CHECKPOINT, INFER_TIMESTEPS, RESAMPLE_SPACING, BACKTRACK_ANGLE_THR
from network_i.model   import SoftRobotTrajectoryPlanner
from network_i.dataset import build_graph, _resample, _adaptive_steps


def apf_init(start, goal, cubes, n=50, max_iter=300,
              k_att=1.0, k_rep=2.0, rep_range=0.3, step=0.05):
    """Artificial potential field path initialisation."""
    path, cur = [start.copy()], start.copy()
    for _ in range(max_iter):
        to_g = goal - cur; d_g = np.linalg.norm(to_g) + 1e-9
        if d_g < 0.05: path.append(goal.copy()); break
        f = k_att * to_g / d_g
        for pos, rot, sc in cubes:
            pos, sc = np.asarray(pos, float), np.asarray(sc, float)
            cx, cz = (pos[0], pos[2]) if pos.size == 3 else (pos[0], pos[1])
            r_obs  = max(sc[0], sc[2] if sc.size > 2 else sc[1]) / 2
            v = cur - np.array([cx, cz]); d = np.linalg.norm(v) - r_obs
            if 0 < d < rep_range:
                f += k_rep * (1/d - 1/rep_range) / d**2 * (v / (np.linalg.norm(v)+1e-9))
        fn = np.linalg.norm(f)
        cur = cur + step * f / fn if fn > 0 else cur
        path.append(cur.copy())
        if np.linalg.norm(cur - goal) < 0.05: path.append(goal.copy()); break
    traj = np.array(path)
    arc  = np.zeros(len(traj))
    for i in range(1, len(traj)):
        arc[i] = arc[i-1] + np.linalg.norm(traj[i] - traj[i-1])
    u = np.linspace(0, arc[-1], n)
    return np.stack([np.interp(u, arc, traj[:, d]) for d in range(2)], axis=1)


def remove_backtracking(traj, goal, angle_thr=BACKTRACK_ANGLE_THR):
    N, cut = len(traj), None
    for i in range(N - 2):
        v1 = traj[i+1] - traj[i]; v2 = traj[i+2] - traj[i+1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-9 and n2 > 1e-9:
            a = np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2), -1, 1)))
            if a > angle_thr: cut = i + 1; break
    if cut is None: return traj
    head = traj[:cut+1]; p0 = head[-1]; d = np.linalg.norm(goal - p0)
    n_sm = max(5, int(d / 0.05))
    tail = np.array([(1-t**2*(3-2*t))*p0 + t**2*(3-2*t)*goal
                      for t in np.linspace(0, 1, n_sm+1)[1:]])
    return np.vstack([head, tail])


def resample_uniform(traj, spacing=RESAMPLE_SPACING):
    arc = np.zeros(len(traj))
    for i in range(1, len(traj)):
        arc[i] = arc[i-1] + np.linalg.norm(traj[i] - traj[i-1])
    n = max(2, int(arc[-1] / spacing) + 1)
    u = np.linspace(0, arc[-1], n)
    return np.stack([np.interp(u, arc, traj[:, d]) for d in range(2)], axis=1)


def infer(model, start, goal, cubes, checkpoint=None):
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    device = next(model.parameters()).device
    start, goal = np.asarray(start, float)[:2], np.asarray(goal, float)[:2]

    init = apf_init(start, goal, cubes, n=INFER_TIMESTEPS)
    init[0], init[-1] = start, goal
    graph = build_graph(init, cubes, start, goal)
    graph.x = graph.x.contiguous()
    graph   = graph.to(device)
    cond    = torch.tensor(np.concatenate([start, goal]), dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(graph, cond).cpu().numpy()

    pred = remove_backtracking(pred, goal)
    pred = resample_uniform(pred)
    pred[0], pred[-1] = start, goal
    return pred


if __name__ == "__main__":
    from network_i.losses import _cuboid_polygon

    cubes = [([0.352, 0.214, 1.762], 63.81, [2.01, 0.72, 0.49])]
    start = np.array([0.0,   0.1])
    goal  = np.array([-0.31, 4.02])

    model = SoftRobotTrajectoryPlanner().to(DEVICE)
    traj  = infer(model, start, goal, cubes, checkpoint=NET1_CHECKPOINT)
    length = np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()
    print(f"Trajectory: {len(traj)} points, length = {length:.3f} m")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2.5, label="Predicted trajectory")
    ax.scatter(*start, color="green", s=120, zorder=5, label="Start")
    ax.scatter(*goal,  color="red",   s=120, marker="*", zorder=5, label="Goal")
    for pos, rot, sc in cubes:
        poly = _cuboid_polygon(pos, rot, sc)
        ax.fill(poly[:, 0], poly[:, 1], color="gray", alpha=0.4)
        ax.plot(*np.vstack([poly, poly[0]]).T, "k-", lw=1.5)
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title("Network I — Trajectory Optimisation")
    plt.tight_layout(); plt.savefig("network_i_result.png", dpi=200, bbox_inches="tight")
    plt.close()

    sio.savemat("network_i_result.mat", {"trajectory": traj, "start": start, "goal": goal})
    print("Saved → network_i_result.mat / network_i_result.png")
