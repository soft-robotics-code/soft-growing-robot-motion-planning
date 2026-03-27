"""
network_i/dataset.py  —  Sim & Real Co-Training dataset for Network I.

Virtual trajectories come from the Unity simulation (SoftRobotDataCollector.cs).
Real trajectories come from physical experiments stored in a .mat file.
"""

from __future__ import annotations
import json, os
import numpy as np
import scipy.io as sio
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.data import Data

from configs import (
    REAL_TRAJ_NAMES, MAPPED_VIRTUAL_COUNT, NUM_VIRTUAL_PER_REAL,
    SAMPLES_PER_GROUP, SAMPLE_OFFSET, RANDOM_SAMPLES,
    REAL_AUG_FACTOR, NOISE_LEVELS, NUM_SEGMENTS_REAL, NUM_SEGMENTS_VIRTUAL,
    REAL_TIMESTEPS, VIRTUAL_TIMESTEPS, SPATIAL_EDGE_THRESHOLD,
    GLOBAL_REAL_DATA_RATIO,
    VIRTUAL_SIMILARITY_WEIGHT, VIRTUAL_DIVERSITY_WEIGHT, RANDOM_DIVERSITY_WEIGHT,
)


# ── File loading ──────────────────────────────────────────────────────────────

def load_real_trajectories(mat_path: str) -> list[np.ndarray]:
    """Load real trajectories from a .mat file (variables listed in REAL_TRAJ_NAMES)."""
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Real trajectory file not found: {mat_path}")
    mat, trajs = sio.loadmat(mat_path), []
    for name in REAL_TRAJ_NAMES:
        if name not in mat:
            print(f"  [WARN] '{name}' not in {mat_path}")
            continue
        arr = mat[name]
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
        trajs.append(arr.astype(np.float32))
    print(f"[Net-I Dataset] Loaded {len(trajs)} real trajectories")
    return trajs


def load_virtual_trajectories(json_path: str) -> list[np.ndarray]:
    """
    Load virtual trajectories from a .json file produced by SoftRobotDataCollector.cs.

    Expected schema:
        [{"coordinates": [[x, z], ...]}, ...]
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Virtual trajectory file not found: {json_path}")
    with open(json_path, "r") as f:
        raw = json.load(f)
    trajs = []
    for item in raw:
        coords = item["coordinates"] if isinstance(item, dict) else item
        arr = np.asarray(coords, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 10:
            continue
        trajs.append(arr[:, :2])
    print(f"[Net-I Dataset] Loaded {len(trajs)} virtual trajectories")
    return trajs


# ── Trajectory utilities ──────────────────────────────────────────────────────

def _curvature(traj):
    k = np.zeros(len(traj))
    for i in range(1, len(traj) - 1):
        v1, v2 = traj[i] - traj[i-1], traj[i+1] - traj[i]
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        k[i] = abs(cross) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return k


def _adaptive_steps(traj, is_real: bool) -> int:
    cfg = REAL_TIMESTEPS if is_real else VIRTUAL_TIMESTEPS
    arc = np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()
    k   = _curvature(traj)
    n   = int(cfg["base"] * (1 + cfg["density"] * (arc - 3.0))) + int(10 * (k.mean() + k.std()))
    return int(np.clip(n, cfg["min"], cfg["max"]))


def _resample(traj, target, is_real):
    arc = np.zeros(len(traj))
    for i in range(1, len(traj)):
        arc[i] = arc[i-1] + np.linalg.norm(traj[i] - traj[i-1])
    if arc[-1] < 1e-9:
        return traj
    u = np.linspace(0, arc[-1], target)
    if is_real and len(traj) > 10:
        out = np.stack([interp1d(arc, traj[:, d], kind="cubic",
                                  bounds_error=False, fill_value="extrapolate")(u)
                         for d in range(2)], axis=1)
    else:
        out = np.stack([np.interp(u, arc, traj[:, d]) for d in range(2)], axis=1)
        s, e = out[0].copy(), out[-1].copy()
        for d in range(2):
            out[:, d] = gaussian_filter1d(out[:, d], sigma=1.0)
        out[0], out[-1] = s, e
    return out


def build_graph(traj, cubes, start, goal):
    """Build a PyG Data object with 36-D node features."""
    T = len(traj)
    s, g = start[:2], goal[:2]
    task_scale = np.linalg.norm(g - s) + 1e-9

    arc = np.zeros(T)
    for i in range(1, T):
        arc[i] = arc[i-1] + np.linalg.norm(traj[i] - traj[i-1])
    arc_total = max(arc[-1], 1e-9)

    feats = np.zeros((T, 36), dtype=np.float32)
    for i, pos in enumerate(traj):
        pos = pos[:2]
        feats[i, 0], feats[i, 1], feats[i, 2] = pos[0], pos[1], i / max(T-1, 1)

        col = 3
        for cube in (cubes or [])[:5]:
            try:
                p, r, sc = cube
                p, sc = np.asarray(p, float), np.asarray(sc, float)
                cx, cz = (p[0], p[2]) if p.size == 3 else (p[0], p[1])
                w = sc[0]; d = (sc[2] if sc.size > 2 else sc[1] if sc.size > 1 else 0.)
                feats[i, col:col+3] = [np.linalg.norm(pos - [cx, cz]), w, d]
            except Exception:
                pass
            col += 3

        to_g = g - pos; d_g = np.linalg.norm(to_g); gdir = to_g / (d_g + 1e-9)
        feats[i, 23:26] = [gdir[0], gdir[1], d_g / task_scale]
        to_s = s - pos; d_s = np.linalg.norm(to_s); sdir = to_s / (d_s + 1e-9)
        feats[i, 26:29] = [sdir[0], sdir[1], d_s / task_scale]
        feats[i, 29] = arc[i] / arc_total
        feats[i, 30] = d_s / task_scale

        if 0 < i < T-1: vel = (traj[i+1] - traj[i-1]) / 2
        elif i == 0:     vel = traj[1] - traj[0] if T > 1 else np.zeros(2)
        else:            vel = traj[-1] - traj[-2]
        vel = vel[:2]; vn = np.linalg.norm(vel)
        if vn > 1e-9:
            feats[i, 31] = float(np.dot(vel / vn, gdir))
        feats[i, 32] = 1.0
        feats[i, 33] = float(np.dot(vel, gdir)) / task_scale
        feats[i, 34] = float(np.sqrt(max(0, vn**2 - np.dot(vel, gdir)**2))) / task_scale
        feats[i, 35] = task_scale

    edges, attrs = [], []
    for i in range(T - 1):
        d = float(np.linalg.norm(traj[i+1] - traj[i]))
        edges += [[i, i+1], [i+1, i]]; attrs += [[d, 1.], [d, 1.]]
    for i in range(T):
        for j in range(i+2, T):
            d = float(np.linalg.norm(traj[j] - traj[i]))
            if d < SPATIAL_EDGE_THRESHOLD:
                edges += [[i, j], [j, i]]; attrs += [[d, 0.], [d, 0.]]

    x  = torch.tensor(feats, dtype=torch.float32)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    ea = torch.tensor(attrs, dtype=torch.float32) if attrs else torch.zeros((0, 2))
    g_data = Data(x=x, edge_index=ei, edge_attr=ea)
    g_data.traj_indices = torch.arange(T, dtype=torch.long)
    return g_data


def _make_sample(traj, cubes, is_real, start, goal):
    if len(traj) < 3:
        return None
    try:
        T    = _adaptive_steps(traj, is_real)
        proc = _resample(traj, T, is_real)
        graph = build_graph(proc, cubes, start, goal)
        cond  = torch.tensor(np.concatenate([start[:2], goal[:2]]), dtype=torch.float32)
        return {"graph": graph, "condition": cond,
                "target": torch.tensor(proc, dtype=torch.float32),
                "start_pos": start[:2], "goal_pos": goal[:2], "is_real": is_real}
    except Exception:
        return None


def _segments(traj, n_segs):
    T, segs = len(traj), [(traj.copy(), traj[0], traj[-1])]
    min_len = max(5, int(T * 0.3))
    rng = np.random.default_rng()
    for _ in range(n_segs // 2):
        e = int(rng.integers(min_len, T))
        segs.append((traj[:e+1].copy(), traj[0], traj[e]))
    for _ in range(n_segs // 2):
        s = int(rng.integers(0, T - min_len))
        segs.append((traj[s:].copy(), traj[s], traj[-1]))
    return segs


# ── Dataset ───────────────────────────────────────────────────────────────────

class CotrainingDataset(Dataset):
    """
    Sim & Real Co-Training dataset with weighted mini-batch sampling.

    Parameters
    ----------
    real_trajs    : list of (N, 2) real trajectories
    virtual_trajs : list of (M, 2) virtual trajectories (from load_virtual_trajectories)
    cubes         : list of (pos, rotation_y, scale) obstacle definitions
    real_ratio    : effective proportion of real data per mini-batch
    """

    def __init__(self, real_trajs, virtual_trajs, cubes, real_ratio=GLOBAL_REAL_DATA_RATIO):
        self.cubes = cubes
        mapped  = virtual_trajs[:MAPPED_VIRTUAL_COUNT]
        random_ = virtual_trajs[MAPPED_VIRTUAL_COUNT:]

        print("[Net-I Dataset] Processing real trajectories …")
        self.real_data    = self._process_real(real_trajs)
        print("[Net-I Dataset] Processing virtual trajectories …")
        self.virtual_data = self._process_virtual(real_trajs, mapped, random_)
        self._build_sampler(real_ratio)
        print(f"[Net-I Dataset] real={len(self.real_data)}  virtual={len(self.virtual_data)}")

    def _process_real(self, trajs):
        samples = []
        for traj in trajs:
            for seg, s, g in _segments(traj, NUM_SEGMENTS_REAL):
                sp = _make_sample(seg, self.cubes, True, s, g)
                if sp: samples.append(sp)
            for _ in range(REAL_AUG_FACTOR):
                for sigma in NOISE_LEVELS:
                    aug = traj + np.random.normal(0, sigma, traj.shape).astype(np.float32)
                    aug[0], aug[-1] = traj[0], traj[-1]
                    sp = _make_sample(aug, self.cubes, True, traj[0], traj[-1])
                    if sp: samples.append(sp)
        return samples

    def _process_virtual(self, real_trajs, mapped, random_):
        samples = []
        for r_idx, r_traj in enumerate(real_trajs):
            base = r_idx * NUM_VIRTUAL_PER_REAL
            for v_idx in range(SAMPLE_OFFSET, SAMPLE_OFFSET + SAMPLES_PER_GROUP):
                abs_idx = base + v_idx
                if abs_idx >= len(mapped): break
                v = mapped[abs_idx]
                for seg, s, g in _segments(v, NUM_SEGMENTS_VIRTUAL):
                    sp = _make_sample(seg, self.cubes, False, s, g)
                    if sp:
                        sp["traj_type"] = "mapped"; samples.append(sp)
        rng  = np.random.default_rng()
        idxs = rng.choice(len(random_), min(RANDOM_SAMPLES, len(random_)), replace=False)
        for i in idxs:
            v  = random_[i]
            sp = _make_sample(v, self.cubes, False, v[0], v[-1])
            if sp:
                sp["traj_type"] = "random"; samples.append(sp)
        return samples

    def _build_sampler(self, rho):
        N_r, N_v = len(self.real_data), len(self.virtual_data)
        w_real   = (rho * N_v) / ((1 - rho) * N_r) if N_r and N_v else 1.0
        weights, self._index = [], []
        for i in range(N_r):
            weights.append(w_real); self._index.append(("real", i))
        for i, vs in enumerate(self.virtual_data):
            w = RANDOM_DIVERSITY_WEIGHT if vs.get("traj_type") == "random" \
                else (VIRTUAL_SIMILARITY_WEIGHT * 0.7 + VIRTUAL_DIVERSITY_WEIGHT * 0.3)
            weights.append(w); self._index.append(("virtual", i))
        self.sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        r_tot = w_real * N_r; v_tot = sum(weights[N_r:])
        actual = r_tot / (r_tot + v_tot) if (r_tot + v_tot) > 0 else 0
        print(f"[Net-I Dataset] Sampler: target={rho:.1%}  actual={actual:.1%}")

    def __len__(self):    return len(self._index)
    def __getitem__(self, idx):
        src, local = self._index[idx]
        item = (self.real_data if src == "real" else self.virtual_data)[local].copy()
        item["sampled_from"] = src
        return item
