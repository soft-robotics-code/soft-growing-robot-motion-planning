"""
network_ii/dataset.py  —  Offline RL dataset for Network II (CQL).

Data format
-----------
Both sim and real JSON files share the same schema:

    [
      {
        "states":              [[l, ex, ez, θ, gx, gz], ...],
        "actions":             [a0, a1, ...],
        "rotation_end_point":  [x, z]
      },
      ...
    ]

where each state is the 6-D vector
  [robot_length, end_x, end_z, growth_angle, goal_x, goal_z].

This file is produced by collect_rl_data.py + SoftRobotRLDataCollector.cs.
"""

import json
import numpy as np
from torch.utils.data import Dataset

from configs import RL_ZERO_ACTION_THR, RL_NONZERO_ACTION_THR, RL_NEAR_DIST, RL_FAR_DIST


# ── Reward function ───────────────────────────────────────────────────────────

def _compute_reward(state, action, next_end_point):
    """
    Distance-adaptive reward (Eq. 30 in the paper).

    Three regimes based on d1 = ||end − goal||:
      near (d < 0.1 m)  : reward zero actions, penalise non-zero
      mid  (0.1–0.25 m) : moderate encouragement of non-zero
      far  (d > 0.25 m) : strongly reward non-zero that reduce distance
    """
    a_mag = abs(float(action))
    end   = np.array(state[1:3])
    goal  = np.array(state[4:6])
    nxt   = np.array(next_end_point)
    d1    = np.linalg.norm(end - goal)
    Δd    = d1 - np.linalg.norm(nxt - goal)   # positive = improvement

    is_zero = a_mag < RL_ZERO_ACTION_THR

    if is_zero:
        if d1 < RL_NEAR_DIST:   r = +8.0
        elif d1 < RL_FAR_DIST:  r = -1.0 - (d1 - RL_NEAR_DIST) / (RL_FAR_DIST - RL_NEAR_DIST)
        else:                   r = -8.0
    else:
        if d1 > RL_FAR_DIST:
            r = 12.0 * min(Δd / 0.1, 1.0) + (3.0 if Δd > 0.05 else 0) + \
                (4.0 + 6.0 * Δd if Δd <= 0 else 0) + 1.5 * min(a_mag / 15.0, 1.0)
        elif d1 > RL_NEAR_DIST:
            r = 8.0 * min(Δd / 0.08, 1.0) + (2.0 if Δd > 0.03 else 0) + \
                ((1.0 + 4.0 * np.tanh(Δd * 3.0)) if Δd <= 0 else 0)
        else:
            r = -6.0 - 2.0 * a_mag / 10.0 + (12.0 * Δd if Δd > 0.02 else 0)

    return float(np.clip(r, -30, 30))


# ── Dataset ───────────────────────────────────────────────────────────────────

class OfflineRLDataset(Dataset):
    """
    Offline RL dataset for Network II.

    Each item is a tuple of 11 float tensors:
      state, action, reward, next_state, done,
      distance, is_last_step, steps_since_nonzero,
      data_source (1=real, 0=sim), is_nonzero, is_success
    """

    def __init__(self, json_file: str, data_source: str = "sim"):
        """
        Parameters
        ----------
        json_file   : path to the dataset JSON
        data_source : "sim" or "real" — used for BC loss weighting
        """
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.data_source = data_source
        self.transitions = []
        self.trajectories = []   # full trajectory objects for validation
        self._process(raw)

        distances = np.array([t["distance"] for t in self.transitions])
        actions   = np.array([t["action"][0] for t in self.transitions])
        print(f"\n[Net-II Dataset] {data_source.upper()} — {len(self.transitions)} transitions, "
              f"{len(self.trajectories)} trajectories")
        print(f"  distance: [{distances.min():.3f}, {distances.max():.3f}]  "
              f"action: [{actions.min():.1f}, {actions.max():.1f}]")

    def _process(self, raw):
        for traj_idx, traj in enumerate(raw):
            states  = traj["states"]
            actions = traj["actions"]
            rot_end = traj["rotation_end_point"]

            traj_info = {"states": [], "actions": [], "next_end_points": []}
            steps_nz  = 0

            for i, action in enumerate(actions):
                state = np.array(states[i], dtype=np.float32)
                act   = np.array([action], dtype=np.float32)

                if i < len(states) - 1:
                    next_state   = np.array(states[i + 1], dtype=np.float32)
                    next_end     = next_state[1:3]
                else:
                    next_state   = state.copy()
                    next_state[1:3] = rot_end
                    next_end     = np.array(rot_end)

                traj_info["states"].append(state)
                traj_info["actions"].append(act)
                traj_info["next_end_points"].append(next_end)

                distance   = float(np.linalg.norm(state[1:3] - state[4:6]))
                is_last    = (i == len(actions) - 1)
                reward     = _compute_reward(state, action, next_end)

                if abs(action) >= RL_NONZERO_ACTION_THR:
                    steps_nz = 1
                else:
                    steps_nz = steps_nz + 1 if steps_nz > 0 else 0

                is_nz   = abs(action) >= RL_NONZERO_ACTION_THR
                is_succ = (is_nz and
                           np.linalg.norm(next_end - state[4:6]) <
                           np.linalg.norm(state[1:3] - state[4:6]))

                self.transitions.append({
                    "state":          state,
                    "action":         act,
                    "reward":         np.float32(reward),
                    "next_state":     next_state,
                    "done":           np.float32(is_last),
                    "distance":       np.float32(distance),
                    "is_last_step":   np.float32(is_last),
                    "steps_nz":       np.float32(min(steps_nz, 10)),
                    "data_src":       np.float32(1 if self.data_source == "real" else 0),
                    "is_nz":          np.float32(is_nz),
                    "is_succ":        np.float32(is_succ),
                })

            self.trajectories.append(traj_info)

    def __len__(self):  return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return (t["state"], t["action"], t["reward"], t["next_state"], t["done"],
                t["distance"], t["is_last_step"], t["steps_nz"],
                t["data_src"], t["is_nz"], t["is_succ"])

    def get_trajectories(self): return self.trajectories
