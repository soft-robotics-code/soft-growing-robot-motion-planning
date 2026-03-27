"""
network_ii/dataset.py  —  Offline RL dataset for Network II (CQL).

JSON schema (matches offline_rl_dataset_real.json provided in data/):

    [
      {
        "trajectory_id": 0,
        "length": 10,
        "states": [
          [robot_length, end_x, end_z, growth_angle, goal_x, goal_z],
          ...
        ],
        "actions": [0, 0, ..., 14.23],
        "rotation_end_point": [x, z]
      },
      ...
    ]

State vector (6-D):
  [0] robot_length  : cumulative robot length (m)
  [1] end_x         : tip X position, original scene (m)
  [2] end_z         : tip Z position, original scene (m)
  [3] growth_angle  : growth direction angle (degrees)
  [4] goal_x        : tip X position, comparison/noisy scene (m)
  [5] goal_z        : tip Z position, comparison/noisy scene (m)

Actions are sparse: mostly 0, non-zero only when tendon steering is applied.
rotation_end_point is the tip position after the final rotation action.
"""

import json
import numpy as np
from torch.utils.data import Dataset

from configs import RL_ZERO_ACTION_THR, RL_NONZERO_ACTION_THR, RL_NEAR_DIST, RL_FAR_DIST


# ── Reward function ───────────────────────────────────────────────────────────

def _compute_reward(state, action, next_end_point):
    """
    Distance-adaptive reward (Eq. 30 in the paper).

    Three regimes based on d = ||end − goal||:
      near  (d < 0.10 m)  : reward zero actions, penalise non-zero
      mid   (0.10–0.25 m) : moderate encouragement for non-zero
      far   (d > 0.25 m)  : strongly reward non-zero that reduce distance
    """
    a_mag = abs(float(action))
    end   = np.array([state[1], state[2]])   # original scene tip
    goal  = np.array([state[4], state[5]])   # comparison scene tip (serves as reference goal)
    nxt   = np.array(next_end_point)
    d1    = np.linalg.norm(end - goal)
    Δd    = d1 - np.linalg.norm(nxt - goal)   # positive → improvement

    is_zero = a_mag < RL_ZERO_ACTION_THR

    if is_zero:
        if d1 < RL_NEAR_DIST:    r = +8.0
        elif d1 < RL_FAR_DIST:   r = -1.0 - (d1 - RL_NEAR_DIST) / (RL_FAR_DIST - RL_NEAR_DIST)
        else:                    r = -8.0
    else:
        if d1 > RL_FAR_DIST:
            r = (12.0 * min(Δd / 0.1, 1.0) + (3.0 if Δd > 0.05 else 0) +
                 (4.0 + 6.0 * Δd if Δd <= 0 else 0) +
                 1.5 * min(a_mag / 15.0, 1.0))
        elif d1 > RL_NEAR_DIST:
            r = (8.0 * min(Δd / 0.08, 1.0) + (2.0 if Δd > 0.03 else 0) +
                 ((1.0 + 4.0 * np.tanh(Δd * 3.0)) if Δd <= 0 else 0))
        else:
            r = -6.0 - 2.0 * a_mag / 10.0 + (12.0 * Δd if Δd > 0.02 else 0)

    return float(np.clip(r, -30, 30))


# ── Dataset ───────────────────────────────────────────────────────────────────

class OfflineRLDataset(Dataset):
    """
    Offline RL dataset for Network II.

    Reads from a JSON file matching the schema in data/README.md.
    Each __getitem__ returns a tuple of 11 float arrays:
      (state, action, reward, next_state, done,
       distance, is_last_step, steps_since_nonzero,
       data_source, is_nonzero, is_success)
    """

    def __init__(self, json_file: str, data_source: str = "sim"):
        """
        Parameters
        ----------
        json_file   : path to the dataset JSON (e.g. data/offline_rl_dataset_real.json)
        data_source : "sim" or "real" — used for BC loss weighting in training
        """
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.data_source  = data_source
        self.transitions  = []
        self.trajectories = []   # full trajectory objects for validation
        self._process(raw)

        distances = np.array([t["distance"] for t in self.transitions])
        actions   = np.array([t["action"][0] for t in self.transitions])
        print(f"\n[Net-II Dataset] {data_source.upper()} — "
              f"{len(self.transitions)} transitions from {len(self.trajectories)} trajectories")
        print(f"  distance: [{distances.min():.3f}, {distances.max():.3f}] m  "
              f"action: [{actions.min():.2f}, {actions.max():.2f}] deg")

    def _process(self, raw):
        for traj in raw:
            states     = traj["states"]
            actions    = traj["actions"]
            rot_end    = traj["rotation_end_point"]   # [x, z]

            traj_info  = {"states": [], "actions": [], "next_end_points": []}
            steps_nz   = 0

            for i, action in enumerate(actions):
                state = np.array(states[i], dtype=np.float32)
                act   = np.array([action], dtype=np.float32)

                # Next tip position
                if i < len(states) - 1:
                    next_state = np.array(states[i + 1], dtype=np.float32)
                    next_end   = next_state[1:3]
                else:
                    # Last step: robot tip moves to rotation_end_point
                    next_state      = state.copy()
                    next_state[1:3] = rot_end
                    next_end        = np.array(rot_end, dtype=np.float32)

                traj_info["states"].append(state)
                traj_info["actions"].append(act)
                traj_info["next_end_points"].append(next_end)

                distance = float(np.linalg.norm(state[1:3] - state[4:6]))
                is_last  = (i == len(actions) - 1)
                reward   = _compute_reward(state, action, next_end)

                # Steps since last non-zero action
                if abs(action) >= RL_NONZERO_ACTION_THR:
                    steps_nz = 1
                else:
                    steps_nz = steps_nz + 1 if steps_nz > 0 else 0

                is_nz   = abs(action) >= RL_NONZERO_ACTION_THR
                is_succ = (is_nz and
                           np.linalg.norm(next_end - state[4:6]) <
                           np.linalg.norm(state[1:3] - state[4:6]))

                self.transitions.append({
                    "state":      state,
                    "action":     act,
                    "reward":     np.float32(reward),
                    "next_state": next_state,
                    "done":       np.float32(is_last),
                    "distance":   np.float32(distance),
                    "is_last":    np.float32(is_last),
                    "steps_nz":   np.float32(min(steps_nz, 10)),
                    "data_src":   np.float32(1 if self.data_source == "real" else 0),
                    "is_nz":      np.float32(is_nz),
                    "is_succ":    np.float32(is_succ),
                })

            self.trajectories.append(traj_info)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return (t["state"], t["action"], t["reward"], t["next_state"], t["done"],
                t["distance"], t["is_last"], t["steps_nz"],
                t["data_src"], t["is_nz"], t["is_succ"])

    def get_trajectories(self):
        """Return all trajectory objects (used by the validation function)."""
        return self.trajectories
