# Dataset

This directory contains partial real-world datasets used in the paper.
Virtual (simulated) datasets are not included due to size; please use the
provided collection scripts to regenerate them.

---

## Provided files

| File | Network | Description |
|------|---------|-------------|
| `real_trajectories_env1.mat` | Network I | Real tip trajectories collected in Env 1 (single obstacle) |
| `offline_rl_dataset_real.json` | Network II | Real offline RL trajectories (partial) |

---

## Network I — Real trajectory format

File: `real_trajectories_env1.mat`

Each variable is a `float64` array of shape `(N, 2)` containing the robot tip
position `[x, z]` (in metres) recorded at each growth step.

```
real_trajectories_env1.mat
  <traj_name>  →  float64 (N, 2)   # columns: [x, z] in metres
  ...
```

Update `REAL_TRAJ_NAMES` and `REAL_MAT_FILE` in `configs.py` to match your
variable names.

To collect additional real trajectories on your own platform, refer to the
**Vision-based trajectory capture** section in the top-level README.

---

## Network II — Offline RL dataset format

File: `offline_rl_dataset_real.json`

Each entry is one trajectory episode:

```json
{
  "trajectory_id": 0,
  "length": 10,
  "states": [
    [robot_length, end_x, end_z, growth_angle, goal_x, goal_z],
    ...
  ],
  "actions": [0, 0, ..., 14.23],
  "rotation_end_point": [x, z]
}
```

**State vector (6-D):**

| Index | Field | Unit | Description |
|-------|-------|------|-------------|
| 0 | `robot_length` | m | Cumulative robot length |
| 1 | `end_x` | m | Tip X position (original scene) |
| 2 | `end_z` | m | Tip Z position (original scene) |
| 3 | `growth_angle` | deg | Growth direction angle |
| 4 | `goal_x` | m | Tip X position (comparison/noisy scene) |
| 5 | `goal_z` | m | Tip Z position (comparison/noisy scene) |

**Fields:**

- `actions`: sparse tendon displacement sequence (degrees); most steps are `0`,
  non-zero only when active steering is applied.
- `rotation_end_point`: tip position after the final rotation action is executed.

To collect additional real RL data, use `dataset_collection/collect_rl_data.py`
with `SoftRobotRLDataCollector.cs` in Unity.
