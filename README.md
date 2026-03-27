# Sim & Real Co-Training Motion Planning for Soft Growing Robots

Official implementation of:  
**"Motion Planning for Soft Growing Robots via Sim & Real Co-Training with Imitation Learning and Offline Reinforcement Learning"**

---

## Repository Structure

```
soft-growing-robot-planning/
├── configs.py                               ← All hyperparameters (both networks)
│
├── network_i/                               ← Network I: Trajectory Optimisation
│   ├── model.py                             ← GAT–Transformer–MLP architecture
│   ├── dataset.py                           ← Sim & real co-training dataset + sampler
│   ├── losses.py                            ← Position / shape / obstacle / endpoint losses
│   ├── train.py                             ← Training entry point
│   └── inference.py                         ← Inference + post-processing
│
├── network_ii/                              ← Network II: CQL Offline RL
│   ├── model.py                             ← Q-network, bimodal policy, CQL agent
│   ├── dataset.py                           ← Offline RL dataset + reward function
│   └── train.py                             ← Training + validation entry point
│
└── dataset_collection/
    ├── collect_traj_data.py                 ← Python TCP client (Network I data)
    ├── SoftRobotDataCollector.cs            ← Unity controller (Network I data)
    ├── collect_rl_data.py                   ← Python TCP client (Network II data)
    └── SoftRobotRLDataCollector.cs          ← Unity controller (Network II data)
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/soft-growing-robot-motion-planning.git
cd soft-growing-robot-motion-planning
pip install torch torch-geometric numpy scipy matplotlib
```

---

## Two-Layer Hierarchical Framework

```
┌─────────────────────────────────────────────────────┐
│  Network I  —  Trajectory Optimisation              │
│  Input : map + start/goal                           │
│  Output: optimised waypoint sequence T*             │
│  Method: GAT + Transformer + MLP (imitation)        │
└──────────────────────┬──────────────────────────────┘
                       │  T* + pre-bending sequence A*
┌──────────────────────▼──────────────────────────────┐
│  Network II  —  Morphology Control & Error Correct. │
│  Input : 6-D tip state  [l, ex, ez, θ, gx, gz]     │
│  Output: sparse tendon action u (degrees)           │
│  Method: CQL offline RL (bimodal policy)            │
└─────────────────────────────────────────────────────┘
```

---

## Dataset Collection

### Network I — Virtual trajectory data

**Step 1.** In Unity, attach `SoftRobotDataCollector.cs` to a GameObject and press Play.  
**Step 2.** Run the Python client:
```bash
python dataset_collection/collect_traj_data.py
```

Output: `coordinates_new.json`  
Format:
```json
[{"coordinates": [[x, z, action], ...]}, ...]
```

### Network II — Offline RL data (dual-scene comparison)

The RL dataset is collected using a dual-scene Unity setup:
- **Original scene**: clean growth trajectory (no rotation applied)
- **Comparison scene**: same growth + Gaussian noise + sparse tendon rotation actions

**Step 1.** Attach `SoftRobotRLDataCollector.cs` to a GameObject and press Play.  
**Step 2.** Run the Python client:
```bash
python dataset_collection/collect_rl_data.py
```

Output: `coordinates_new.json` and `coordinates_new_comparison.json`  
Format:
```json
[{
  "growth_coordinates":   [[x, z, action], ...],
  "rotation_coordinates": [[x, z, action], ...]
}, ...]
```

The resulting files need to be converted to the RL training format
(`offline_rl_dataset_sim.json` / `offline_rl_dataset_real.json`) using a
preprocessing script. Each entry should follow the schema:
```json
{"states": [[l, ex, ez, θ, gx, gz], ...], "actions": [...], "rotation_end_point": [x, z]}
```

### Real trajectories

Real tip trajectories are captured using an IMU + encoder Kalman filter on
the physical prototype. We do **not** release the physical dataset.

Expected `.mat` format:
```
real_trajectories.mat
  <traj_name>  →  float64 (N, 2)   # [x, z] tip positions (metres)
  ...
```
Update `REAL_TRAJ_NAMES` in `configs.py` to match your variable names.

---

## Training

Update `configs.py` with your file paths, then:

```bash
# Network I — trajectory optimisation
python -m network_i.train

# Network II — CQL offline RL
python -m network_ii.train
```

Key hyperparameters (see `configs.py` for full list):

| Parameter | Default | Note |
|-----------|---------|------|
| `GLOBAL_REAL_DATA_RATIO` | `0.04` | ~4 % real data gives best Net-I loss (Fig. 9b) |
| `RL_SIM_RATIO` | `0.85` | ~85 % sim gives best Net-II reward (Fig. 11d) |
| `NET1_EPOCHS` | `400` | |
| `RL_EPOCHS` | `400` | |

---

## Inference

```bash
python -m network_i.inference
```

Generates an optimised trajectory for the configured start/goal pair,
applies backtracking removal and uniform resampling, and saves results to
`network_i_result.mat` and `network_i_result.png`.

---

## Citation

```bibtex
@article{yourpaper2025,
  title   = {Motion Planning for Soft Growing Robots via Sim \& Real Co-Training},
  author  = {Your Authors},
  journal = {IEEE Transactions on Robotics},
  year    = {2025}
}
```

---

## License

MIT License
