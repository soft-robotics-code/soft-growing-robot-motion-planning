# Trajectory Optimization and Morphology Control for Soft Growing Robots via Sim & Real Co-training

Official implementation of:  
**"Trajectory Optimization and Morphology Control for Soft Growing Robots via Sim & Real Co-training"**

🔗 Repository: https://github.com/soft-robotics-code/soft-growing-robot-motion-planning

---

## Repository Structure

```
soft-growing-robot-motion-planning/
├── configs.py                               ← All hyperparameters (both networks)
├── data/
│   ├── README.md                            ← Dataset format documentation
│   ├── real_trajectories_env1.mat           ← Real trajectories, Env 1 (Network I)
│   └── offline_rl_dataset_real.json         ← Real RL dataset, partial (Network II)
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
git clone https://github.com/soft-robotics-code/soft-growing-robot-motion-planning.git
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

## Dataset

### Provided datasets

Partial real-world datasets are provided in `data/` to facilitate
reproducibility:

| File | Network | Description |
|------|---------|-------------|
| `data/real_trajectories_env1.mat` | Network I | Real tip trajectories, Env 1 |
| `data/offline_rl_dataset_real.json` | Network II | Real offline RL trajectories (partial) |

See `data/README.md` for full format specifications.

### Virtual trajectories (simulated)

Virtual datasets are generated using the provided Unity simulation scripts.
All collection code is in `dataset_collection/`.

**Network I:**
1. Attach `SoftRobotDataCollector.cs` in Unity and press Play.
2. Run:
```bash
python dataset_collection/collect_traj_data.py
```
Output: `coordinates_new.json`

**Network II:**
1. Attach `SoftRobotRLDataCollector.cs` in Unity and press Play.
2. Run:
```bash
python dataset_collection/collect_rl_data.py
```
Output: `coordinates_new.json`, `coordinates_new_comparison.json`

Convert raw outputs to the training format and place them at
`data/offline_rl_dataset_sim.json`. See `data/README.md` for the required schema.

### Collecting additional real trajectories

The provided real datasets cover one physical environment. To collect data
in additional environments, the following procedure is used.

**Hardware:** The robot tip is equipped with a 9-axis IMU and the base module
with an encoder; a Kalman filter fuses both signals to estimate real-time tip
pose.

**Vision-based ground-truth capture:**
1. Attach a coloured marker to the robot tip.
2. Record a top-down video during each growth episode using an overhead RGB camera.
3. Apply HSV colour thresholding to segment the tip marker per frame.
4. Convert pixel coordinates to world coordinates via a pre-calibrated
   homography matrix (checkerboard calibration).
5. Save the `[x, z]` position sequence as a `.mat` file and update
   `REAL_MAT_FILE` and `REAL_TRAJ_NAMES` in `configs.py`.

---

## Training

```bash
# Network I — trajectory optimisation
python -m network_i.train

# Network II — CQL offline RL
python -m network_ii.train
```

Key hyperparameters (see `configs.py` for the full list):

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
  title   = {Trajectory Optimization and Morphology Control for Soft Growing Robots
             via Sim \& Real Co-training},
  author  = {Your Authors},
  journal = {IEEE Transactions on Robotics},
  year    = {2025}
}
```

---

## License

MIT License
