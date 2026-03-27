"""
configs.py  —  All hyperparameters for the Sim & Real Co-Training pipeline.

Network I : GAT–Transformer–MLP trajectory optimisation (imitation learning)
Network II: CQL offline RL morphology control and error correction
"""

import torch

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Workspace bounds (metres) ────────────────────────────────────────────────
X_MIN, X_MAX = -2.0, 2.0
Z_MIN, Z_MAX = -1.0, 6.0

# ════════════════════════════════════════════════════════════════════════════
#  Network I  —  Trajectory Optimisation
# ════════════════════════════════════════════════════════════════════════════

# ── Data files (update paths to match your environment) ──────────────────────
REAL_MAT_FILE     = "data/real_trajectories.mat"
VIRTUAL_JSON_FILE = "data/virtual_trajectories.json"
REAL_TRAJ_NAMES   = ["traji_7161", "traji_7164", "traji_7166"]

# ── Sim / real ratio ─────────────────────────────────────────────────────────
#   Best training loss achieved at ~4 % real data (see paper Fig. 9b).
GLOBAL_REAL_DATA_RATIO = 0.04

# ── Virtual trajectory layout inside the JSON ────────────────────────────────
NUM_REAL_TRAJS        = 3
NUM_VIRTUAL_PER_REAL  = 100
MAPPED_VIRTUAL_COUNT  = 300    # = NUM_REAL_TRAJS × NUM_VIRTUAL_PER_REAL
RANDOM_VIRTUAL_COUNT  = 500

SAMPLES_PER_GROUP  = 10
SAMPLE_OFFSET      = 50
RANDOM_SAMPLES     = 50

# ── Real-data augmentation ───────────────────────────────────────────────────
REAL_AUG_FACTOR      = 2
NOISE_LEVELS         = [0.008, 0.015, 0.025]
NUM_SEGMENTS_REAL    = 8
NUM_SEGMENTS_VIRTUAL = 3

# ── Weighted sampler ─────────────────────────────────────────────────────────
VIRTUAL_SIMILARITY_WEIGHT = 0.8
VIRTUAL_DIVERSITY_WEIGHT  = 0.2
RANDOM_DIVERSITY_WEIGHT   = 0.3

# ── Trajectory resampling ────────────────────────────────────────────────────
REAL_TIMESTEPS    = dict(min=80,  max=200, base=120, density=0.4)
VIRTUAL_TIMESTEPS = dict(min=30,  max=80,  base=50,  density=0.2)

# ── Graph construction ───────────────────────────────────────────────────────
SPATIAL_EDGE_THRESHOLD = 0.5

# ── Network I architecture ───────────────────────────────────────────────────
NODE_FEAT_DIM          = 36
GNN_DIM                = 128
TRANS_DIM              = 128
NUM_TRANS_LAYERS       = 4
SOFT_ROBOT_FLEXIBILITY = 0.8

# ── Loss weights ─────────────────────────────────────────────────────────────
NET1_LOSS_WEIGHTS = dict(position=1.0, shape=1.5, obstacle=0.5,
                          endpoint=1.5, smoothness=0.1)
REAL_DOMAIN_WEIGHT    = 1.8
VIRTUAL_DOMAIN_WEIGHT = 0.6
SOFT_ROBOT_PENETRATION_TOLERANCE = 0.05

# ── Training ─────────────────────────────────────────────────────────────────
NET1_EPOCHS      = 400
NET1_LR          = 5e-4
NET1_WEIGHT_DECAY = 1e-4
NET1_GRAD_CLIP   = 0.5
NET1_LR_MIN      = 1e-6
NET1_PLOT_INTERVAL = 10
NET1_CHECKPOINT  = "checkpoints/network_i.pt"

# ── Inference / post-processing ──────────────────────────────────────────────
INFER_TIMESTEPS     = 50
RESAMPLE_SPACING    = 0.05
BACKTRACK_ANGLE_THR = 120.0

# ════════════════════════════════════════════════════════════════════════════
#  Network II  —  Offline RL (CQL) Morphology Control
# ════════════════════════════════════════════════════════════════════════════

# ── Data files ───────────────────────────────────────────────────────────────
RL_SIM_JSON  = "data/offline_rl_dataset_sim.json"
RL_REAL_JSON = "data/offline_rl_dataset_real.json"

# ── State / action space ─────────────────────────────────────────────────────
#   State (6-D): [robot_length, end_x, end_z, growth_angle, goal_x, goal_z]
#   Action (1-D): tendon displacement (degrees, range ±30)
RL_STATE_DIM    = 6
RL_ACTION_DIM   = 1
RL_ACTION_RANGE = 30.0

# ── Sim / real mixing ────────────────────────────────────────────────────────
#   Best performance at ~85 % simulated data (see paper Fig. 11d).
RL_SIM_RATIO = 0.85

# ── CQL hyperparameters ──────────────────────────────────────────────────────
RL_LR         = 2e-5
RL_GAMMA      = 0.99
RL_TAU        = 0.001
RL_ALPHA      = 0.015     # CQL conservatism coefficient
RL_CQL_TEMP   = 0.5
RL_BATCH_SIZE = 128

# ── Training ─────────────────────────────────────────────────────────────────
RL_EPOCHS          = 400
RL_VAL_INTERVAL    = 10
RL_CHECKPOINT      = "checkpoints/network_ii.pt"
RL_CHECKPOINT_BEST = "checkpoints/network_ii_best.pt"

# ── LR scheduler ─────────────────────────────────────────────────────────────
RL_WARMUP_EPOCHS  = 30
RL_STABLE_EPOCHS  = 200
RL_LR_MIN_RATIO   = 0.05

# ── Action thresholds ────────────────────────────────────────────────────────
RL_ZERO_ACTION_THR    = 1.0   # |action| < thr → treated as zero
RL_NONZERO_ACTION_THR = 1.5   # |action| ≥ thr → treated as non-zero

# ── Distance thresholds ──────────────────────────────────────────────────────
RL_NEAR_DIST = 0.10    # metres — "near" regime boundary
RL_FAR_DIST  = 0.25    # metres — "far"  regime boundary

# ── Error-correction trigger ─────────────────────────────────────────────────
RL_CORRECTION_THRESHOLD = 0.02   # metres — activate tendon if |Δ| > this
