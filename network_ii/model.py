"""
network_ii/model.py  —  CQL offline RL networks (Network II).

Components
----------
StableQNetwork   : twin Q-network with residual connections
AdaptivePolicy   : bimodal policy (zero-action gate + non-zero action head)
DistanceAdaptiveCQL : CQL agent that combines both, with adaptive scheduling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs import (
    RL_STATE_DIM, RL_ACTION_DIM, RL_ACTION_RANGE,
    RL_LR, RL_GAMMA, RL_TAU, RL_ALPHA, RL_CQL_TEMP,
    RL_EPOCHS, RL_WARMUP_EPOCHS, RL_STABLE_EPOCHS, RL_LR_MIN_RATIO,
    RL_NEAR_DIST, RL_FAR_DIST, RL_ZERO_ACTION_THR,
)


# ── Learning-rate scheduler ───────────────────────────────────────────────────

class LRScheduler:
    """Warm-up → stable → cosine-decay schedule for Q and policy networks."""

    def __init__(self, q_lr, policy_lr, total_epochs=RL_EPOCHS,
                 warmup=RL_WARMUP_EPOCHS, stable=RL_STABLE_EPOCHS,
                 min_ratio=RL_LR_MIN_RATIO):
        self.q_lr0, self.p_lr0 = q_lr, policy_lr
        self.total, self.warmup, self.stable = total_epochs, warmup, stable
        self.min_ratio = min_ratio
        self.epoch = 0

    def _lr(self, base):
        e = self.epoch
        if e < self.warmup:
            m = self.min_ratio + (1 - self.min_ratio) * e / self.warmup
        elif e < self.stable:
            m = 1.0
        else:
            prog = (e - self.stable) / max(self.total - self.stable, 1)
            m = self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + np.cos(np.pi * prog))
        return base * m

    def get_q_lr(self):     return self._lr(self.q_lr0)
    def get_policy_lr(self): return self._lr(self.p_lr0)
    def step(self):          self.epoch += 1


# ── Constraint scheduler ──────────────────────────────────────────────────────

class ConstraintScheduler:
    """Gradually increases constraint strength and adapts CQL α."""

    def __init__(self, total_epochs=RL_EPOCHS):
        self.total = total_epochs
        self.epoch = 0

    def step(self):  self.epoch += 1

    def _prog(self): return min(self.epoch / self.total, 1.0)

    def constraint_strength(self): return 0.5 + 0.5 * self._prog() ** 0.8
    def near_threshold(self):      return 0.12 - 0.04 * self._prog() ** 0.7
    def far_threshold(self):       return 0.25 + 0.05 * self._prog()
    def cql_alpha(self):           return 0.008 + 0.017 * self._prog() ** 1.5


# ── Q-network ─────────────────────────────────────────────────────────────────

class StableQNetwork(nn.Module):
    """Twin-Q network with residual connections and layer normalisation."""

    def __init__(self, state_dim=RL_STATE_DIM, action_dim=RL_ACTION_DIM, hidden=256):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ELU(), nn.Dropout(0.05))
        self.action_enc = nn.Sequential(
            nn.Linear(action_dim, hidden // 4), nn.LayerNorm(hidden // 4), nn.ELU())
        self.fusion = nn.Sequential(
            nn.Linear(hidden + hidden // 4, hidden), nn.LayerNorm(hidden), nn.ELU(), nn.Dropout(0.05))
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ELU(), nn.Linear(hidden // 2, 1))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)
        last = list(self.out.modules())[-1]
        nn.init.uniform_(last.weight, -1e-5, 1e-5); nn.init.zeros_(last.bias)

    def forward(self, state, action):
        s = self.state_enc(state)
        x = torch.cat([s, self.action_enc(action)], dim=-1)
        return self.out(self.fusion(x) + s)


# ── Policy network ────────────────────────────────────────────────────────────

class AdaptivePolicy(nn.Module):
    """
    Bimodal policy network (Eq. 29 in the paper).

    Two output branches:
      zero_action_head   → P(zero action)  via sigmoid
      nonzero_action_head → continuous non-zero action via tanh × action_range

    The gating variable z ~ Bernoulli(P_0) selects between them,
    yielding sparse control: action = (1 − z) × a_nz.
    """

    def __init__(self, state_dim=RL_STATE_DIM, action_dim=RL_ACTION_DIM,
                 hidden=256, action_range=RL_ACTION_RANGE):
        super().__init__()
        self.action_range = action_range

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.ELU(), nn.Dropout(0.1))

        self.zero_head    = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ELU(), nn.Linear(hidden // 2, 1))
        self.nonzero_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ELU(), nn.Linear(hidden // 2, action_dim))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.constant_(list(self.zero_head.modules())[-1].bias, 3.5)   # favour zero initially
        nn.init.uniform_(list(self.nonzero_head.modules())[-1].weight, -0.01, 0.01)

    def forward(self, state, deterministic=False, return_prob=False):
        feat = self.encoder(state)
        zero_prob  = torch.sigmoid(self.zero_head(feat))
        nz_action  = torch.tanh(self.nonzero_head(feat)) * self.action_range

        use_zero = (zero_prob > 0.5).float() if deterministic else torch.bernoulli(zero_prob)
        action   = (1 - use_zero) * nz_action

        return (action, zero_prob) if return_prob else action


# ── CQL agent ─────────────────────────────────────────────────────────────────

def _logsumexp(x, dim, temp=1.0):
    mx = x.max(dim=dim, keepdim=True)[0]
    x  = torch.clamp((x - mx) / temp, -20, 20)
    return mx + temp * torch.log(torch.exp(x).sum(dim=dim, keepdim=True) + 1e-8)


def _soft_clamp(x, lo, hi, sharpness=0.5):
    return lo + (hi - lo) * torch.sigmoid((x - (lo + hi) / 2) / sharpness)


def _target_zero_prob(dist, near=RL_NEAR_DIST, far=RL_FAR_DIST, sharpness=50.0):
    """Distance-adaptive target probability for the zero-action gate."""
    centre = (near + far) / 2
    scale  = sharpness / (far - near)
    return 0.001 + 0.998 * torch.sigmoid(-(dist - centre) * scale)


class DistanceAdaptiveCQL:
    """
    Conservative Q-Learning agent with distance-adaptive reward shaping.
    Trains from a mixed offline dataset of sim and real trajectories.
    """

    def __init__(self, state_dim=RL_STATE_DIM, action_dim=RL_ACTION_DIM,
                 device="cpu"):
        self.device       = device
        self.gamma        = RL_GAMMA
        self.tau          = RL_TAU
        self.alpha        = RL_ALPHA
        self.cql_temp     = RL_CQL_TEMP
        self.action_range = RL_ACTION_RANGE
        self.action_dim   = action_dim
        self.state_dim    = state_dim

        self.constraint_sched = ConstraintScheduler(RL_EPOCHS)
        self.lr_sched         = LRScheduler(RL_LR * 0.3, RL_LR)

        self.q1 = StableQNetwork(state_dim, action_dim).to(device)
        self.q2 = StableQNetwork(state_dim, action_dim).to(device)
        self.q1_target = StableQNetwork(state_dim, action_dim).to(device)
        self.q2_target = StableQNetwork(state_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy = AdaptivePolicy(state_dim, action_dim).to(device)

        self.q1_opt = optim.AdamW(self.q1.parameters(), lr=RL_LR * 0.3, weight_decay=1e-6)
        self.q2_opt = optim.AdamW(self.q2.parameters(), lr=RL_LR * 0.3, weight_decay=1e-6)
        self.pi_opt = optim.AdamW(self.policy.parameters(), lr=RL_LR,    weight_decay=1e-6)

    def select_action(self, state, deterministic=False):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.policy(s, deterministic=deterministic)
        return a.cpu().numpy()[0]

    def train_step(self, batch):
        (states, actions, rewards, next_states, dones,
         distances, is_last, steps_nz, data_src, is_nz, is_succ) = [
            b.float().to(self.device) for b in batch]

        B = states.shape[0]
        rewards    = rewards.view(B, 1)
        dones      = dones.view(B, 1)
        distances  = distances.view(B, 1)
        is_last    = is_last.view(B, 1)
        data_src   = data_src.view(B, 1)
        is_nz      = is_nz.view(B, 1)
        is_succ    = is_succ.view(B, 1)

        near_thr = self.constraint_sched.near_threshold()
        far_thr  = self.constraint_sched.far_threshold()
        self.alpha = self.constraint_sched.cql_alpha()

        # ── Target Q ──────────────────────────────────────────────────────────
        with torch.no_grad():
            na = self.policy(next_states)
            na = (na + torch.randn_like(na) * 0.1).clamp(-self.action_range, self.action_range)
            q_next = torch.min(self.q1_target(next_states, na),
                               self.q2_target(next_states, na))
            q_next = _soft_clamp(q_next, -15, 15)
            q_tgt  = _soft_clamp(rewards + (1 - dones) * self.gamma * q_next, -15, 15)

        q1_c = self.q1(states, actions)
        q2_c = self.q2(states, actions)

        # Bellman loss
        q1_bell = nn.SmoothL1Loss()(q1_c, q_tgt)
        q2_bell = nn.SmoothL1Loss()(q2_c, q_tgt)

        # CQL penalty
        N = 10
        rand_a = torch.FloatTensor(B, N, self.action_dim).uniform_(
            -self.action_range, self.action_range).to(self.device)
        rep_s  = states.unsqueeze(1).repeat(1, N, 1).view(B * N, self.state_dim)
        q1_r   = self.q1(rep_s, rand_a.view(B * N, self.action_dim)).view(B, N)
        q2_r   = self.q2(rep_s, rand_a.view(B * N, self.action_dim)).view(B, N)
        cql1   = nn.Softplus()(_logsumexp(q1_r, dim=1).mean() - q1_c.mean()) * 0.05
        cql2   = nn.Softplus()(_logsumexp(q2_r, dim=1).mean() - q2_c.mean()) * 0.05

        # Update Q networks
        for opt, loss in [(self.q1_opt, q1_bell + self.alpha * cql1),
                           (self.q2_opt, q2_bell + self.alpha * cql2)]:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.q1.parameters(), 0.3)
            opt.step()

        # Update learning rates
        q_lr = self.lr_sched.get_q_lr(); p_lr = self.lr_sched.get_policy_lr()
        for opt in (self.q1_opt, self.q2_opt):
            for pg in opt.param_groups: pg["lr"] = q_lr
        for pg in self.pi_opt.param_groups: pg["lr"] = p_lr

        # ── Policy update ──────────────────────────────────────────────────────
        pi_a, zero_prob = self.policy(states, return_prob=True)
        q_pi   = torch.min(self.q1(states, pi_a), self.q2(states, pi_a))
        dist   = torch.norm(states[:, 1:3] - states[:, 4:6], dim=-1, keepdim=True)
        tgt_zp = _target_zero_prob(dist, near_thr, far_thr)

        bc_loss = ((pi_a - actions) ** 2 * data_src).sum() / (data_src.sum() + 1e-8)
        pi_loss = (-3.0 * q_pi.mean() +
                    4.0 * nn.BCELoss()(zero_prob, tgt_zp) +
                    8.0 * bc_loss)

        self.pi_opt.zero_grad(); pi_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.3)
        self.pi_opt.step()

        # Soft update targets
        for src, tgt in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
            for tp, p in zip(tgt.parameters(), src.parameters()):
                tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)

        # Training success rate (non-zero actions that reduce distance)
        nz_mask = is_nz.squeeze() > 0.5
        sr = (is_succ[nz_mask].sum() / nz_mask.sum()).item() if nz_mask.sum() > 0 else 0.0

        return {"policy_loss": pi_loss.item(),
                "q_loss": ((q1_bell + q2_bell) / 2).item(),
                "avg_reward": rewards.mean().item(),
                "train_success_rate": sr}

    def step_schedulers(self):
        self.constraint_sched.step()
        self.lr_sched.step()

    def save(self, path):
        torch.save({"q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                    "policy": self.policy.state_dict()}, path)
        print(f"[Net-II] Checkpoint saved → {path}")

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.q1.load_state_dict(ck["q1"]); self.q2.load_state_dict(ck["q2"])
        self.policy.load_state_dict(ck["policy"])
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        print(f"[Net-II] Weights loaded from {path}")
