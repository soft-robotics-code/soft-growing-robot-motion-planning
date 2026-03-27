"""
network_i/losses.py  —  Loss functions for Network I (trajectory optimisation).

Total loss (Eq. 21):
    L = w_pos·L_pos + w_shape·L_shape + w_obs·L_obs + w_ep·L_endpoint + w_sm·L_smooth

Obstacle penalty uses the four-segment design (Eq. 22–23).
"""

import numpy as np
import torch
import torch.nn.functional as F
from configs import NET1_LOSS_WEIGHTS, SOFT_ROBOT_PENETRATION_TOLERANCE


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _cuboid_polygon(pos, rotation_y, scale):
    pos, scale = np.asarray(pos, float), np.asarray(scale, float)
    cx, cz = (pos[0], pos[2]) if pos.size == 3 else (pos[0], pos[1])
    sx, sz = (scale[0], scale[2]) if scale.size == 3 else (scale[0], scale[1])
    theta  = -np.deg2rad(rotation_y)
    dx, dz = sx / 2, sz / 2
    local  = np.array([[-dx, -dz], [dx, -dz], [dx, dz], [-dx, dz]])
    rot    = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
    return local @ rot.T + np.array([cx, cz])


def _in_poly(p, poly):
    x, y, n, inside = p[0], p[1], len(poly), False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xi = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xi:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _seg_dist(p, a, b):
    ab = b - a; ap = p - a
    t  = np.clip(ap @ ab / (ab @ ab + 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _poly_dist(p, poly):
    n = len(poly)
    return min(_seg_dist(p, poly[i], poly[(i + 1) % n]) for i in range(n))


# ── Obstacle collision loss ───────────────────────────────────────────────────

def obstacle_loss_np(traj_np, cubes, tau=SOFT_ROBOT_PENETRATION_TOLERANCE):
    """Four-segment penalty (Eq. 22–23). Returns float."""
    polys = [_cuboid_polygon(p, r, s) for p, r, s in (cubes or [])]
    total = 0.0
    for pt in traj_np:
        for poly in polys:
            if _in_poly(pt, poly):
                d = _poly_dist(pt, poly)
                total += (d - tau) ** 2 if d > tau else 0.1 * d
            else:
                d = _poly_dist(pt, poly)
                if d < 2 * tau:
                    total += 0.01 * (2 * tau - d)
    return total


# ── Differentiable loss components ───────────────────────────────────────────

def shape_loss(pred, target):
    if len(pred) < 2: return pred.new_zeros(())
    pd = (pred[1:] - pred[:-1]); pd = pd / (pd.norm(dim=1, keepdim=True) + 1e-9)
    td = (target[1:] - target[:-1]); td = td / (td.norm(dim=1, keepdim=True) + 1e-9)
    return (1.0 - (pd * td).sum(dim=1)).mean()


def endpoint_loss(pred, start_t, goal_t):
    T = pred.shape[0]
    s_loss = F.mse_loss(pred[0], start_t)
    g_loss = F.mse_loss(pred[-1], goal_t)
    prog   = pred.new_zeros(())
    if T > 5:
        region = max(1, int(0.8 * T))
        pts = [((1.0 + (i - region) / max(T - 1 - region, 1)) * F.mse_loss(pred[i], goal_t))
               for i in range(region, T - 1)]
        if pts: prog = torch.stack(pts).mean()
    return s_loss + g_loss + 0.5 * prog


def smoothness_loss(pred):
    if len(pred) < 3: return pred.new_zeros(())
    return (pred[2:] - 2 * pred[1:-1] + pred[:-2]).norm(dim=1).mean()


# ── Combined loss ─────────────────────────────────────────────────────────────

def total_loss(pred, target, cubes, start_np, goal_np):
    w      = NET1_LOSS_WEIGHTS
    device = pred.device
    pos_l  = F.mse_loss(pred, target)
    shp_l  = shape_loss(pred, target)
    obs_l  = torch.tensor(obstacle_loss_np(pred.detach().cpu().numpy(), cubes),
                           device=device, dtype=torch.float32)
    smt_l  = smoothness_loss(pred)
    ep_l   = endpoint_loss(pred,
                            torch.tensor(start_np[:2], device=device, dtype=torch.float32),
                            torch.tensor(goal_np[:2],  device=device, dtype=torch.float32))
    loss = (w["position"] * pos_l + w["shape"] * shp_l + w["obstacle"] * obs_l +
            w["endpoint"] * ep_l + w["smoothness"] * smt_l)
    return loss, pos_l, shp_l, obs_l
