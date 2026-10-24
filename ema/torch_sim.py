"""Differentiable closed-loop IDM simulation in PyTorch (for training networks).

Same rules as ema.simulate and ema.sim_calibrate: explicit Euler at the data rate,
v(k+1) = max(0, v + a dt), follower position x(k+1) = x + v dt, simulated spacing =
leader position - follower position (leader position from the record). Batched over
segments; every segment can have its own IDM parameters. Unlike the evaluation, the
simulation does not stop at a collision (the gap inside the IDM is clipped at min_gap);
collisions are flagged so that the loss can penalize them.
"""
import torch


def idm_accel(s, v, v_lead, p, min_gap=0.1):
    """IDM acceleration; p maps names to tensors broadcastable to s (or floats)."""
    s = torch.clamp(s, min=min_gap)
    closing = v * (v - v_lead) / (2.0 * torch.sqrt(p["a"] * p["b"]))
    s_star = p["s0"] + torch.clamp(v * p["T"] + closing, min=0.0)
    return p["a"] * (1.0 - (v / p["v0"]) ** p["delta"] - (s_star / s) ** 2)


def rollout(p, seg, dt, collision_gap=0.0, min_gap=0.1):
    """Simulate a batch of segments.

    seg: dict of tensors v_lead, lead_pos, s_obs (n x T), s0, v0, length (n).
    p: IDM parameters, tensors of shape (n,) or floats.
    Returns dict(s, v) of shape (n x T) and collided (n,) bool (only steps < length count).
    """
    v_lead, lead_pos = seg["v_lead"], seg["lead_pos"]
    n, T = v_lead.shape
    s = seg["s0"].clone()
    v = seg["v0"].clone()
    x = lead_pos[:, 0] - s
    s_hist, v_hist = [s], [v]
    collided = torch.zeros(n, dtype=torch.bool, device=s.device)
    for k in range(T - 1):
        a = idm_accel(s, v, v_lead[:, k], p, min_gap)
        x = x + v * dt
        v = torch.clamp(v + a * dt, min=0.0)
        s = lead_pos[:, k + 1] - x
        active = (k + 1) < seg["length"]
        collided = collided | (active & (s.detach() <= collision_gap))
        s_hist.append(s)
        v_hist.append(v)
    return dict(s=torch.stack(s_hist, dim=1), v=torch.stack(v_hist, dim=1), collided=collided)


def spacing_rmse(s_sim, seg, eps=1e-8):
    """Spacing RMSE per segment over its valid steps (length), shape (n,)."""
    T = s_sim.shape[1]
    mask = (torch.arange(T, device=s_sim.device)[None, :] < seg["length"][:, None]).to(s_sim.dtype)
    se = ((s_sim - seg["s_obs"]) ** 2) * mask
    return torch.sqrt(se.sum(dim=1) / seg["length"].to(s_sim.dtype) + eps)


def to_torch(seg_np, dtype=torch.float32, device="cpu"):
    """Convert a segment dict from ema.sim_calibrate (numpy) to tensors."""
    keys = ["v_lead", "lead_pos", "s_obs", "s0", "v0"]
    out = {k: torch.as_tensor(seg_np[k], dtype=dtype, device=device) for k in keys}
    out["length"] = torch.as_tensor(seg_np["length"], dtype=torch.long, device=device)
    return out
