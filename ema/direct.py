"""M4-C (D033): a context encoder (the M4 GRU encoder) gives z; a policy network gives the
acceleration at every step of the closed loop, without the IDM.

Policy features (scaled): gap, follower speed, speed difference (leader - follower),
leader acceleration ((v_lead(k) - v_lead(k-1)) / dt, 0 at the first step) and the
follower's acceleration of the previous step (0 at the first step), concatenated with z;
output lo + (hi - lo) * sigmoid(.), i.e. inside accel_limits. The same rules are used in
the differentiable rollout (training, PyTorch) and in the controller (evaluation, NumPy),
with the Euler and leader-position simulation of ema.simulate. DirectNet has its own
episode_loss, which ema.train_m4 uses, so pretraining and fine-tuning reuse that code.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ema import config, nets

FEATURES = ("gap", "v", "dv", "a_lead", "a_prev")


def scale_vector(cfg_policy):
    return np.array([float(cfg_policy["scales"][k]) for k in FEATURES])


class PolicyNet(nn.Module):
    def __init__(self, z_dim, hidden=(64, 64), limits=(-8.0, 4.0)):
        super().__init__()
        layers, n_in = [], len(FEATURES) + int(z_dim)
        for h in hidden:
            layers += [nn.Linear(n_in, int(h)), nn.Tanh()]
            n_in = int(h)
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(n_in, 1)
        self.lo, self.hi = float(limits[0]), float(limits[1])

    def forward(self, feat, z):
        u = self.out(self.body(torch.cat([feat, z], dim=1))).squeeze(-1)
        return self.lo + (self.hi - self.lo) * torch.sigmoid(u)


def rollout(policy, z, seg, dt, feat_scales):
    """Differentiable closed loop of the policy on a batch of segments (tensors v_lead,
    lead_pos, s_obs (n x T), s0, v0, length (n)); z (n x z_dim). Returns (spacing RMSE
    per segment over its valid steps, simulated gaps n x T)."""
    v_lead, lead_pos, length = seg["v_lead"], seg["lead_pos"], seg["length"]
    n, T = v_lead.shape
    sc = torch.as_tensor(feat_scales, dtype=v_lead.dtype, device=v_lead.device)
    s, v = seg["s0"].clone(), seg["v0"].clone()
    x = lead_pos[:, 0] - s
    a_prev, se = torch.zeros_like(s), torch.zeros_like(s)
    s_hist = [s]
    for k in range(T - 1):
        a_lead = torch.zeros_like(s) if k == 0 else (v_lead[:, k] - v_lead[:, k - 1]) / dt
        feat = torch.stack([s, v, v_lead[:, k] - v, a_lead, a_prev], dim=1) / sc
        a = policy(feat, z)
        x = x + v * dt
        v = torch.clamp(v + a * dt, min=0.0)
        s = lead_pos[:, k + 1] - x
        active = ((k + 1) < length).to(s.dtype)
        se = se + active * (s - seg["s_obs"][:, k + 1]) ** 2
        a_prev = a
        s_hist.append(s)
    return torch.sqrt(se / length.to(s.dtype) + 1e-8), torch.stack(s_hist, dim=1)


class DirectNet(nn.Module):
    """Encoder (nets.AdaptNet; its parameter head is not used) + PolicyNet."""

    def __init__(self, cfg_net, param_spec, cfg_policy):
        super().__init__()
        self.enc = nets.AdaptNet(cfg_net, param_spec)
        self.policy = PolicyNet(cfg_net["z_dim"], cfg_policy["hidden"], cfg_policy["accel_limits"])
        self.register_buffer("feat_scales", torch.tensor(scale_vector(cfg_policy), dtype=torch.float32))

    def encode(self, x, lengths, episode, n_episodes):
        return self.enc.encode(x, lengths, episode, n_episodes)

    def episode_rmse(self, episodes, dt, scales, min_gap=0.1):
        """Per-episode mean closed-loop spacing RMSE of the targets (min_gap unused)."""
        from ema import train_m4
        device = self.feat_scales.device
        x, lengths, ep = train_m4.context_tensors(episodes, dt, scales, device)
        z = self.encode(x, lengths, ep, len(episodes))
        seg, tgt_ep = train_m4.pack_targets(episodes, dt, dtype=x.dtype, device=device)
        rmse, _ = rollout(self.policy, z[tgt_ep], seg, dt, self.feat_scales)
        per_ep = torch.zeros(len(episodes), dtype=rmse.dtype, device=device).index_add(0, tgt_ep, rmse)
        return per_ep / torch.bincount(tgt_ep, minlength=len(episodes)).clamp(min=1)

    def episode_loss(self, episodes, dt, scales, min_gap=0.1):
        return self.episode_rmse(episodes, dt, scales, min_gap).mean()


def make_pretraining_net(cfg_net, param_spec, cfg_direct):
    """DirectNet whose encoder starts from cfg_direct['init_encoder'] (md5 checked) and whose
    policy starts from random weights (seeded by the caller)."""
    net = DirectNet(cfg_net, param_spec, cfg_direct["policy"])
    path = Path(cfg_direct["init_encoder"]["weights"])
    if not path.is_absolute():
        path = config.REPO_ROOT / path
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != cfg_direct["init_encoder"]["md5"]:
        raise ValueError(f"{path}: md5 {md5} does not match config/direct.yaml")
    net.enc.load_state_dict(torch.load(path, map_location="cpu"))
    return net


def encode_context(net, pieces, dt, scales):
    """z (float64 array) of one context given as a list of DataFrames or dicts."""
    feats = [nets.piece_features(np.asarray(p["s"]), np.asarray(p["v_f"]), np.asarray(p["v_l"]),
                                 dt, scales) for p in pieces]
    device = net.feat_scales.device
    x, lengths, ep = nets.collate_contexts([feats], device)
    net.eval()
    with torch.no_grad():
        return net.encode(x, lengths, ep, 1)[0].cpu().double().numpy()


def numpy_weights(policy):
    lin = [m for m in policy.body if isinstance(m, nn.Linear)] + [policy.out]
    return [(m.weight.detach().cpu().double().numpy(), m.bias.detach().cpu().double().numpy())
            for m in lin]


class DirectController:
    """The policy as a closed-loop controller (NumPy, float64; interface of IDMController)."""

    def __init__(self, weights, z, limits, scales, dt):
        self.weights, self.z = weights, np.asarray(z, dtype=float)
        self.lo, self.hi = float(limits[0]), float(limits[1])
        self.scales, self.dt = np.asarray(scales, dtype=float), float(dt)
        self.prev_v_lead, self.a_prev = None, 0.0

    def reset(self, s0, v0, v_lead0):
        self.prev_v_lead, self.a_prev = None, 0.0

    def accel(self, s, v, v_lead):
        a_lead = 0.0 if self.prev_v_lead is None else (v_lead - self.prev_v_lead) / self.dt
        h = np.concatenate([np.array([s, v, v_lead - v, a_lead, self.a_prev]) / self.scales, self.z])
        for W, b in self.weights[:-1]:
            h = np.tanh(W @ h + b)
        W, b = self.weights[-1]
        u = float((W @ h + b)[0])
        a = self.lo + (self.hi - self.lo) / (1.0 + np.exp(-u))
        self.prev_v_lead, self.a_prev = v_lead, a
        return a
