"""M4 network (D017): a GRU context encoder and an IDM parameter head.

The encoder reads every context piece (10 Hz: gap, speed, relative speed, acceleration,
scaled), takes the final hidden state of each piece and averages the pieces of an
episode weighted by their length; a linear layer gives the latent z. The head maps z to
the free IDM parameters inside the bounds of config/models.yaml (log scale, sigmoid), so
every output is a valid IDM.
"""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ema import calibrate

FEATURES = ["s", "v_f", "dv", "a_fd"]


def piece_features(s, v_f, v_l, dt, scales):
    """(n, 4) float32 array of scaled features of one context piece; the acceleration is
    the forward difference of v_f (last sample repeats the previous value)."""
    s, v_f, v_l = (np.asarray(x, dtype=float) for x in (s, v_f, v_l))
    a = np.zeros_like(v_f)
    if len(v_f) > 1:
        a[:-1] = np.diff(v_f) / dt
        a[-1] = a[-2]
    return np.stack([s / scales["s"], v_f / scales["v_f"], (v_l - v_f) / scales["dv"],
                     a / scales["a_fd"]], axis=1).astype(np.float32)


def collate_contexts(contexts, device="cpu"):
    """contexts: list (episodes) of lists of feature arrays. Returns padded x (P, L, 4),
    lengths (P,), episode index (P,) as tensors (x and episode on `device`)."""
    pieces = [(i, f) for i, ctx in enumerate(contexts) for f in ctx]
    lengths = torch.tensor([len(f) for _, f in pieces], dtype=torch.long)
    x = torch.zeros(len(pieces), int(lengths.max()), len(FEATURES))
    for j, (_, f) in enumerate(pieces):
        x[j, :len(f)] = torch.from_numpy(f)
    episode = torch.tensor([i for i, _ in pieces], dtype=torch.long)
    return x.to(device), lengths, episode.to(device)


class AdaptNet(nn.Module):
    """Context -> latent z -> IDM parameters (dict of tensors of shape (n_episodes,))."""

    def __init__(self, cfg_net, param_spec):
        super().__init__()
        self.free, bounds, self.fixed = calibrate.split_spec(param_spec)
        self.register_buffer("log_lo", torch.log(torch.tensor([b[0] for b in bounds])))
        self.register_buffer("log_hi", torch.log(torch.tensor([b[1] for b in bounds])))
        h, zd, hh = int(cfg_net["hidden"]), int(cfg_net["z_dim"]), int(cfg_net["head_hidden"])
        self.gru = nn.GRU(len(FEATURES), h, batch_first=True)
        self.to_z = nn.Linear(h, zd)
        self.head = nn.Sequential(nn.Linear(zd, hh), nn.Tanh(), nn.Linear(hh, len(self.free)))

    def encode(self, x, lengths, episode, n_episodes):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = h[-1]
        w = lengths.to(h.dtype)
        w = w.to(h.device)
        num = torch.zeros(n_episodes, h.shape[1], dtype=h.dtype, device=h.device).index_add(
            0, episode, h * w[:, None])
        den = torch.zeros(n_episodes, dtype=h.dtype, device=h.device).index_add(0, episode, w)
        return self.to_z(num / den[:, None])

    def params_from_z(self, z):
        u = torch.sigmoid(self.head(z))
        logp = self.log_lo + u * (self.log_hi - self.log_lo)
        p = {k: torch.exp(logp[:, i]) for i, k in enumerate(self.free)}
        for k, v in self.fixed.items():
            p[k] = torch.full_like(logp[:, 0], float(v))
        return p

    def forward(self, x, lengths, episode, n_episodes):
        return self.params_from_z(self.encode(x, lengths, episode, n_episodes))
