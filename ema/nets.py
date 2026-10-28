"""M4 network (D017): a GRU (or, D032, Transformer) context encoder and an IDM parameter head.

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


def positional_encoding(length, d_model, device=None, dtype=torch.float32):
    """Sinusoidal position encoding (length, d_model) of the samples of a piece."""
    pos = torch.arange(length, device=device, dtype=dtype)[:, None]
    i = torch.arange(0, d_model, 2, device=device, dtype=dtype)
    angle = pos / torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), i / d_model)
    pe = torch.zeros(length, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle[:, : d_model // 2])
    return pe


class AdaptNet(nn.Module):
    """Context -> latent z -> IDM parameters (dict of tensors of shape (n_episodes,)).

    encoder "gru" (default): final hidden state of each piece. encoder "transformer"
    (D032): linear input layer, sinusoidal positions, a Transformer encoder (no dropout)
    and the mean over the valid samples of each piece (padding is masked). Either way the
    pieces of an episode are averaged weighted by their length."""

    def __init__(self, cfg_net, param_spec):
        super().__init__()
        self.free, bounds, self.fixed = calibrate.split_spec(param_spec)
        self.register_buffer("log_lo", torch.log(torch.tensor([b[0] for b in bounds])))
        self.register_buffer("log_hi", torch.log(torch.tensor([b[1] for b in bounds])))
        zd, hh = int(cfg_net["z_dim"]), int(cfg_net["head_hidden"])
        self.encoder = cfg_net.get("encoder", "gru")
        if self.encoder == "transformer":
            d = int(cfg_net["d_model"])
            self.inp = nn.Linear(len(FEATURES), d)
            layer = nn.TransformerEncoderLayer(d, int(cfg_net["heads"]), int(cfg_net["ff"]),
                                               dropout=0.0, batch_first=True)
            self.tf = nn.TransformerEncoder(layer, int(cfg_net["layers"]), enable_nested_tensor=False)
            self.to_z = nn.Linear(d, zd)
        elif self.encoder == "gru":
            h = int(cfg_net["hidden"])
            self.gru = nn.GRU(len(FEATURES), h, batch_first=True)
            self.to_z = nn.Linear(h, zd)
        else:
            raise ValueError(f"unknown encoder {self.encoder}")
        self.head = nn.Sequential(nn.Linear(zd, hh), nn.Tanh(), nn.Linear(hh, len(self.free)))

    def _piece_states(self, x, lengths):
        """One vector per piece."""
        if self.encoder == "transformer":
            L = x.shape[1]
            valid = torch.arange(L, device=x.device)[None, :] < lengths.to(x.device)[:, None]
            hs = self.tf(self.inp(x) + positional_encoding(L, self.inp.out_features, x.device,
                                                           x.dtype)[None],
                         src_key_padding_mask=~valid)
            hs = hs * valid[:, :, None].to(hs.dtype)
            return hs.sum(dim=1) / valid.sum(dim=1, keepdim=True).to(hs.dtype)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]

    def encode(self, x, lengths, episode, n_episodes):
        h = self._piece_states(x, lengths)
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
