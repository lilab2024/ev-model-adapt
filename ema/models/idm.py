"""Intelligent Driver Model (IDM).

a = a_max * [1 - (v / v0)^delta - (s_star / s)^2],
s_star = s0 + max(0, v*T + v*(v - v_lead) / (2*sqrt(a_max*b))).
"""
import math

import numpy as np

PARAM_NAMES = ["v0", "T", "s0", "a", "b", "delta"]


def accel(s, v, v_lead, p, min_gap=0.1):
    """Vectorized IDM acceleration."""
    s = np.maximum(np.asarray(s, dtype=float), min_gap)
    v = np.asarray(v, dtype=float)
    v_lead = np.asarray(v_lead, dtype=float)
    s_star = p["s0"] + np.maximum(
        0.0, v * p["T"] + v * (v - v_lead) / (2.0 * np.sqrt(p["a"] * p["b"])))
    return p["a"] * (1.0 - (v / p["v0"]) ** p["delta"] - (s_star / s) ** 2)


class IDMController:
    """IDM as a closed-loop controller (scalar arithmetic for speed)."""

    def __init__(self, p, min_gap=0.1):
        self.p = {k: float(p[k]) for k in PARAM_NAMES}
        self.min_gap = float(min_gap)
        self._sqrt_ab = 2.0 * math.sqrt(self.p["a"] * self.p["b"])

    def reset(self, s0, v0, v_lead0):
        pass

    def accel(self, s, v, v_lead):
        p = self.p
        s = max(s, self.min_gap)
        s_star = p["s0"] + max(0.0, v * p["T"] + v * (v - v_lead) / self._sqrt_ab)
        return p["a"] * (1.0 - (v / p["v0"]) ** p["delta"] - (s_star / s) ** 2)
