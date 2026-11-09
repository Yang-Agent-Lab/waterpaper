import copy
import torch
from torch import nn
from multiwindow_peak_v1.model import blocks, maxima, nll, crps


class RainPeak(nn.Module):
    """Exact S2 plus one zero-initialized six-hour rainfall residual."""
    def __init__(self, base, mode):
        super().__init__()
        if base.arm != 'S2' or mode not in ('history', 'oracle'):
            raise ValueError('S2 / declared rainfall mode required')
        self.base = copy.deepcopy(base)
        self.rain = nn.Linear(6, 64)
        nn.init.zeros_(self.rain.weight); nn.init.zeros_(self.rain.bias)
        self.mode = mode

    def encode(self, x, f, future):
        p = x[:, -6:, 0] if self.mode == 'history' else future
        return self.base.encode(x, f) + self.rain(p)

    def loss(self, x, f, y, future, generator):
        h = self.encode(x, f, future)
        mu, sigma = self.base.distribution(h, blocks(y))
        ll = nll(blocks(y), mu, sigma).mean()
        c = crps(self.base.sample(h, 16, generator), maxima(y, True)).mean()
        return ll + c, {'nll': float(ll.detach()), 'crps': float(c.detach()),
                         'sigma_mean': float(sigma.mean().detach())}

    def sample(self, x, f, future, n, generator):
        return self.base.sample(self.encode(x, f, future), n, generator)

    def train(self, mode=True):
        super().train(mode); self.base.protected.eval(); return self
