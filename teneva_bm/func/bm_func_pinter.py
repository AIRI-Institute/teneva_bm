import numpy as np
from teneva_bm.func.func import Func


try:
    import torch
    with_torch = True
except Exception as e:
    with_torch = False


class BmFuncPinter(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Pinter function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-10, 10] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("89. Pinter Function"; Continuous, Differentiable,
            Non-separable, Scalable, Multimodal).
        """)

        self.set_grid(-10., +10., sh=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 100., 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 1844.09462695079

    def target_batch(self, X):
        Xm1 = np.hstack([X[:, -1].reshape(-1, 1), X[:, :-1]])
        Xp1 = np.hstack([X[:, +1:], X[:, +0].reshape(-1, 1)])

        A = Xm1 * np.sin(X) + np.sin(Xp1)
        B = Xm1**2 - 2. * X + 3. * Xp1 - np.cos(X) + 1.

        i = np.arange(1, self.d+1)

        y1 = np.sum(i * X**2, axis=1)
        y2 = np.sum(20 * i * np.sin(A)**2, axis=1)
        y3 = np.sum(i * np.log10(1. + i * B**2), axis=1)

        return y1 + y2 + y3

    def target_batch_pt(self, X):
        if not with_torch:
            raise ValueError('Can not import torch')
        
        device = X.device

        Xm1 = torch.hstack([X[:, -1].reshape(-1, 1), X[:, :-1]])
        Xp1 = torch.hstack([X[:, +1:], X[:, +0].reshape(-1, 1)])

        A = Xm1 * torch.sin(X) + torch.sin(Xp1)
        B = Xm1**2 - 2. * X + 3. * Xp1 - torch.cos(X) + 1.

        i = torch.arange(1, self.d+1, device=device)

        y1 = torch.sum(i * X**2, dim=1)
        y2 = torch.sum(20 * i * torch.sin(A)**2, dim=1)
        y3 = torch.sum(i * torch.log10(1. + i * B**2), dim=1)

        return y1 + y2 + y3