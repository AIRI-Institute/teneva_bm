import numpy as np
from teneva_bm.func.func import Func


try:
    import torch
    with_torch = True
except Exception as e:
    with_torch = False


class BmFuncAlpine(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Alpine function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-10, 10] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("6. Alpine Function 1"; Continuous, Non-Differentiable, Separable,
            Non-scalable, Multimodal).
        """)

        self.set_grid(-10., +10., sh=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 25., 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 32.67394403036597

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        return self.cores_add([np.abs(x * (np.sin(x) + 0.1)) for x in X.T])

    def target_batch(self, X):
        return np.sum(np.abs(X * np.sin(X) + 0.1 * X), axis=1)

    def target_batch_pt(self, X):
        if not with_torch:
            raise ValueError('Can not import torch')
        
        return torch.sum(torch.abs(X * torch.sin(X) + 0.1 * X), dim=1)