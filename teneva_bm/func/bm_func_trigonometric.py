import numpy as np
from teneva_bm.func.func import Func
import teneva


try:
    import torch
    with_torch = True
except Exception as e:
    with_torch = False


class BmFuncTrigonometric(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Trigonometric function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [0, pi] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("152. Trigonometric Function 1"; Continuous, Differentiable,
            Non-separable, Scalable, Multimodal).
        """)

        self.set_grid(0., np.pi, sh=True, sh_out=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 25., 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 543.0253567898842

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        Yj = self.cores_add([-np.cos(x) for  x in X.T])
        Yj[-1][0, :, -1] += self.d

        Yj2 = teneva.mul(Yj, Yj)

        Ya = self.cores_add([2*(i+1) * (1. - np.cos(x) - np.sin(x)) for i, x in enumerate(X.T)])
        Yb = self.cores_add([ (  (i+1) * (1. - np.cos(x) - np.sin(x)) )**2 for i, x in enumerate(X.T)])

        Yj2[-1] *= self.d

        return teneva.add(Yj2,
                teneva.add(Yb,
                    teneva.mul(Ya, Yj)
                 )
                )

    def target_batch(self, X):
        i = np.arange(1, self.d+1)

        y1 = self.d
        y2 = -np.sum(np.cos(X), axis=1)
        Y2 = np.hstack([y2.reshape(-1, 1)]*self.d)
        Y3 = i * (1. - np.cos(X) - np.sin(X))

        return np.sum((y1 + Y2 + Y3)**2, axis=1)

    def target_batch_pt(self, X):
        if not with_torch:
            raise ValueError('Can not import torch')

        device = X.device
        
        y1 = self.d
        
        y2 = -torch.sum(torch.cos(X), dim=1)
        Y2 = torch.hstack([y2.reshape(-1, 1)]*self.d)
        
        i = torch.arange(1, self.d+1, device=device)
        Y3 = i * (1. - torch.cos(X) - torch.sin(X))
        
        return torch.sum((y1 + Y2 + Y3)**2, dim=1)