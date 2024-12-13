import numpy as np
from teneva_bm.func.func import Func


try:
    import torch
    with_torch = True
except Exception as e:
    with_torch = False


class BmFuncWavy(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Wavy function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-pi, pi] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("164. W / Wavy Function"; Continuous, Differentiable,
            Separable, Scalable, Multimodal).
        """)

        self.set_grid(-np.pi, +np.pi, sh=True)

        self.set_min(x=0., y=0.)

    @property
    def opts_info(self):
        return {**super().opts_info,
            'opt_k': {
                'desc': 'Param "k" for Wavy function',
                'kind': 'float',
                'form': '.2f',
                'dflt': 10.
            }
        }

    @property
    def opts_plot(self):
        return {'dy_min': 2., 'dy_max': 1.}

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        Y = self.cores_add(
            [-np.cos(self.opt_k * x) * np.exp(-x**2 / 2)/self.d
                for i, x in enumerate(X.T)])
        Y[-1][0, :, -1] += 1
        return Y

    @property
    def ref(self):
        return self.ref_i, 1.1671695279181264

    def target_batch(self, X):
        Y = np.cos(self.opt_k * X) * np.exp(-X**2 / 2)
        return 1. - np.sum(Y, axis=1) / self.d

    def target_batch_pt(self, X):
        if not with_torch:
            raise ValueError('Can not import torch')
        
        Y = torch.cos(self.opt_k * X) * torch.exp(-X**2 / 2)
        
        return 1. - torch.sum(Y, dim=1) / self.d