import numpy as np
from teneva_bm.func.func import Func


try:
    import torch
    with_torch = True
except Exception as e:
    with_torch = False


class BmFuncDixon(Func):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Analytical Dixon function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-10, 10] (with small random shift);
            the exact global minimum is known: x_i = 2^{(2^i-2) / 2^i}
            (i = 1, ..., d), y = 0;  note that this function achieves a global
            minimum at more than one point.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("48. Dixon & Price Function"; Continuous, Differentiable,
            Non-Separable, Scalable, Unimodal).
            See also https://www.sfu.ca/~ssurjano/dixonpr.html for details.
        """)

        self.set_grid(-10., +10., sh=True)

        x = [1.]
        for _ in range(d-1):
            x.append(np.sqrt(x[-1] / 2.))
        self.set_min(x=x, y=0.)

    @property
    def opts_plot(self):
        return {'dy_min': 1.E+5, 'dy_max': 0.}

    @property
    def ref(self):
        return self.ref_i, 383674.42504801514

    @property
    def with_cores(self):
        return True

    def cores(self, X):
        def _core(x, i, pos='m'):
            x = np.asarray(x)
            n = len(x)
            if i == 1:
                pos = 'f'

            x2 = x*x
            x4 = x2*x2

            if pos[0] != 'f':
                c = np.zeros([3, n, 3])
                c[0, :, 0] = 1
                c[-1, :, -1] = 1
                c[0, :, 1]  = x
                c[0, :, -1]  = i*4*x4 + (pos[0] != 'l')*(i+1)*x2
                c[1, :, -1]  = -4*i*x2

            else: # First core
                c = np.zeros([1, n, 3])
                c[0, :, 0] = 1
                c[0, :, 1]  = x
                c[0, :, -1]  = (x-1)**2 + (i+1)*x2

            if pos[0] == 'l':
                c = np.copy(c[..., -1:])

            return c

        return [_core(x, i+1, 'l' if i == self.d-1 else 'm')
            for i, x in enumerate(X.T)]

    def target_batch(self, X):
        y1 = (X[:, 0] - 1)**2

        i = np.arange(2, self.d+1)
        y2 = i * (2. * X[:, 1:]**2 - X[:, :-1])**2
        y2 = np.sum(y2, axis=1)

        return y1 + y2

    def target_batch_pt(self, X):
        if not with_torch:
            raise ValueError('Can not import torch')
        
        device = X.device

        y1 = (X[:, 0] - 1)**2

        i = torch.arange(2, self.d+1, device=device)
        y2 = i * (2. * X[:, 1:]**2 - X[:, :-1])**2
        y2 = torch.sum(y2, dim=1)

        return y1 + y2