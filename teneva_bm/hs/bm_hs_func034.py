import numpy as np
from teneva_bm import Bm


class BmHsFunc034(Bm):
    def __init__(self, d=3, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 034 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 0 | <= 100
                x[1] | >= 0 | <= 100
                x[2] | >= 0 | <= 10
            F - objective function
                -x[0]
            C - constraint function
                x[1] - exp(x[0]) >= 0
                x[2] - exp(x[1]) >= 0
            The exact global minimum is known:
                y = -log(log(10)) 
                x[0] = log(log(10)) 
                x[1] = log(10) 
                x[2] = 10
            Hyperparameters: 
                * The dimension d should be 3
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0], [100, 100, 10])
        self.set_min(x=[np.log(np.log(10)), np.log(10), 10], y=-np.log(np.log(10)))
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 3}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    @property
    def with_constr(self):
        return True

    def _constr_batch(self, X):
        c_1 = -1 * (X[:, 1] - np.exp(X[:, 0]))
        c_2 = -1 * (X[:, 2] - np.exp(X[:, 1]))
        return np.array([c_1, c_2])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return -X[:, 0]