import numpy as np
from teneva_bm import Bm


class BmHsFunc018(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 018 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 2 | <= 50
                x[1] | >= 0 | <= 50
            F - objective function
                (1 / 100) * x[0] ** 2 + x[1] ** 2
            C - constraint function
                x[0] * x[1] - 25 >= 0
                x[0] ** 2 + x[1] ** 2 - 25 >= 0
            The exact global minimum is known:
                y = 5
                x[0] = sqrt(250)
                x[1] = sqrt(2.5)
            Hyperparameters: 
                * The dimension d should be 2
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([2, 0], [50, 50])
        self.set_min(x=[np.sqrt(250), np.sqrt(2.5)], y=5)
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 2}

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
        c_1 = -1 * (X[:, 0] * X[:, 1] - 25)
        c_2 = -1 * (X[:, 0] ** 2 + X[:, 1] ** 2 - 25)
        return np.array([c_1, c_2])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return (1 / 100) * X[:, 0] ** 2 + X[:, 1] ** 2