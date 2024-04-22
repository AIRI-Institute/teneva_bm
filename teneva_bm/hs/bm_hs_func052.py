import numpy as np
from teneva_bm import Bm


class BmHsFunc052(Bm):
    def __init__(self, d=5, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 052 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0]
                x[1]
                x[2]
                x[3]
                x[4]
            F - objective function
                (4 * x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2
            C - constraint function
                x[0] + 3 * x[1] = 0
                x[2] + x[3] - 2 * x[4] = 0
                x[1] - x[4] = 0
            The exact global minimum is known:
                y = 1859/349 
                x[0] = -33/349 
                x[1] = 11/349 
                x[2] = 180/349 
                x[3] = -158/349 
                x[4] = 11/349 
            Hyperparameters: 
                * The dimension d should be 5
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, -10, -10, -10, -10], [+10, +10, +10, +10, +10])
        self.set_min(x=[-33/349, 11/349, 180/349, -158/349, 11/349], y=1859/349)
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 5}

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
        c_1 = np.abs(X[:, 0] + 3 * X[:, 1])
        c_2 = np.abs(X[:, 2] + X[:, 3] - 2 * X[:, 4])
        c_3 = np.abs(X[:, 1] - X[:, 4])
        return np.array([c_1, c_2, c_3])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return (4 * X[:, 0] - X[:, 1]) ** 2 + (X[:, 1] + \
                X[:, 2] - 2) ** 2 + (X[:, 3] - 1) ** 2 + (X[:, 4] - 1) ** 2