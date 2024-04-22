import numpy as np
from teneva_bm import Bm


class BmHsFunc019(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 019 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 13 | <= 100
                x[1] | >= 0  | <= 100
            F - objective function
                (x[0] - 10) ** 3 + (x[1] - 20) ** 3
            C - constraint function
                (x[0] - 5) ** 2 + (x[1] - 5) ** 2 - 100 >= 0
                (-1) * (x[1] - 5) ** 2 - (x[0] - 6) ** 2 + 82.81 >= 0
            The exact global minimum is approx. known:
                y ~= -6961.814
                x[0] ~= 14.095
                x[1] ~= 0.843
            Hyperparameters: 
                * The dimension d should be 2
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([13, 0], [100, 100])
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
        c_1 = -1 * ((X[:, 0] - 5) ** 2 + (X[:, 1] - 5) ** 2 - 100)
        c_2 = -1 * ((-1) * (X[:, 1] - 5) ** 2 - (X[:, 0] - 6) ** 2 + 82.81)
        return np.array([c_1, c_2])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return (X[:, 0] - 10) ** 3 + (X[:, 1] - 20) ** 3