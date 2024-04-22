import numpy as np
from teneva_bm import Bm


class BmHsFunc073(Bm):
    def __init__(self, d=4, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 073 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 0
                x[1] | >= 0
                x[2] | >= 0
                x[3] | >= 0
            F - objective function
                24.55 * x[0] + 26.75 * x[1] + 39 * x[2] + 40.50 * x[3]
            C - constraint function
                2.3 * x[0] + 5.6 * x[1] + 11.1 * x[2] + 1.3 * x[3] - 5 >= 0
                12 * x[0] + 11.9 * x[1] + 41.8 * x[2] + 52.1 * x[3] - 21 - \
                1.645 * sqrt(0.28 * x[0] ** 2 + 0.19 * x[1] ** 2 + 20.5 * x[2] ** 2 + 0.62 * x[3] ** 2) >= 0
                x[0] + x[1] + x[2] + x[3] - 1 = 0
            The exact global minimum is approx. known:
                y ~= 29.894
                x[0] ~= 0.636
                x[1] ~= 0
                x[2] ~= 0.313
                x[3] ~= 0.052
            Hyperparameters: 
                * The dimension d should be 4
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0, 0], [+10, +10, +10, +10])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 4}

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
        c_1 = -1 * (2.3 * X[:, 0] + 5.6 * X[:, 1] + 11.1 * X[:, 2] + 1.3 * X[:, 3] - 5)
        c_2 = -1 * (
            12 * X[:, 0] + 11.9 * X[:, 1] + 41.8 * X[:, 2] + 52.1 * X[:, 3] - 21 - 
            1.645 * np.sqrt(0.28 * X[:, 0] ** 2 + 0.19 * X[:, 1] ** 2 + 20.5 * X[:, 2] ** 2 + 0.62 * X[:, 3] ** 2)
        )
        c_3 = np.abs(X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] - 1)
        return np.array([c_1, c_2, c_3])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return 24.55 * X[:, 0] + 26.75 * X[:, 1] + 39 * X[:, 2] + 40.50 * X[:, 3]