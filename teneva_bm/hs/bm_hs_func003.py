import numpy as np
from teneva_bm import Bm


class BmHsFunc003(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 003 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem without constraints:
            .-------------.
            | F(x) -> min |
            .-------------.
            x - continuous control
                x[0]
                x[1] | >= 0
            F - objective function
                x[1] + 0.00001 * (x[1] - x[0]) ** 2
            The exact global minimum is known: 
                y = 0
                x[0] = 0
                x[1] = 0
            Hyperparameters: 
            * The dimension d should be 2
            * The mode size n may be any (default is 64)
            * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, 0], [+10, +10])
        self.set_min(x=0, y=0)

    @property
    def args_constr(self):
        return {'d': 2}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    def target_batch(self, X):
        return X[:, 1] + 0.00001 * (X[:, 1] - X[:, 0]) ** 2
