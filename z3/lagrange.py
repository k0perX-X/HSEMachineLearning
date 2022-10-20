# import numba
# import numba.types
from typing import List, Union
import numpy as np
from pprint import pprint
import pandas as pd


# @numba.jit(parallel=True)
def _iner_(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([y[i] / np.prod(x[i] - np.delete(x, i)) for i in range(len(x))])


# @numba.jit(parallel=True)
def _func_array_(value: np.ndarray, x: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    out = np.empty(len(value), dtype="float")
    for j in range(len(value)):
        sum = 0
        for i in range(len(x)):
            sum += np.prod(value[j] - np.delete(x, i)) * coefs[i]
        out[j] = sum
    return out


# @numba.jit(parallel=True)
def _func_one_el_(value: float, x: np.ndarray, coefs: np.ndarray) -> float:
    sum = 0
    for i in range(len(x)):
        sum += np.prod(value - np.delete(x, i)) * coefs[i]
    return sum


# print("Compiled")


class Lagrange:

    def __init__(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray]):
        self.n = len(x)
        if len(x) != len(y):
            raise Exception("Length of x not the same of y")
        self.x = np.array(x)
        self.y = np.array(y)
        self.coefs = _iner_(self.x, self.y)
        return

    def __call__(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        if type(x) == np.ndarray:
            return _func_array_(x, self.x, self.coefs)
        else:
            return _func_one_el_(float(x), self.x, self.coefs)


if __name__ == '__main__':
    from scipy.interpolate import lagrange
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1, 0])
    l = Lagrange(x, y)
    pprint(l.coefs)
    l1 = lagrange(x, y)
    pprint(l1.coef)
    pprint(pd.DataFrame({"x": x, "y": y, "my lag": l(x), "scipy lag": l1(x)}))
