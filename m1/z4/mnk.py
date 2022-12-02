import numpy as np
from typing import List, Union, Callable


class MNK:
    def __init__(self, x: Union[List[Union[int, float]], np.ndarray], y: Union[List[Union[int, float]], np.ndarray],
                 funcs: List[Callable[[Union[int, float, np.ndarray]], Union[int, float, np.ndarray]]]):
        self.y = np.array(y.copy())
        self.x = np.array(x.copy())
        self.funcs = funcs.copy()
        a = np.array([f(x) for f in funcs]).T
        self.c = np.dot(np.dot(np.linalg.inv(np.dot(a.T, a)), a.T), y)

        def global_func(x: Union[int, float, np.ndarray]):
            result = 0
            for i in range(len(funcs)):
                result += self.c[i] * self.funcs[i](x)
                # print(x, c[i] * funcs[i](x))
            return result

        self.func = global_func

    def __call__(self, x: Union[int, float, np.ndarray]):
        return self.func(x)
