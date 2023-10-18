"""Newton Raphson"""


import numpy as np
from numpy import ndarray as array

from scipy.optimize import newton

N = 4  # No. of buses
Ng = 3  # No. of generator buses


B_bus = np.array(
    [
        [-30, 10, 10, 10],
        [10, -20, 10, 0],
        [10, 10, -30, 10],
        [10, 0, 10, -20],
    ]
)

gen_bus = [2, 4]

P_spec = np.array([1, -4, 1])
Q_spec = np.array([-2])
V_spec = {1: 1, 2: 1, 4: 1}


def angle_diff(theta: array) -> array:
    return np.subtract.outer(theta, theta)


def delta_P(V: array, theta: array) -> array:
    P_calc = V * np.matmul((B_bus * np.sin(angle_diff(theta))), V)
    return P_calc[1:] - P_spec


def delta_Q(V: array, theta: array) -> array:
    Q_calc = -1 * V * np.matmul((B_bus * np.cos(angle_diff(theta))), V)
    return Q_calc[Ng:] - Q_spec


def get_V(x: array) -> array:
    it = iter(i for i in x)
    return np.array([V_spec.get(i, None) or next(it) for i in range(1, N + 1)])


def delta(x: array) -> array:
    V = get_V(x[N - 1 :])
    theta = np.concatenate((np.zeros(1), x[: N - 1]))
    return np.concatenate((delta_P(V, theta), delta_Q(V, theta)))


x0 = np.concatenate((np.zeros(N - 1), np.ones(N - Ng)))

root = newton(delta, x0, maxiter=2)
