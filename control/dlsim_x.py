import math
from numpy import zeros, empty, cos, sin, any, copy
from numba import float32, float64, jit, NumbaPerformanceWarning
from numba.types import Tuple
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

"""""
##MAYBE I CAN WRITE A FASTER ONE-STEP VERSION ??
signatures = [
    Tuple((float32[:], float32[:]))(float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:], float32[:]),
    Tuple((float64[:], float64[:]))(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:])
]

@jit(signatures, nopython=True, cache=True)
def dlsim_x_onestep(A, B, C, D, u, x0):
    x_prev = copy(x0)
    y = C.dot(x_prev) + D.dot(u)
    x_i = A.dot(x_prev) + B.dot(u)
    y=y.reshape(-1)
    x_i = x_i.reshape(-1)
    return y , x_i
## maybe i can use this to speed up
"""

signatures = [
    Tuple((float32[:, :], float32[:, :]))(float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:], float32[:]),
    Tuple((float64[:, :], float64[:, :]))(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:])
]
@jit(signatures, nopython=True, cache=True)
def dlsim_x(A, B, C, D, u, x0):
    seq_len = u.shape[0]
    nx, nu = B.shape
    ny, _ = C.shape
    y = empty(shape=(seq_len, ny), dtype=u.dtype)
    x = empty(shape=(seq_len, nx), dtype=u.dtype)
    x[0] = copy(x0)  # x_step = zeros((nx,), dtype=u.dtype)
    for idx in range(seq_len):
        u_step = u[idx]
        x_step = x[idx]
        y[idx] = C.dot(x_step) + D.dot(u_step)
        if idx +1 < seq_len :
            x[idx+1] = A.dot(x_step) + B.dot(u_step)
    return y , x
