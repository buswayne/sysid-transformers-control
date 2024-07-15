import numpy as np
import cvxpy as cp
from control.matlab import *

def isproper(tf):
    degree_numerator = len(tf.num[0][0]) - 1
    degree_denominator = len(tf.den[0][0]) - 1
    # Determine if the transfer function is proper
    is_proper = degree_numerator <= degree_denominator
    return is_proper
def VRFT_ry(u, y, M, B, W=None, preFilt=None, return_error=False):

    L = M # approximation

    ts = M.dt
    t = np.arange(0, len(u)*ts, ts)

    u_L = lsim(L, u, t)[0]
    y_L = lsim(L, y, t)[0]

    z = tf([1, 0], [1], dt=ts)

    M_inv = M**-1
    while not isproper(M_inv):
        M_inv = M_inv/z

    r_v = lsim(M_inv, y_L, t)[0]
    e_v = r_v - y_L
    theta = cp.Variable(3)
    u_theta = theta[0] * lsim(B[0], e_v, t)[0] + theta[1] * lsim(B[1], e_v, t)[0] + theta[2] * lsim(B[2], e_v, t)[0]
    cost = cp.sum_squares(u_L - u_theta)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    theta_sol = theta.value

    for i in range(len(B)):
        if i == 0:
            C = theta_sol[i] * B[i]
        else:
            C += theta_sol[i] * B[i]

    if return_error:
        return theta_sol, C, e_v
    else:
        return theta_sol, C