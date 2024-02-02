# !/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
from numba import float32, float64, jit, NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning
import numba
import warnings

warnings.simplefilter('ignore', category=(NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0

@jit(nopython=True)
def problem_data(perturbation):
    """ Problem data, numeric constants,...
    """
    perturbation = np.float32(perturbation)
    data = {}
    data['g'] = 9.8
    # Parameters - motor
    data['R'] = np.float32(9.5) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['L'] = np.float32(0.84E-3) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['K'] = np.float32(53.6E-3) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['J'] = np.float32(2.2E-4) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['b'] = np.float32(6.6E-5) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['m'] = np.float32(0.07) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['l'] = np.float32(0.042) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    return data

@jit(nopython=True)
def vars():
    """ System states and controls
    """
    x = np.array([1e-3, 1e-3, 1e-3])  # theta, omega, I
    u = np.array([1e-3])
    return x, u

@jit(nopython=True)
def servo_motor(x, u, data):
    """ System dynamics function (discrete time)
    """
    theta = x[0].item()
    omega = x[1].item()
    I = x[2].item()

    # State derivative expression
    M1_00 = 0
    M1_01 = 1
    M1_02 = 0

    M1_10 = 0
    M1_11 = -data['b'] / data['J']
    M1_12 = data['K'] / data['J']

    M1_20 = 0
    M1_21 = -data['K'] / data['L']
    M1_22 = -data['R'] / data['L']

    M2_00 = 0
    M2_01 = 1 * np.sin(theta) / theta
    M2_02 = 0

    M2_10 = data['m'] * data['g'] * data['l'] / data['J'] * np.sin(theta) / theta
    M2_11 = 0
    M2_12 = 0

    M2_20 = 0
    M2_21 = 0
    M2_22 = 0

    M3_00 = theta
    M3_01 = omega
    M3_02 = I

    # Matrix multiplication using explicit element-wise operations
    xdot_0 = (M1_00 + M2_00) * M3_00 + (M1_01 + M2_01) * M3_01 + (M1_02 + M2_02) * M3_02
    xdot_1 = (M1_10 + M2_10) * M3_00 + (M1_11 + M2_11) * M3_01 + (M1_12 + M2_12) * M3_02
    xdot_2 = (M1_20 + M2_20) * M3_00 + (M1_21 + M2_21) * M3_01 + (M1_22 + M2_22) * M3_02 + u.item() / data['L']

    xdot = np.array([xdot_0, xdot_1, xdot_2])

    return xdot


@jit(nopython=True)
def simulate_servo_positioning_system(Ts, u, data=None, perturbation=0.0, save_params=False, process_noise=False):
    # Data are constants that we can compute once
    if not data:
        data = problem_data(perturbation)

    # Initial conditions (u_0 actually is not required since we generated u_forced)
    x_0, u_0 = vars()

    # Initialize the tensor to store the trajectory
    X_log = np.zeros((u.shape[0], 3))

    # Perturb the initial state (meta model)
    x_0 = x_0

    for i in range(len(u)-1):
        s = x_0
        a = u[i+1]

        # Runge-Kutta
        k1 = Ts * servo_motor(s, a, data)
        k2 = Ts * servo_motor(s + 0.5 * k1, a, data)
        s_next = s + k2

        x_0 = s_next

        X_log[i, :] = x_0

    y = X_log

    if save_params:
        return X_log, u, y, data
    else:
        return X_log, u, y, numba.typed.Dict.empty(key_type=numba.types.unicode_type, value_type=numba.types.float64)

if __name__ == "__main__":
    # Generate random forced inputs for simulation
    Ts = 0.0001
    T = 10
    t = np.arange(0, T, Ts)
    u = np.random.normal(0, 200, len(t))
    #u = 1e-3 * np.ones((t.shape[0],1))
    s = tf('s')
    tau = 1 / (1.6 * 2 * np.pi)
    F_input = 1 / (1 + tau * s)
    u_filtered = lsim(F_input, u, t)[0].reshape(-1,1)

    # Perturbation factor for initial conditions
    perturbation = 0.1

    # Simulate the system trajectory using the model
    x, u, y, _ = simulate_servo_positioning_system(Ts, u_filtered, perturbation=perturbation)

    plt.subplot(211)
    plt.plot(t, x[:, 0])
    plt.plot(t, x[:, 1])
    plt.plot(t, x[:, 2])
    plt.legend([r'$x_1 = \theta$', '$x_2 = \omega$', '$x_3 = I$'])
    plt.subplot(212)
    plt.plot(t, u)
    plt.legend(['$u = V$'])
    plt.show()
