# !/usr/bin/env python
# coding: utf-8

import autograd.numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
from numba import float32, float64, jit, NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=(NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0

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
def vars():
    """ System states and controls
    """
    # x = ct.struct_symMX(['theta','omega','I'])
    # u = ct.struct_symMX(['V'])
    x = np.array([1e-3, 1e-3, 1e-3])  # theta, omega, I
    u = np.array([1e-3])
    return x, u


def servo_motor(x, u, data):
    """ System dynamics function (discrete time)
    """
    theta = x[0].item()
    omega = x[1].item()
    I = x[2].item()

    # state derivative expression
    xdot = (np.dot(np.array([[0, 1, 0], [0, -data['b'] / data['J'], data['K'] / data['J']], [0, -data['K'] / data['L'], -data['R'] / data['L']]]) +
                   np.array([[0, 1, 0], [(data['m'] * data['g'] * data['l'])/data['J'], 0, 0], [0, 0, 0]]) * np.sin(theta) / theta,
                   np.array([[theta], [omega], [I]])) +
            np.dot(np.array([[0], [0], [1 / data['L']]]), u.item()))

    return xdot


@jit
def simulate_servo_positioning_system(Ts, u, perturbation, save_params=False, process_noise = False):
    # Data are constants that we can compute once
    data = problem_data(perturbation)

    # Initial conditions (u_0 actually is not required since we generated u_forced)
    x_0, u_0 = vars()

    # Initialize the tensor to store the trajectory
    X_log = np.zeros((u.shape[0], 3))
    # U_log = np.zeros((u.shape[0], 1))
    # Noisy

    # Perturb the initial state (meta model)
    x_0 = x_0# * (1. + perturbation * np.random.uniform(-1., 1.))


    for i in range(len(u)):

        # print(i)
        s = x_0
        a = u[i]

        # Euler
        # s_dot = servo_motor(s, a)
        # s_next = s.reshape(-1,1) + s_dot * Ts

        # Runge-Kutta
        k1 = Ts * servo_motor(s, a, data)
        k2 = Ts * servo_motor(s.reshape(-1, 1) + 0.5 * k1, a, data)
        s_next = s.reshape(-1, 1) + k2

        x_0 = np.array(s_next).flatten()

        X_log[i,:] = x_0

    y = X_log

    if save_params:
        return X_log, u, y, data
    else:
        return X_log, u, y


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
    perturbation = 0.0

    # Simulate the system trajectory using the model
    x, u, y = simulate_servo_positioning_system(Ts, u_filtered, perturbation)

    plt.subplot(211)
    plt.plot(t, x[:, 0])
    plt.plot(t, x[:, 1])
    plt.plot(t, x[:, 2])
    plt.legend([r'$x_1 = \theta$', '$x_2 = \omega$', '$x_3 = I$'])
    plt.subplot(212)
    plt.plot(t, u)
    plt.legend(['$u = V$'])
    plt.show()
