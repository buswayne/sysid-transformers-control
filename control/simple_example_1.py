# !/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
from numba import float32, float64, jit, NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning
import numba
import warnings

import warnings

# Disable all user warnings
warnings.filterwarnings("ignore")

# Your code goes here

# Re-enable user warnings
warnings.filterwarnings("default")

warnings.simplefilter('ignore', category=(NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning))

# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams['axes.labelsize']=14
# plt.rcParams['xtick.labelsize']=11
# plt.rcParams['ytick.labelsize']=11
# plt.rcParams['axes.grid']=True
# plt.rcParams['axes.xmargin']=0

@jit(nopython=True)
def problem_data(perturbation):
    """ Problem data, numeric constants,...
    """
    perturbation = np.float32(perturbation)
    data = {}
    data['num_1'] = np.float32(1) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    # data['num_2'] = np.float32(0) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['den_1'] = np.float32(1) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['den_2'] = np.float32(3) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['den_3'] = np.float32(2) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    return data

def simulate_simple_example_1(t, u, perturbation, save_params=False, process_noise=False):
    # Data are constants that we can compute once
    data = problem_data(perturbation)

    num = [data['num_1']]
    den = [data['den_1'], data['den_2'], data['den_3']]

    ts = t[1] - t[0]

    s = tf('s')
    z = tf([1, 0], [1], ts)
    G = tf(num, den)
    # G_proper = z ** 2 * c2d(G, ts, 'matched')
    # tau = stepinfo(G)['SettlingTime'] / 5
    # G_proper = G * (1 + 5e-1 * (tau / (2 * np.pi)) * s) ** 2

    # amplitude = np.random.uniform(-100,100)
    # r = amplitude * np.ones(t.shape)
    #
    #
    # tau = 1  # s
    # M = 1 / (1 + (tau / (2 * np.pi)) * s)
    # M = c2d(M, ts, 'matched')
    #
    # y_d = lsim(M, r, t)[0]

    # print(G)
    # u = lsim(G_proper**-1, y_d, t)[0]

    # u = lsim(G_proper**(-1), u, t)[0]

    # y = lsim(G_proper, u, t)[0]
    y = lsim(G, u, t)[0]

    if save_params:
        return u, y, data
    else:
        return u, y

if __name__ == "__main__":
    # Generate random forced inputs for simulation
    ts = 1e-2
    T = 5
    t = np.arange(0, T, ts)
    u = np.random.normal(0, 10, t.shape)

    print(len(u))
    # Perturbation factor for initial conditions
    perturbation = 0.0

    # Simulate the system trajectory using the model
    x, u, y = simulate_simple_example_1(t, u, perturbation)

    plt.subplot(211)
    plt.plot(t, y)
    plt.legend([r'$x_1 = \theta$', '$x_2 = \omega$', '$x_3 = I$'])
    plt.subplot(212)
    plt.plot(t, u)
    plt.legend(['$u = V$'])
    plt.show()
