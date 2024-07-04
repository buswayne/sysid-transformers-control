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
    G = tf(num, den)
    #print(G)

    s = tf('s')
    tau = 0.5  # s
    M = 1 / (1 + (tau / (2 * np.pi)) * s)
    M = M * (1 + 1e-2 * (tau / (2 * np.pi)) * s)
    #print(M)
    u_prefilter = lsim(M, u, t)[0]
    y, _, x = lsim(G, u, t)
    y_prefilter = lsim(M, y, t)[0]

    if save_params:
        return x, u_prefilter, y_prefilter, data
    else:
        return x, u, y

if __name__ == "__main__":
    # Generate random forced inputs for simulation
    ts = 1e-2
    T = 5
    t = np.arange(0, T, ts)
    u = np.random.normal(0, 1000, t.shape)

    print(len(u))
    # Perturbation factor for initial conditions
    perturbation = 0.0

    # Simulate the system trajectory using the model
    x, u, y = simulate_simple_example_1(t, u, perturbation)

    plt.subplot(211)
    plt.plot(t, y)
    plt.legend(['y'])
    plt.subplot(212)
    plt.plot(t, u)
    plt.legend(['u'])
    plt.show()
