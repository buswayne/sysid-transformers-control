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
    data['den_2'] = np.float32(-1.479) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['den_3'] = np.float32(0.48) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    return data

def simulate_bldc(t, u, perturbation, save_params=False, process_noise=False):
    # Data are constants that we can compute once
    data = problem_data(perturbation)

    num = [data['num_1']]
    den = [data['den_1'], data['den_2'], data['den_3']]
    Ts = 0.02
    G = tf(num, den, dt=Ts)

    y, _, x = lsim(G, u, t)

    if save_params:
        return x, u, y, data
    else:
        return x, u, y

if __name__ == "__main__":
    # Generate random forced inputs for simulation
    ts = 0.02
    T = 30
    t = np.arange(0, T, ts)
    u = np.random.normal(0, 5, t.shape)

    print(len(u))
    # Perturbation factor for initial conditions
    perturbation = 0.0

    # Simulate the system trajectory using the model
    x, u, y = simulate_bldc(t, u, perturbation)

    tau = 5  # s
    s = tf('s')
    M = 1 / (1 + (tau / (2 * np.pi)) * s)
    M = M * (1 + 1e-2 * (tau / (2 * np.pi)) * s)  # add a high freq zero for inversion
    r_v = lsim(M ** (-1), y, t)[0]
    e_v = r_v - y

    plt.subplot(211)
    plt.plot(t, y)
    plt.legend([r'$y$', '$x_2 = \omega$', '$x_3 = I$'])
    plt.subplot(212)
    plt.plot(t, u)
    plt.legend(['$u = V$'])
    plt.show()