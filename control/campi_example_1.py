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

def problem_data(perturbation):
    """ Problem data, numeric constants,...
    """
    perturbation = np.float32(perturbation)
    data = {}
    # Parameters
    data['m1'] = np.float32(1) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['m2'] = np.float32(0.5) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['c1'] = np.float32(0.2) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['c2'] = np.float32(0.5) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['k1'] = np.float32(1) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    data['k2'] = np.float32(0.5) * (1.0 + perturbation * np.random.uniform(-1.0, 1.0))
    return data


def simulate_campi_example_1(perturbation=0.0):
    Ts = 0.1; Fs = 1 / Ts
    s = tf('s')
    # problem data
    data = problem_data(perturbation)
    # system
    P = ((data['m1'] * s ** 2 + (data['c1'] + data['c2']) * s + (data['k1'] + data['k2'])) /
        ((data['m1'] * s ** 2 + (data['c1'] + data['c2']) * s + (data['k1'] + data['k2'])) *
         (data['m2'] * s ** 2 +  data['c2'] * s + data['k2']) - (data['k2'] + data['c2'] * s) ** 2))
    # input experiment
    T = 150
    t = np.arange(0, T, Ts)
    u = np.random.normal(0, 10, t.shape)
    # T_u = 50  # s
    # u = np.zeros(t.shape)
    # u[(t >= 0 * T_u / 2) & (t < 1 * T_u / 2)] = 1
    # u[(t >= 2 * T_u / 2) & (t < 3 * T_u / 2)] = 1
    # u[(t >= 4 * T_u / 2) & (t < 5 * T_u / 2)] = 1
    # u[(t >= 6 * T_u / 2) & (t < 7 * T_u / 2)] = 1
    # simulation
    y, _, x = lsim(P, u, t)

    return t, x, u, y, data


if __name__ == "__main__":

    # Perturbation factor for initial conditions
    perturbation = 0.0

    # Simulate the system trajectory using the model
    t, x, u, y, _ = simulate_campi_example_1(perturbation=perturbation)

    plt.figure()
    plt.plot(t, u, label="u")
    plt.plot(t, y, label="y")
    plt.legend()
    plt.show()