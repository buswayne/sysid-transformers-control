# !/usr/bin/env python
# coding: utf-8

import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as ca
import casadi.tools as ct
from numba import float32, float64, jit, NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning
from typing import Dict
import warnings
# import torch
# import torch.nn as nn

warnings.simplefilter('ignore', category=(NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning))


def problem_data(perturbation):
    """ Problem data, numeric constants,...
    """
    perturbation = np.float32(perturbation)
    data = {}
    data['a'] = np.float32(0.5616)
    data['b'] = np.float32(0.3126)
    data['c'] = np.float32(48.43)
    data['d'] = np.float32(0.507)
    data['e'] = np.float32(55.0)
    data['f'] = np.float32(0.1538)
    data['g'] = np.float32(90.0)
    data['h'] = np.float32(0.16)
    data['M'] = 20.0 * (1.0 + perturbation * np.random.uniform(-1.0, 1.0)) #(1 + perturbation * np.random.uniform(-1, 1))
    data['C'] = 4.0 * (1.0 + perturbation * np.random.uniform(-1.0, 1.0)) #(1 + perturbation * np.random.uniform(-1, 1))
    data['UA2'] = 6.84 * (1.0 + perturbation * np.random.uniform(-1., 1.))#(1 + perturbation * np.random.uniform(-1, 1))
    data['Cp'] = 0.07 * (1.0 + perturbation * np.random.uniform(-1., 1.))#(1 + perturbation * np.random.uniform(-1, 1))
    data['lam'] = 38.5 * (1.0 + perturbation * np.random.uniform(-1., 1.))#(1 + perturbation * np.random.uniform(-1, 1))
    data['lams'] = 36.6 * (1.0 + perturbation * np.random.uniform(-1., 1.))#(1 + perturbation * np.random.uniform(-1, 1))
    data['F1'] = 10.0 * (1.0 + perturbation * np.random.uniform(-1., 1.))#(1 + perturbation * np.random.uniform(-1, 1))
    data['X1'] = 5.0 * (1. + perturbation * np.random.uniform(-1., 1.))#(1 + perturbation * np.random.uniform(-1, 1))
    data['F3'] = 50.0 * (1. + perturbation * np.random.uniform(-1., 1.))#(1 + perturbation * np.random.uniform(-1, 1))
    data['T1'] = 40.0 * (1. + perturbation * np.random.uniform(-1., 1.))#(1 + perturbation * np.random.uniform(-1, 1))
    data['T200'] = 25.0 * (1. + perturbation * np.random.uniform(-1., 1.))#(1 + perturbation * np.random.uniform(-1, 1))
    return data
def vars():
    """ System states and controls
    """
    # x = ['X2', 'P2']
    # x_s = [25.0, 49.743]
    # u_s = [191.713, 215.888]

    x = np.array([25.0, 49.743])  # Initial state, modify as needed
    # u = ['P100', 'F200']
    u = np.array([0.0, 0.0])  # Initial control input, modify as needed
    return x, u

def intermediate_vars(x, u, data):
    """ Intermediate model variables
    """
    # Intermediate variable calculations based on the provided equations
    T2 = data['a'] * x[1] + data['b'] * x[0] + data['c']
    T3 = data['d'] * x[1] + data['e']
    T100 = data['f'] * u[0] + data['g']  # added noise
    UA1 = data['h'] * (data['F1'] + data['F3'])
    Q100 = UA1 * (T100 - T2)
    F100 = Q100 / data['lams']
    Q200 = data['UA2'] * (T3 - data['T200']) / (1.0 + data['UA2'] / (2.0 * data['Cp'] * u[1]))
    F5 = Q200 / data['lam']
    F4 = (Q100 - data['F1'] * data['Cp'] * (T2 - data['T1'])) / data['lam']
    F2 = (data['F1'] - F4)

    return data | {
        'T2': T2,
        'T3': T3,
        'T100': T100,
        'UA1': UA1,
        'Q100': Q100,
        'F100': F100,
        'Q200': Q200,
        'F5': F5,
        'F4': F4,
        'F2': F2
        }

def dynamics(x, u, intermediate_data):
    """ System dynamics function (discrete time)
    """
    F1 = intermediate_data['F1']
    F2 = intermediate_data['F2']
    F4 = intermediate_data['F4']
    F5 = intermediate_data['F5']
    X1 = intermediate_data['X1']
    Cp = intermediate_data['Cp']
    T1 = intermediate_data['T1']

    # State derivative calculations
    X2_dot = (F1 * X1 - F2 * x[0]) / intermediate_data['M']
    P2_dot = (F4 - F5) / intermediate_data['C']

    # Return state derivatives as a tensor
    xdot = np.array([X2_dot, P2_dot])

    return xdot

#
# signatures = [
#     (float32[:, :], float32)
# ]


@jit
def simulate_cstr(u, perturbation):
    # Hyperparams
    Ts = 1
    # Data are constants that we can compute once
    data = problem_data(perturbation)
    # Initial conditions (u_0 actually is not required since we generated u_forced)
    x_0, u_0 = vars()
    # Initialize the tensor to store the trajectory
    y = np.zeros((len(u), 2))

    # Perturb the initial state (meta model)
    x_0 = x_0 * (1. + perturbation * np.random.uniform(-1., 1.))

    while True:
        for i in range(len(u)):
            s = x_0                 # state
            a = u[i,:]       # action
            # Calculate intermediate variables
            intermediate_data = intermediate_vars(s, a, data)
            # Dynamics equation
            x_dot = dynamics(s, a, intermediate_data)
            # Integrate dynamics using forward Euler integration
            s_next = s + Ts * x_dot

            y[i, :] = s    # y1 and y2 at time t
            # recursively update the state
            x_0 = s_next

        return y


if __name__ == "__main__":
    # Generate random forced inputs for simulation
    num_steps = 300
    u_forced = np.random.normal(0, 10, (num_steps, 2))  # Example: random inputs

    # Perturbation factor for initial conditions
    perturbation = 0.25

    # Simulate the system trajectory using the model
    system_trajectory = simulate_cstr(u_forced, perturbation)

    plt.subplot(211)
    plt.plot(system_trajectory)
    plt.subplot(212)
    plt.plot(system_trajectory)
    plt.show()
