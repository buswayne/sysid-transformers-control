# !/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as ca
import casadi.tools as ct
from numba import float32, float64, jit, NumbaPerformanceWarning
from typing import Dict
import warnings
import torch
import torch.nn as nn

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

class CSTRModel(nn.Module):
    def __init__(self):
        super(CSTRModel, self).__init__()

    def forward(self, u_forced, perturbation=torch.tensor(0.5)):
        system_trajectory = simulate_cstr(u_forced, perturbation)
        return system_trajectory

def problem_data(perturbation: torch.Tensor = torch.tensor(0.5)) -> Dict[str, torch.Tensor]:
    """ Problem data, numeric constants,...
    """
    perturbation = torch.FloatTensor(perturbation)
    #data = {}
    data: Dict[str, torch.FloatTensor] = {}
    data['a'] = torch.FloatTensor(0.5616).to('cuda')
    data['b'] = torch.FloatTensor(0.3126).to('cuda')
    data['c'] = torch.FloatTensor(48.43).to('cuda')
    data['d'] = torch.FloatTensor(0.507).to('cuda')
    data['e'] = torch.FloatTensor(55.0).to('cuda')
    data['f'] = torch.FloatTensor(0.1538).to('cuda')
    data['g'] = torch.FloatTensor(90.0).to('cuda')
    data['h'] = torch.FloatTensor(0.16).to('cuda')
    data['M'] = 20.0 * (1.0 + perturbation * torch.FloatTensor(1).uniform_(-1.0, 1.0)).to('cuda') #(1 + perturbation * np.random.uniform(-1, 1))
    data['C'] = 4.0  * (1.0 + perturbation * torch.FloatTensor(1).uniform_(-1.0, 1.0)).to('cuda') #(1 + perturbation * np.random.uniform(-1, 1))
    data['UA2'] = 6.84 * (1.0 + perturbation * torch.FloatTensor(1).uniform_(-1., 1.)).to('cuda')#(1 + perturbation * np.random.uniform(-1, 1))
    data['Cp'] = 0.07 * (1.0 + perturbation * torch.FloatTensor(1).uniform_(-1., 1.)).to('cuda')#(1 + perturbation * np.random.uniform(-1, 1))
    data['lam'] = 38.5 * (1.0 + perturbation * torch.FloatTensor(1).uniform_(-1., 1.)).to('cuda')#(1 + perturbation * np.random.uniform(-1, 1))
    data['lams'] = 36.6 * (1.0 + perturbation * torch.FloatTensor(1).uniform_(-1., 1.)).to('cuda')#(1 + perturbation * np.random.uniform(-1, 1))
    data['F1'] = 10.0 * (1.0 + perturbation * torch.FloatTensor(1).uniform_(-1., 1.)).to('cuda')#(1 + perturbation * np.random.uniform(-1, 1))
    data['X1'] = 5.0 * (1. + perturbation * torch.FloatTensor(1).uniform_(-1., 1.)).to('cuda')#(1 + perturbation * np.random.uniform(-1, 1))
    data['F3'] = 50.0 * (1. + perturbation * torch.FloatTensor(1).uniform_(-1., 1.)).to('cuda')#(1 + perturbation * np.random.uniform(-1, 1))
    data['T1'] = 40.0 * (1. + perturbation * torch.FloatTensor(1).uniform_(-1., 1.)).to('cuda')#(1 + perturbation * np.random.uniform(-1, 1))
    data['T200'] = 25.0 * (1. + perturbation * torch.FloatTensor(1).uniform_(-1., 1.)).to('cuda')#(1 + perturbation * np.random.uniform(-1, 1))
    return data
def vars():
    """ System states and controls
    """
    # x = ['X2', 'P2']
    # x_s = [25.0, 49.743]
    # u_s = [191.713, 215.888]

    x = torch.tensor([25.0, 49.743], requires_grad=True).to('cuda')  # Initial state, modify as needed
    # u = ['P100', 'F200']
    u = torch.tensor([0.0, 0.0], requires_grad=True).to('cuda')  # Initial control input, modify as needed
    return x, u

def intermediate_vars(x: torch.Tensor, u:torch.Tensor, data: Dict[str, torch.Tensor]):
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

    merged_dict: Dict[str, torch.Tensor] = data.copy()
    new_data: Dict[str, torch.Tensor] = {
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
    merged_dict.update(new_data)
    return merged_dict

def dynamics(x:torch.Tensor, u:torch.Tensor, intermediate_data: Dict[str, torch.Tensor]):
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
    #xdot = torch.tensor([X2_dot, P2_dot])
    xdot = torch.cat((X2_dot, P2_dot))
    return xdot

#
# signatures = [
#     (float32[:, :], float32)
# ]


@torch.jit.script
def simulate_cstr(u_forced:torch.Tensor, perturbation: torch.Tensor = torch.tensor(0.5)):
    # Hyperparams
    Ts = 1
    # Data are constants that we can compute once
    data = problem_data(perturbation)
    # Initial conditions (u_0 actually is not required since we generated u_forced)
    x_0, u_0 = vars()
    # Initialize the tensor to store the trajectory
    X_log = torch.zeros((len(u_forced), 2), dtype=torch.float32).to('cuda')

    # Perturb the initial state (meta model)
    x_0 = x_0 * (1. + perturbation * torch.FloatTensor(2).uniform_(-1., 1.).to('cuda'))

    while True:
        #try:
        for i in range(len(u_forced)):
            s = x_0                 # state
            a = u_forced[i,:]       # action
            # Calculate intermediate variables
            intermediate_data = intermediate_vars(s, a, data)
            # Dynamics equation
            x_dot = dynamics(s, a, intermediate_data)
            # Integrate dynamics using forward Euler integration
            s_next = s + Ts * x_dot

            X_log[i, :] = x_0.detach()
            # recursively update the state
            x_0 = s_next
        return X_log

        # except Exception as e:
        #     print("Error:", e)
        #     continue  # try again to generate the system


if __name__ == "__main__":
    cstr_model = CSTRModel()
    # Generate random forced inputs for simulation
    num_steps = 300
    u_forced = 10 * torch.randn(num_steps, 2)  # Example: random inputs

    # Perturbation factor for initial conditions
    perturbation = torch.tensor(0.5)

    # Simulate the system trajectory using the model
    system_trajectory = cstr_model(u_forced, perturbation)

    plt.subplot(211)
    plt.plot(system_trajectory.cpu())
    plt.subplot(212)
    plt.plot(system_trajectory.cpu())
    plt.show()
