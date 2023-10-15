# !/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as ca
import casadi.tools as ct
from numba import float32, float64, jit, NumbaPerformanceWarning
import warnings

def create_cstr(u_forced, perturbation=0.5, noisy=False):
    # Hyperparams
    Ts = 1

    # Time vector
    # t = np.arange(0,len(u_forced)) * Ts

    # steady-state conditions for the nominal model
    x_s = [25.0, 49.743]
    u_s = [191.713, 215.888]

    # set-up system (either with or without noise)

    if noisy:
        # set-up noisy system
        x, u_noisy = vars_noisy()
        data_noisy = intermediate_vars_noisy(x, u_noisy, problem_data())
        f_noisy = dynamics(x, u_noisy, data_noisy)[0]
        # l_noisy = objective(x, u_noisy, data_noisy)
        # h_noisy = constraints(x, u_noisy, data_noisy)
        res_noisy = f_noisy(x0=x, p=u_noisy)
        x_next_noisy = res_noisy['xf']
        F_noisy = ca.Function('F_noisy', [x, u_noisy], [x_next_noisy], ['x', 'u_noisy'], ['x_next'])

    else:
        x, u = vars()
        data = intermediate_vars(x, u, problem_data())
        f = dynamics(x, u, data)[0]
        # l = objective(x, u, data)
        # h = constraints(x, u, data)
        res = f(x0=x, p=u)
        x_next = res['xf']  # Access x_next as the DAE symbolic solution
        # Define function F to simplify mapping between symbolic (x,u) --> (x_next)
        F = ca.Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])

    X_log = np.zeros((len(u_forced), 2))

    # Perturb also the initial state
    x_0 = np.array(x_s) * (1 + perturbation * np.random.uniform(-1, 1))

    while True:
        try:
            for i in range(len(u_forced)):
                s = x_0                 # state
                a = u_forced[i, :]      # action

                if noisy:
                    a = ca.vertcat(a,
                                   np.random.normal(loc=.0, scale=1),
                                   np.random.normal(loc=.0, scale=2),
                                   np.random.normal(loc=.0, scale=8),
                                   np.random.normal(loc=.0, scale=5))
                    s_next = F_noisy(s, a)
                else:
                    s_next = F(s, a)

                x_0 = np.array(s_next).flatten()

                #U_log = np.column_stack((U_log, a[:2]))
                #X_log = np.row_stack((X_log, x_0))
                X_log[i,:] = x_0

            #print(X_log.shape)
            return X_log
        except:
            continue # try again to generate the system

def problem_data(perturbation=0.5):
    """ Problem data, numeric constants,...
    """
    data = {}
    data['a'] = 0.5616
    data['b'] = 0.3126
    data['c'] = 48.43
    data['d'] = 0.507
    data['e'] = 55.0
    data['f'] = 0.1538
    data['g'] = 90.0
    data['h'] = 0.16
    data['M'] = 20.0 * (1 + perturbation * np.random.uniform(-1, 1))
    data['C'] = 4.0 * (1 + perturbation * np.random.uniform(-1, 1))
    data['UA2'] = 6.84 * (1 + perturbation * np.random.uniform(-1, 1))
    data['Cp'] = 0.07 * (1 + perturbation * np.random.uniform(-1, 1))
    data['lam'] = 38.5 * (1 + perturbation * np.random.uniform(-1, 1))
    data['lams'] = 36.6 * (1 + perturbation * np.random.uniform(-1, 1))
    data['F1'] = 10.0 * (1 + perturbation * np.random.uniform(-1, 1))
    data['X1'] = 5.0 * (1 + perturbation * np.random.uniform(-1, 1))
    data['F3'] = 50.0 * (1 + perturbation * np.random.uniform(-1, 1))
    data['T1'] = 40.0 * (1 + perturbation * np.random.uniform(-1, 1))
    data['T200'] = 25.0 * (1 + perturbation * np.random.uniform(-1, 1))
    return data

def intermediate_vars(x, u, data):
    """ Intermediate model variables
    """
    data['T2'] = data['a'] * x['P2'] + data['b'] * x['X2'] + data['c']
    data['T3'] = data['d'] * x['P2'] + data['e']
    data['T100'] = data['f'] * u['P100'] + data['g']  # added noise
    data['UA1'] = data['h'] * (data['F1'] + data['F3'])
    data['Q100'] = data['UA1'] * (data['T100'] - data['T2'])
    data['F100'] = data['Q100'] / data['lams']
    data['Q200'] = data['UA2'] * (data['T3'] - data['T200']) / (1.0 + data['UA2'] / (2.0 * data['Cp'] * u['F200']))
    data['F5'] = data['Q200'] / data['lam']
    data['F4'] = (data['Q100'] - data['F1'] * data['Cp'] * (data['T2'] - data['T1'])) / data['lam']
    data['F2'] = (data['F1'] - data['F4'])
    return data

def intermediate_vars_noisy(x, u, data):
    """ Intermediate model variables
    """
    # Add noise to X1, F2, T1, T200
    data['X1'] += u['X1_noise']
    data['T1'] += u['T1_noise']
    data['T200'] += u['T200_noise']
    data['T2'] = data['a'] * x['P2'] + data['b'] * x['X2'] + data['c']
    data['T3'] = data['d'] * x['P2'] + data['e']
    data['T100'] = data['f'] * u['P100'] + data['g']  # added noise
    data['UA1'] = data['h'] * (data['F1'] + data['F3'])
    data['Q100'] = data['UA1'] * (data['T100'] - data['T2'])
    data['F100'] = data['Q100'] / data['lams']
    data['Q200'] = data['UA2'] * (data['T3'] - data['T200']) / (1.0 + data['UA2'] / (2.0 * data['Cp'] * u['F200']))
    data['F5'] = data['Q200'] / data['lam']
    data['F4'] = (data['Q100'] - data['F1'] * data['Cp'] * (data['T2'] - data['T1'])) / data['lam']
    data['F2'] = (data['F1'] - data['F4'])
    data['F2'] += u['F2_noise']
    return data

def dynamics(x, u, data):
    """ System dynamics function (discrete time)
    """
    # state derivative expression
    xdot = ca.vertcat((data['F1'] * data['X1'] - data['F2'] * x['X2']) / data['M'],
                      (data['F4'] - data['F5']) / data['C']
    )
    # create ode for integrator
    ode = {'x': x, 'p': u, 'ode': xdot}
    integrator_opts = {'jit':True,}
    return [ca.integrator('F', 'rk', ode, integrator_opts), ode]

def vars():
    """ System states and controls
    """
    x = ct.struct_symMX(['X2', 'P2'])
    u = ct.struct_symMX(['P100', 'F200'])
    return x, u

def vars_noisy():
    """ System states and controls (with additional noise)
    """
    x = ct.struct_symMX(['X2', 'P2'])
    u = ct.struct_symMX(['P100', 'F200', 'X1_noise', 'F2_noise', 'T1_noise', 'T200_noise'])
    return x, u

def objective(x, u, data):
    """ Economic objective function
    """
    # cost definition
    obj = 10.09 * (data['F2'] + data['F3']) + 600.0 * data['F100'] + 0.6 * u['F200']
    return ca.Function('economic_cost', [x, u], [obj])

def constraints(x, u, data):
    """ Path inequality constraints function (convention h(x,u) >= 0)
    """
    constr = ca.vertcat(
        x['X2'] - 25.0,
        x['P2'] - 40.0,
        100.0 - x['X2'],
        80.0 - x['P2'],
        400.0 - u['P100'],
        400.0 - u['F200'],
    )
    return ca.Function('h', [x, u], [constr])