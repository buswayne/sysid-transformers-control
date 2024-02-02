# !/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
#from numba import float32, float64, jit, NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning
#import warnings
import casadi as ca
import casadi.tools as ct

#warnings.simplefilter('ignore', category=(NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning))

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
    data['ts'] = 1e-2
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
    x = ct.struct_symMX(['theta','omega','I'])
    u = ct.struct_symMX(['V'])

    return x, u


def servo_motor(x, u, data):
    """ System dynamics function (discrete time)
    """
    theta = x['theta']
    omega = x['omega']
    I = x['I']

    ts = 0.01
    fs = 1 / ts
    g = 9.8
    # Parameters - motor
    R = 9.5
    L = 0.84E-3
    K = 53.6E-3
    J = 2.2E-4
    b = 6.6E-5
    m = 0.07
    l = 0.042
    # state derivative expression
    # xdot = (ca.mtimes(np.array([[0, 1, 0], [0, -data['b'] / data['J'], data['K'] / data['J']], [0, -data['K'] / data['L'], -data['R'] / data['L']]]) +
    #                  np.array([[0, 1, 0], [data['m'] * data['g'] * data['l'], 0, 0], [0, 0, 0]]) * np.sin(theta) / theta,
    #                  ca.vertcat(theta, omega, I))
    #        + ca.mtimes(np.array([[0], [0], [1 / data['L']]]), u))

    # state derivative expression
    xdot = (ca.mtimes(np.array([[0, 1, 0], [0, -b / J, K / J], [0, -K / L, -R / L]]) +
                      np.array([[0, 1, 0], [m * g * l, 0, 0], [0, 0, 0]]) * np.sin(theta) / theta,
                      ca.vertcat(theta, omega, I))
            + ca.mtimes(np.array([[0], [0], [1 / L]]), u))

    # create ode for integrator
    ode = {'x': x, 'p': u, 'ode': xdot}

    return [ca.integrator('F', 'collocation', ode, {'tf': ts}), ode]


#@jit
def simulate_servo_positioning_system(Ts, u_in, perturbation, save_params=False, process_noise = False):
    # Data are constants that we can compute once
    data = problem_data(perturbation)

    # set-up system
    x, u = vars()
    f = servo_motor(x, u, Ts)[0]
    res = f(x0=x, p=u)
    x_next = res['xf']  # Access x_next as the DAE symbolic solution
    # Define function F to simplify mapping between symbolic (x,u) --> (x_next)
    F = ca.Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])

    # Define values for x and u
    x_s = [1e-6, 50, 1e-6]
    u_s = [1e-6]

    X_log = np.empty((3, 0))
    U_log = np.empty((1, 0))

    x_0 = np.array(x_s)
    u_0 = np.array(u_s)

    s = x_0

    for i in range(len(u_in)):

        a = u[i]                # u_{k}
        s_next = F(s, a)        # x_{k+1}


        print(s_next)

        U_log = np.column_stack((U_log, a))
        X_log = np.column_stack((X_log, s))

        print("B")
        s = np.array(s_next).flatten()  # update rule

        print("C")

    Y_log = X_log[0,:]

    if save_params:
        return X_log, U_log, Y_log, data
    else:
        return X_log, U_log, Y_log


if __name__ == "__main__":
    # Generate random forced inputs for simulation
    Ts = 1e-2
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
