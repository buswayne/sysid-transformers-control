
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
from numba import float32, float64, jit, NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning
import numba
import warnings
from control import tf, step_response
from lti import drss_matrices, dlsim
from control.matlab import *
import control

warnings.simplefilter('ignore', category=(NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0

"""
def fixed_wh_system_tentative(): ##SEEMS LIKE I CAN'T TRANSFORM TO TF G1 AND G2
    def nn_fun(x):
        out = x @ w1.transpose() + b1
        out = np.tanh(out)
        out = out @ w2.transpose() + b2
        return out

    n_in = 1
    n_out = 1
    n_hidden = 32
    n_skip = 200
    nx = 3
    random_order = True
    system_rng = np.random.default_rng(0)
    strictly_proper = True

    w1 = system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
    b1 = system_rng.normal(size=(1, n_hidden)) * 1.0
    w2 = system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
    b2 = system_rng.normal(size=(1, n_out)) * 1.0

    G1 = drss_matrices(states=system_rng.integers(1, nx + 1) if random_order else nx,
                       inputs=1,
                       outputs=1,
                       strictly_proper=strictly_proper,
                       rng=system_rng,
                       )

    G2 = drss_matrices(states=system_rng.integers(1, nx + 1) if random_order else nx,
                       inputs=1,
                       outputs=1,
                       strictly_proper=False,
                       rng=system_rng
                       )

    A1 = np.array([[*G1[0][0]], [*G1[0][1]]])
    B1 = np.array([*G1[1][0],*G1[1][1]])
    B1 = np.where(B1 == 0, 1e-6, B1)
    C1 = [*G1[2]]
    C1 = np.array(C1)
    # Replace 0s with 1e-6
    C1= np.where(C1 == 0, 1e-6, C1)
    D1 = [*G1[3]]
    D1 = np.array(D1)
    D1 = np.where(D1 == 0, 1e-6, D1)
    G1_ss = control.ss(A1, B1, C1, D1)
    G1_tf = control.ss2tf(G1_ss)


    A2 = np.array([[*G2[0][0]], [*G2[0][1]], [*G2[0][2]]])
    B2 = np.array([*G2[1][0],*G2[1][1],*G2[1][2]])
    B2 = np.where(B2 == 0, 1e-6, B2)
    C2 = [*G2[2]]
    C2 = np.array(C2)
    # Replace 0s with 1e-6
    C2 = np.where(C2 == 0, 1e-6, C2)
    D2 = [*G2[3]]
    D2 = np.array(D2)
    D2 = np.where(D2 == 0, 1e-6, D2)
    G2_ss = control.ss(A2, B2, *G2[2], *G2[3])
    G2_tf = control.ss2tf(G2_ss)


    result_dict = {
        'G1_tf': G1_tf,
        'G2_tf': G2_tf,
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2
    }

    return result_dict
"""

def fixed_wh_system(): ##SEEMS LIKE I CAN'T TRANSFORM TO TF G1 AND G2
    def nn_fun(x):
        out = x @ w1.transpose() + b1
        out = np.tanh(out)
        out = out @ w2.transpose() + b2
        return out

    n_in = 1
    n_out = 1
    n_hidden = 32
    n_skip = 200
    nx = 3
    random_order = True
    system_rng = np.random.default_rng(0)
    strictly_proper = True

    w1 = system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
    b1 = system_rng.normal(size=(1, n_hidden)) * 1.0
    w2 = system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
    b2 = system_rng.normal(size=(1, n_out)) * 1.0

    G1 = drss_matrices(states=system_rng.integers(1, nx + 1) if random_order else nx,
                       inputs=1,
                       outputs=1,
                       strictly_proper=strictly_proper,
                       rng=system_rng,
                       )

    G2 = drss_matrices(states=system_rng.integers(1, nx + 1) if random_order else nx,
                       inputs=1,
                       outputs=1,
                       strictly_proper=False,
                       rng=system_rng
                       )



    result_dict = {
        'G1': G1,
        'G2': G2,
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2
    }

    return result_dict

def simulate_wh(system,u):
    def nn_fun(x):
        out = x @ w1.transpose() + b1
        out = np.tanh(out)
        out = out @ w2.transpose() + b2
        return out

    G1 = system['G1']
    G2 = system['G2']
    w1 = system['w1']
    b1 = system['b1']
    w2 = system['w2']
    b2 = system['b2']

    # G1
    y1 = dlsim(*G1, u)
    y1 = (y1 - y1.mean()) / (y1.std() + 1e-6)

    # F
    y2 = nn_fun(y1)

    # G2
    y3 = dlsim(*G2, y2)

    u = u[:]
    y = y3[:]

    return u, y




if __name__ == "__main__":
    # Generate random forced inputs for simulation
    ts = 1e-2
    T = 5
    t = np.arange(0, T, ts)
    data_rng = np.random.default_rng(0)
    seq_len = 500
    n_skip = 200
    u = data_rng.normal(size=(seq_len , 1))
    u = u.reshape(-1, 1)


    # Simulate the system trajectory using the model
    #u,y, A1, B1, C1, D1, A2, B2, C2, D2, w1, b1, w2, b2 = simulate_wh(t,u)
    system = fixed_wh_system()

    G1 = system['G1']
    G2 = system['G2']
    w1 = system['w1']
    b1 = system['b1']
    w2 = system['w2']
    b2 = system['b2']

    #print(G1)
    #print(D2)
    u,y = simulate_wh(system,u)


    plt.subplot(211)
    plt.plot(t, u)
    plt.legend(['u'])
    plt.subplot(212)
    plt.plot(t, y)
    plt.legend(['y'])
    plt.show()




