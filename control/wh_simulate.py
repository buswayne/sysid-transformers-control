
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *
from numba import float32, float64, jit, NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning
import numba
import warnings
from control import tf, step_response
from lti import drss_matrices, dlsim

warnings.simplefilter('ignore', category=(NumbaWarning, NumbaPerformanceWarning, NumbaDeprecationWarning))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0




def simulate_wh(t,u):
    def nn_fun(x):
        out = x @ w1.transpose() + b1
        out = np.tanh(out)
        out = out @ w2.transpose() + b2
        return out

    n_in = 1
    n_out = 1
    n_hidden = 32
    n_skip = 200
    nx = 5
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

    # G1
    y1 = dlsim(*G1, u)
    y1 = (y1 - y1[n_skip:].mean(axis=0)) / (y1[n_skip:].std(axis=0) + 1e-6)


    # F
    y2 = nn_fun(y1)

    # G2
    y3 = dlsim(*G2, y2)

    u = u[n_skip:]
    y = y3[n_skip:]

    return u,y,G1, G2, w1, b1, w2, b2




if __name__ == "__main__":
    # Generate random forced inputs for simulation
    ts = 1e-2
    T = 5
    t = np.arange(0, T, ts)
    data_rng = np.random.default_rng(0)
    seq_len = 500
    n_skip = 200
    u = data_rng.normal(size=(seq_len + n_skip, 1))
    u = u.reshape(-1, 1)


    # Simulate the system trajectory using the model
    u,y, G1, G2, w1, b1, w2, b2 = simulate_wh(t,u)



    plt.subplot(211)
    plt.plot(t, u)
    plt.legend(['u'])
    plt.subplot(212)
    plt.plot(t, y)
    plt.legend(['e_v'])
    plt.show()




