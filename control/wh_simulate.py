
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


def nn_fun(x):
    out = x @ w1.transpose() + b1
    out = np.tanh(out)
    out = out @ w2.transpose() + b2
    return out

def simulate_wh():

    n_in = 1
    n_out = 1
    n_hidden = 32
    n_skip = 200
    nx = 5
    random_order = True
    system_rng = np.random.default_rng(None)
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

    return G1, G2, w1, b1, w2, b2




if __name__ == "__main__":
    # Generate random forced inputs for simulation
    ts = 1e-2
    T = 5
    t = np.arange(0, T, ts)
    u = np.random.normal(0, 1000, t.shape)


    # Simulate the system trajectory using the model
    G1, G2, w1, b1, w2, b2 = simulate_wh()

    s = tf('s')
    tau = 1 # s
    M = 1 / (1 + (tau / (2 * np.pi)) * s)
    M = M * (1 + 1e-2 * (tau / (2 * np.pi)) * s)  # add a high freq zero for inversion

    u = u.reshape(-1, 1)
    print(u.shape)
    # G1
    y1 = dlsim(*G1, u)
    y1 = (y1 - y1[:].mean(axis=0)) / (y1[:].std(axis=0) + 1e-6)
    print(y1.shape)
    # F
    y2 = nn_fun(y1)
    print(y2.shape)
    # G2
    y3 = dlsim(*G2, y2)
    print(y3.shape)
    r_v = lsim(M ** (-1), y3, t)[0]
    r_v= r_v.reshape(-1, 1)
    e_v = (r_v - y3).reshape(-1, 1)
    print(r_v.shape)
    print(e_v.shape)

    plt.subplot(211)
    plt.plot(t, u)
    plt.legend(['u'])
    plt.subplot(212)
    plt.plot(t, e_v)
    plt.legend(['e_v'])
    plt.show()




