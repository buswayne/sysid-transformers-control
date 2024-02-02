import numpy as np
from control.matlab import *
def prbs(nstep):
    #a_range = [0, 2]
    #a = np.random.rand(nstep) * (a_range[1] - a_range[0]) + a_range[0]  # range for amplitude

    b_range = [50, 500]
    b = np.random.rand(nstep) * (b_range[1] - b_range[0]) + b_range[0]  # range for frequency
    b = np.round(b)
    b = b.astype(int)

    b[0] = 0

    for i in range(1, np.size(b)):
        b[i] = b[i - 1] + b[i]

    # PRBS
    a = np.zeros(nstep)
    j = 0
    while j < nstep:
        a[j] = 10
        a[j + 1] = -10
        j = j + 2

    i = 0
    prbs = np.zeros(nstep)
    while b[i] < np.size(prbs):
        k = b[i]
        prbs[k:] = a[i]
        i = i + 1
    return prbs

def simulate_onestep_campi_example_1(data, u, t):

    s = tf('s')

    P = ((data['m1'] * s ** 2 + (data['c1'] + data['c2']) * s + (data['k1'] + data['k2'])) /
        ((data['m1'] * s ** 2 + (data['c1'] + data['c2']) * s + (data['k1'] + data['k2'])) *
         (data['m2'] * s ** 2 + data['c2'] * s + data['k2']) - (data['k2'] + data['c2'] * s) ** 2))

    y, _, _ = lsim(P, u, t)

    return y
