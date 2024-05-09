import math
import time

import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import control  # pip install python-control, pip install slycot (optional)
from lti import drss_matrices, dlsim
from simple_example_1 import simulate_simple_example_1
import matplotlib.pyplot as plt
from utils import prbs
from control.matlab import *


class SimpleExample1Dataset(IterableDataset):
    def __init__(self, seq_len=1e6, normalize=False, dtype="float32", **mdlargs):
        super(SimpleExample1Dataset).__init__()
        self.seq_len = seq_len
        self.dtype = dtype
        self.normalize = normalize
        self.mdlargs = mdlargs

    def __iter__(self):

        # Call the function to generate data
        ts = 1e-2
        T = ts*self.seq_len
        T = 5
        t = np.arange(0, T, ts)

        while True:  # infinite dataset

            u = np.random.normal(0, 10, t.shape)
            # System
            x, u, y = simulate_simple_example_1(t, u, perturbation=0.8)

            # Desired variable to be controlled is x1 = \theta. Let's compute virtual error
            s = tf('s')
            tau = 0.05  # s
            M = 1 / (1 + (tau / (2 * np.pi)) * s)
            M = M * (1 + 1e-2 * (tau / (2 * np.pi)) * s)  # add a high freq zero for inversion

            # get virtual error
            r_v = lsim(M ** (-1), y, t)[0]
            e_v = r_v - y

            e_v = e_v.astype(self.dtype)
            u = u.astype(self.dtype)#.reshape(-1,1)
            y = y.astype(self.dtype)#.reshape(-1,1)

            e_1 = e_v[1:].flatten()  #
            e_2 = e_v[:-1].flatten()  #

            input_vector = np.stack((e_1,e_2),axis=1)
            output_vector = u[1:].reshape(-1,1)

            yield torch.tensor(input_vector), torch.tensor(output_vector)

if __name__ == "__main__":
    # train_ds = WHDataset(nx=2, seq_len=32, mag_range=(0.5, 0.96),
    #                      phase_range=(0, math.pi / 3),
    #                      system_seed=42, data_seed=445, fixed_system=False)
    # start = time.time()
    train_ds = SimpleExample1Dataset(seq_len=500)
    # train_ds = LinearDynamicalDataset(nx=5, nu=2, ny=3, seq_len=1000)
    train_dl = DataLoader(train_ds, batch_size=32)
    #batch_u, batch_e_v, batch_y = next(iter(train_dl))
    batch_input, batch_output = next(iter(train_dl))

    # print(batch_input.shape)
    # print(batch_output.shape)

    #plt.plot(batch_input[0,:,0])
    for i in range(0,batch_output.shape[0]):
        print(i)
        plt.plot(batch_input[i, :, 0], c='tab:blue', alpha=0.2)
    plt.show()
    # print(batch_u.shape)
    # print(batch_e_v.shape)
    #t = np.arange(0, batch_u.shape[1]*1e-2, 1e-2)
    #plt.subplot(311)
    #plt.plot(t, batch_y[:, :, 0].squeeze().T)
    #plt.ylabel(r'$y_1 = \theta$')
    #plt.subplot(312)
    #plt.plot(t, batch_y[:, :, 1].squeeze().T)
    #plt.ylabel(r'$y_2 = \omega$')
    #plt.subplot(313)
    #plt.plot(t, batch_e_v[:, :, 0].squeeze().T)
    #plt.ylabel('$e_v = (M^{-1} - 1)y_1$')
    #plt.show()
    # print(batch_y.shape, batch_u.shape)
