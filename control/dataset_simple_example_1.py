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
from scipy.interpolate import interp1d

class SimpleExample1Dataset(IterableDataset):
    def __init__(self, seq_len=1e6, normalize=False, dtype="float32", return_y=False):
        super(SimpleExample1Dataset).__init__()
        self.seq_len = seq_len
        self.dtype = dtype
        self.normalize = normalize
        self.return_y = return_y

    def __iter__(self):

        # Call the function to generate data
        ts = 1e-2
        T = 20#ts*self.seq_len# * 2
        t = np.arange(0, T, ts)

        n_context = self.seq_len

        while True:  # infinite dataset

            # prbs instead
            # random
            n_steps = np.random.randint(2, 50)
            u = np.random.normal(0, 1000, t.shape)

            f = interp1d(t[::n_steps], u[::n_steps], kind='next',
                         bounds_error=False,
                         fill_value=0.0)
            #u = f(t)
            u = np.nan_to_num(u)
            #print(np.isnan(u).sum())
            # System
            x, u, y = simulate_simple_example_1(t, u, perturbation=0.2)

            # Desired variable to be controlled is x1 = \theta. Let's compute virtual error
            s = tf('s')
            tau = 0.05  # s
            M = 1 / (1 + (tau / (2 * np.pi)) * s)
            M = M * (1 + 1e-2 * (tau / (2 * np.pi)) * s)  # add a high freq zero for inversion
            # get virtual error
            r_v = lsim(M ** (-1), y, t)[0]
            e_v = (r_v - y).reshape(-1, 1)  # must be 2d
            u = u.reshape(-1, 1)
            # e_v_integral = np.cumsum(e_v).reshape(-1,1)

            e_v = e_v.astype(self.dtype)
            # e_v_integral = e_v_integral.astype(self.dtype)
            u = np.insert(u, 0, 1e-6)
            u = u[:-1].astype(self.dtype)
            y = y.astype(self.dtype)

            # lunghezza contesto 5
            #start_idx = np.random.randint(0, len(e_v)-n_context)
            start_idx = 0
            e_v = e_v[start_idx:start_idx + n_context]
            u = u[start_idx:start_idx + n_context]
            y = y[start_idx:start_idx + n_context]

            # e_1 = e_v[1:].flatten()  #
            # e_2 = e_v[:-1].flatten()  #

            if self.normalize:
                e_v = e_v / 6  # mean 0, std 10
                u = u / 1000  # mean 0, std 17
                # e_v = (e_v - e_v.mean(axis=0)) / (e_v.std(axis=0) + 1e-6)
                # u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-6)
            # input_vector = np.stack((e_1,e_2),axis=1)
            input_vector = e_v.reshape(-1, 1)
            output_vector = u.reshape(-1, 1)
            y = y.reshape(-1, 1)

            if self.return_y:
                yield torch.tensor(output_vector), torch.tensor(input_vector), torch.tensor(y)
            else:
                yield torch.tensor(output_vector), torch.tensor(input_vector)


if __name__ == "__main__":
    # train_ds = WHDataset(nx=2, seq_len=32, mag_range=(0.5, 0.96),
    #                      phase_range=(0, math.pi / 3),
    #                      system_seed=42, data_seed=445, fixed_system=False)
    # start = time.time()
    train_ds = SimpleExample1Dataset(seq_len=500, normalize=True)
    train_dl = DataLoader(train_ds, batch_size=32)
    batch_output, batch_input = next(iter(train_dl))

    print(batch_output.shape)

    print(batch_output[:, :, 0].mean())
    print(batch_output[:, :, 0].std())
    print(batch_input[:, :, 0].mean())
    print(batch_input[:, :, 0].std())

    plt.figure()
    #plt.plot(batch_input[0,:,0])
    Ts = 1e-2
    T = batch_input.shape[1]*Ts  # ts*self.seq_len# * 2
    t = np.arange(0, T, Ts)

    for i in range(0,batch_output.shape[0]):
        plt.subplot(211)
        plt.plot(t, batch_input[i, :, 0], c='tab:blue', alpha=0.2)
        plt.legend(['$e_v$'])
        plt.subplot(212)
        plt.plot(t, batch_output[i, :, 0], c='tab:blue', alpha=0.2)
        plt.legend(['$u$'])
        plt.xlabel("$t$ [s]")
    plt.show()