import math
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import control  # pip install python-control, pip install slycot (optional)
from lti import drss_matrices, dlsim
from campi_example_1 import simulate_campi_example_1
import matplotlib.pyplot as plt
from utils import prbs
from control.matlab import *


class CampiExample1Dataset(IterableDataset):
    def __init__(self, seq_len=1e6, normalize=False, dtype="float32", return_y=False):
        super(CampiExample1Dataset).__init__()
        self.seq_len = seq_len
        self.dtype = dtype
        self.normalize = normalize
        self.return_y = return_y

    def __iter__(self):

        while True:  # infinite dataset

            n_context = self.seq_len

            # System
            t, x, u, y, data = simulate_campi_example_1(perturbation=0.0)

            # Desired variable to be controlled is x1 = \theta. Let's compute virtual error
            s = tf('s')
            M = (1e-1*s + 1) / (3*s + 1)

            r_v = lsim(M**(-1), y, t)[0]

            e_v = (r_v - y).reshape(-1,1)  # must be 2d
            u = u.reshape(-1,1)
            #e_v_integral = np.cumsum(e_v).reshape(-1,1)

            e_v = e_v[1:].astype(self.dtype)
            #e_v_integral = e_v_integral.astype(self.dtype)
            u = u[:-1].astype(self.dtype)
            y = y[1:].astype(self.dtype)

            # lunghezza contesto 5
            # start_idx = np.random.randint(0, len(e_v)-n_context)
            start_idx = 0
            e_v = e_v[start_idx:start_idx+n_context]
            u = u[start_idx:start_idx + n_context]
            y = y[start_idx:start_idx + n_context]

            # e_1 = e_v[1:].flatten()  #
            # e_2 = e_v[:-1].flatten()  #

            if self.normalize:
                e_v = e_v / 17   # mean 0, std 10
                u = u / 10       # mean 0, std 17
                # e_v = (e_v - e_v.mean(axis=0)) / (e_v.std(axis=0) + 1e-6)
                # u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-6)
            #input_vector = np.stack((e_1,e_2),axis=1)
            input_vector = e_v.reshape(-1,1)
            output_vector = u.reshape(-1,1)
            y = y.reshape(-1,1)

            if self.return_y:
                yield torch.tensor(output_vector), torch.tensor(input_vector), torch.tensor(y)
            else:
                yield torch.tensor(output_vector), torch.tensor(input_vector)


if __name__ == "__main__":
    # train_ds = WHDataset(nx=2, seq_len=32, mag_range=(0.5, 0.96),
    #                      phase_range=(0, math.pi / 3),
    #                      system_seed=42, data_seed=445, fixed_system=False)
    start = time.time()
    train_ds = CampiExample1Dataset(seq_len=500, normalize=True)
    # train_ds = LinearDynamicalDataset(nx=5, nu=2, ny=3, seq_len=1000)
    train_dl = DataLoader(train_ds, batch_size=256)
    batch_output, batch_input = next(iter(train_dl))

    print(batch_output[:, :, 0].mean())
    print(batch_output[:, :, 0].std())
    print(batch_input[:, :, 0].mean())
    print(batch_input[:, :, 0].std())

    plt.figure()
    #plt.plot(batch_input[0,:,0])
    for i in range(0,batch_output.shape[0]):
        plt.subplot(211)
        plt.plot(batch_input[i, :, 0], c='tab:blue', alpha=0.2)
        plt.subplot(212)
        plt.plot(batch_output[i, :, 0], c='tab:blue', alpha=0.2)
    plt.show()