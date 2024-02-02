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
    def __init__(self, nx=4, nu=1, ny=1, seq_len=1e6, random_order=True,
                 strictly_proper=True, normalize=False, dtype="float32",
                 fixed_system=False, system_seed=None, data_seed=None, **mdlargs):
        super(CampiExample1Dataset).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.seq_len = seq_len
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize
        self.strictly_proper = strictly_proper
        self.random_order = random_order  # random number of states from 1 to nx
        self.system_rng = np.random.default_rng(system_seed)  # source of randomness for model generation
        self.data_rng = np.random.default_rng(data_seed)  # source of randomness for model generation
        self.fixed_system = fixed_system  # same model at each iteration (classical identification)
        self.mdlargs = mdlargs

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

            # lunghezza contesto 5
            start_idx = np.random.randint(0, len(e_v)-n_context)

            e_v = e_v[start_idx:start_idx+n_context]
            u = u[start_idx:start_idx + n_context]


            # consider that u is the ouput to be predicted by the transformer, e_v is the input
            # yield torch.tensor(u), torch.tensor(np.concatenate((e_v, e_v_integral),axis=1))#, torch.tensor(y)
            yield torch.tensor(u), torch.tensor(e_v)

if __name__ == "__main__":
    # train_ds = WHDataset(nx=2, seq_len=32, mag_range=(0.5, 0.96),
    #                      phase_range=(0, math.pi / 3),
    #                      system_seed=42, data_seed=445, fixed_system=False)
    start = time.time()
    train_ds = CampiExample1Dataset(nx=3, seq_len=100, system_seed=42, data_seed=445, fixed_system=False)
    # train_ds = LinearDynamicalDataset(nx=5, nu=2, ny=3, seq_len=1000)
    train_dl = DataLoader(train_ds, batch_size=160)
    batch_u, batch_e_v = next(iter(train_dl))

    # print(batch_u.shape)
    # print(batch_e_v.shape)
    #print(time.time() - start)

    plt.subplot(211)
    plt.plot(batch_u[:, :, 0].squeeze().T)
    plt.ylabel(r'$u$')

    plt.subplot(212)
    plt.plot(batch_e_v[:, :, 0].squeeze().T)
    plt.ylabel('$e_v$')
    plt.show()
    # print(batch_y.shape, batch_u.shape)
