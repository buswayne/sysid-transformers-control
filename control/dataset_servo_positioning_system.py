import math
import time

import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import control  # pip install python-control, pip install slycot (optional)
from lti import drss_matrices, dlsim
from servo_positioning_system_chatgpt import simulate_servo_positioning_system
import matplotlib.pyplot as plt
from utils import prbs
from control.matlab import *


class ServoPositioningSystemDataset(IterableDataset):
    def __init__(self, nx=3, nu=1, ny=3, seq_len=1e6, random_order=True,
                 strictly_proper=True, normalize=False, dtype="float32",
                 fixed_system=False, system_seed=None, data_seed=None, **mdlargs):
        super(ServoPositioningSystemDataset).__init__()
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

        # Call the function to generate data

        Ts = 1e-4 # must be small otherwise RK diverges
        Ts_f = 1e-2
        T_f = Ts_f * self.seq_len

        t = np.arange(0, T_f, Ts)
        t_final = np.arange(0, T_f, Ts_f)

        while True:  # infinite dataset

            u_s = np.array([1e-3]) # optional offset, set to zero ftb
            u = np.zeros((len(t), self.nu))
            u[:, 0] = prbs(len(t)) + u_s[0]

            # System
            x, u, y, _ = simulate_servo_positioning_system(Ts, u, perturbation=0.25)

            # input to the model are (e_v) at time t, output of the model is (u)
            #             __________________________
            #  e_v ----->|   Transformer-based      |-----> u
            #            |        Controller        |
            #            |__________________________|


            # Desired variable to be controlled is x1 = \theta. Let's compute virtual error
            s = tf('s')
            tau = 5  # 2 sec t. ass
            M = 1 / (1 + (tau / (2 * np.pi)) * s)
            M = c2d(M, Ts)
            z = tf('z')

            u_downsampled = u[::int(Ts_f / Ts)]
            y_downsampled = y[::int(Ts_f/Ts), 0]
            r_v = lsim((z * M) ** (-1), y_downsampled, t[::int(Ts_f/Ts)])[0]
            r_v = r_v[:-1] # multiply by z^-1
            e_v = (r_v - y_downsampled[:-1]).reshape(-1,1)  # must be 2d
            e_v_integral = np.cumsum(e_v).reshape(-1,1)

            e_v = e_v[1:]
            e_v_integral = e_v_integral[1:]
            u = u_downsampled[:-2]

            e_v = e_v.astype(self.dtype)
            e_v_integral = e_v_integral.astype(self.dtype)
            u = u.astype(self.dtype)
            # we need to shift the vector u: currently is u_t,...u_t+k, it must become u_{t-1},....u_{t+k-1}


            #y = y[::int(Ts_f/Ts)].astype(self.dtype)

            # consider that u is the ouput to be predicted by the transformer, e_v is the input
            # yield torch.tensor(u), torch.tensor(np.concatenate((e_v, e_v_integral),axis=1))#, torch.tensor(y)
            yield torch.tensor(u), torch.tensor(e_v_integral)

if __name__ == "__main__":
    # train_ds = WHDataset(nx=2, seq_len=32, mag_range=(0.5, 0.96),
    #                      phase_range=(0, math.pi / 3),
    #                      system_seed=42, data_seed=445, fixed_system=False)
    start = time.time()
    train_ds = ServoPositioningSystemDataset(nx=3, seq_len=502, system_seed=42, data_seed=445, fixed_system=False)
    # train_ds = LinearDynamicalDataset(nx=5, nu=2, ny=3, seq_len=1000)
    train_dl = DataLoader(train_ds, batch_size=32)
    batch_u, batch_e_v = next(iter(train_dl))
    print(batch_u.shape)
    #print(batch_e_v.shape)
    print(time.time() - start)
    t = np.arange(0, batch_u.shape[1]*1e-2, 1e-2)
    plt.subplot(311)
    # plt.plot(t, batch_y[:, :, 0].squeeze().T)
    # plt.ylabel(r'$y_1 = \theta$')
    # plt.subplot(312)
    # plt.plot(t, batch_y[:, :, 1].squeeze().T)
    plt.ylabel(r'$y_2 = \omega$')
    plt.subplot(313)
    plt.plot(t, batch_e_v[:, :, 0].squeeze().T)
    plt.ylabel('$e_v = (M^{-1} - 1)y_1$')
    plt.show()
    # print(batch_y.shape, batch_u.shape)
