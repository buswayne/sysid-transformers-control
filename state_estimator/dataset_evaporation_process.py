import math
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import control  # pip install python-control, pip install slycot (optional)
from lti import drss_matrices, dlsim
from evaporation_process import simulate_evaporation_process
import matplotlib.pyplot as plt
from utils import prbs


class EvaporationProcessDataset(IterableDataset):
    def __init__(self, nx=2, nu=3, ny=2, seq_len=600, random_order=True,
                 strictly_proper=True, normalize=False, dtype="float32",
                 fixed_system=False, system_seed=None, data_seed=None, **mdlargs):
        super(EvaporationProcessDataset).__init__()
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

        n_skip = 200

        while True:  # infinite dataset

            u = np.zeros((self.seq_len + n_skip, 3))

            u_s = np.array([191.713, 215.888])

            #u[:,:2] = 5 * self.data_rng.normal(size=(self.seq_len + n_skip, 2)) + u_s # input signal at time t

            u[:, 0] = prbs(self.seq_len + n_skip) + u_s[0]
            u[:, 1] = prbs(self.seq_len + n_skip) + u_s[1]
            #u = np.random.randn(self.seq_len + n_skip, 2)  # input to be improved (filtered noise, multisine, etc)
            # u = torch.from_numpy(self.data_rng.normal(size=(self.seq_len + n_skip, 2))).to(torch.float32).to('cuda') # cstr has two inputs u1, u2

            # System
            # this is both y1 and y2 (at time t)

            x_n, x, y = simulate_evaporation_process(u, perturbation=0.2)

            # input to the model are (u1, u2, y1) at time t, output of the model is (y2 at time t)
            u[:,2] = y[:,1]

            # output to the model are (x1, x2) noiseless
            y = x_n.reshape(-1, 2)
            #y = x_n[:,0].reshape(-1,1)

            # print(y.dtype)
            # print(u.dtype)

            # u = u[n_skip:]
            # y = y[n_skip:]

            # print(u.shape)
            # print(y.shape)
            # print('')

            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)


            # plt.subplot(211)
            # plt.plot(y)
            # plt.subplot(212)
            # plt.plot(u)
            # plt.show()

            u = u.astype(self.dtype)
            y = y.astype(self.dtype)
            #
            yield torch.tensor(y), torch.tensor(u)
            #yield y, u#torch.tensor(u)

            
if __name__ == "__main__":
    # train_ds = WHDataset(nx=2, seq_len=32, mag_range=(0.5, 0.96),
    #                      phase_range=(0, math.pi / 3),
    #                      system_seed=42, data_seed=445, fixed_system=False)
    train_ds = EvaporationProcessDataset(nx=2, seq_len=600, system_seed=42, data_seed=445, fixed_system=False)
    # train_ds = LinearDynamicalDataset(nx=5, nu=2, ny=3, seq_len=1000)
    train_dl = DataLoader(train_ds, batch_size=32)
    batch_y, batch_u = next(iter(train_dl))
    print(batch_u.shape)
    print(batch_y.shape)
    plt.subplot(211)
    plt.plot(batch_u[:,:,2].squeeze().T)
    plt.ylabel('$y_1$')
    plt.subplot(212)
    plt.plot(batch_y[:,:,0].squeeze().T)
    plt.ylabel('$y_2$')
    plt.show()

    plt.subplot(211)
    plt.plot(batch_u[:,:,0].squeeze().T)
    plt.ylabel('$u_1$')
    plt.subplot(212)
    plt.plot(batch_u[:,:,1].squeeze().T)
    plt.ylabel('$u_2$')
    plt.show()
    #print(batch_y.shape, batch_u.shape)
