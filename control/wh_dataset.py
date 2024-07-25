import math
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import control  # pip install python-control, pip install slycot (optional)
from lti import drss_matrices, dlsim
from control.matlab import *
from matplotlib import pyplot as plt

class WHDataset(IterableDataset):
    def __init__(self, nx=2, nu=1, ny=1, seq_len=600, random_order=True,
                 strictly_proper=True, normalize=True, dtype="float32",
                 fixed_system=False, system_seed=None, data_seed=None, return_y=False, signal="white noise", use_prefilter=False, **mdlargs):
        super(WHDataset).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.seq_len = seq_len
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize
        self.strictly_proper = strictly_proper
        self.random_order = random_order  # random number of states from 1 to nx
        self.system_seed = system_seed
        self.data_seed = data_seed
        self.system_rng = np.random.default_rng(system_seed)  # source of randomness for model generation
        self.data_rng = np.random.default_rng(data_seed)  # source of randomness for model generation
        self.fixed_system = fixed_system  # same model at each iteration (classical identification)
        self.mdlargs = mdlargs

        self.ts = 1e-2
        self.T = 20  # ts*self.seq_len# * 2
        self.t = np.arange(0, self.T, self.ts)
        self.return_y = return_y
        self.signal = signal
        self.use_prefilter = use_prefilter

    def __iter__(self):

        n_context = self.seq_len

        # A simple ff neural network
        def nn_fun(x):
            out = x @ w1.transpose() + b1
            out = np.tanh(out)
            out = out @ w2.transpose() + b2
            return out

        n_in = 1
        n_out = 1
        n_hidden = 32

        if self.fixed_system:  # same model at each step, generate only once!
            w1 = self.system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
            b1 = self.system_rng.normal(size=(1, n_hidden)) * 1.0
            w2 = self.system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
            b2 = self.system_rng.normal(size=(1, n_out)) * 1.0

            G1 = drss_matrices(states=self.system_rng.integers(1, self.nx + 1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=self.strictly_proper,
                               rng=self.system_rng,
                               **self.mdlargs)

            G2 = drss_matrices(states=self.system_rng.integers(1, self.nx + 1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=False,
                               rng=self.system_rng,
                               **self.mdlargs)

        while True:  # infinite dataset

            if not self.fixed_system:  # different model for different instances!
                w1 = self.system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
                b1 = self.system_rng.normal(size=(1, n_hidden)) * 1.0
                w2 = self.system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
                b2 = self.system_rng.normal(size=(1, n_out)) * 1.0

                G1 = drss_matrices(states=self.system_rng.integers(1, self.nx + 1) if self.random_order else self.nx,
                                   inputs=1,
                                   outputs=1,
                                   strictly_proper=self.strictly_proper,
                                   rng=self.system_rng,
                                   **self.mdlargs)

                G2 = drss_matrices(states=self.system_rng.integers(1, self.nx + 1) if self.random_order else self.nx,
                                   inputs=1,
                                   outputs=1,
                                   strictly_proper=False,
                                   rng=self.system_rng,
                                   **self.mdlargs)

            # u = np.random.randn(self.seq_len, 1)  # input to be improved (filtered noise, multisine, etc)
            u = self.data_rng.normal(loc=0, scale=100, size=(len(self.t), 1))

            # G1
            y1 = dlsim(*G1, u)
            y1 = (y1 - y1.mean(axis=0)) / (y1.std(axis=0) + 1e-6)

            # F
            y2 = nn_fun(y1)

            # G2
            y3 = dlsim(*G2, y2)

            u = u
            y = y3

            #### New
            s = tf('s')
            z = tf([1, 0], [1], dt=self.ts)
            tau = 1  # s
            M = 1 / (1 + (tau / (2 * np.pi)) * s)
            M = c2d(M, self.ts, 'tustin')

            L = M

            if self.use_prefilter:
                u_L = lsim(L, u, self.t)[0]
                y_L = lsim(L, y, self.t)[0]
            else:
                u_L = u
                y_L = y

            r_v = (lsim(M ** (-1), y_L, self.t)[0]).reshape(-1, 1)  # r(1), ... r(T)
            e_v = (r_v - y_L).reshape(-1, 1)  # e(1), ... e(T)

            if self.normalize:
                if self.signal == 'white noise' and not self.use_prefilter:
                    r_std = 33
                    e_std = 6.15
                    u_std = 1000
                    y_std = 26.8
                elif self.signal == 'white noise' and self.use_prefilter:
                    r_std = 26.8
                    e_std = 0.45
                    u_std = 17
                    y_std = 2.5
                elif self.signal == 'prbs' and not self.use_prefilter:
                    e_std = 4.22
                    u_std = 100
                elif self.signal == 'prbs' and self.use_prefilter:
                    e_std = 4
                    u_std = 88
                elif self.signal == 'random' and not self.use_prefilter:
                    e_std = 2.6
                    u_std = 103
                elif self.signal == 'random' and self.use_prefilter:
                    e_std = 2.27
                    u_std = 67.5
                r_v = r_v / r_std
                e_v = e_v / e_std  # 2.49#4.8320#2.22  # mean 0, std 10
                u_L = u_L / u_std  #
                y_L = y_L / y_std

            e_v = e_v[1:].astype(self.dtype)
            u_L = u_L[:-1].astype(self.dtype)
            y_L = y_L[1:].astype(self.dtype)
            r_v = r_v[1:].astype(self.dtype)

            e_v = e_v.astype(self.dtype)
            u_L = u_L.astype(self.dtype)
            y_L = y_L.astype(self.dtype)
            r_v = r_v.astype(self.dtype)

            # lunghezza contesto 5

            start_idx = 0
            e_v = e_v[start_idx:start_idx + n_context]
            u_L = u_L[start_idx:start_idx + n_context]
            y_L = y_L[start_idx:start_idx + n_context]
            r_v = r_v[start_idx:start_idx + n_context]

            e_v = e_v.reshape(-1, 1)
            u_L = u_L.reshape(-1, 1)
            y_L = y_L.reshape(-1, 1)
            r_v = r_v.reshape(-1, 1)

            if self.return_y:
                yield torch.tensor(y_L), torch.tensor(u_L), torch.tensor(e_v)
            else:
                yield torch.tensor(u_L), torch.tensor(e_v)#, torch.tensor(r_v)

def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    dataset.data_rng = np.random.default_rng(dataset.data_seed + 1000 * worker_id)
    dataset.system_rng = np.random.default_rng(dataset.system_seed + 1000 * worker_id)
    # print(worker_id, worker_info.id)


if __name__ == "__main__":
    train_ds = WHDataset(seq_len=500, system_seed=42, data_seed=445, return_y=True, fixed_system=True, normalize=True, use_prefilter=True)
    # train_ds = LinearDynamicalDataset(nx=5, nu=2, ny=3, seq_len=1000)
    train_dl = DataLoader(train_ds, batch_size=12, num_workers=10, worker_init_fn=seed_worker)
    batch_y, batch_u, batch_e = next(iter(train_dl))

    print(batch_u.shape)

    print('y mean:', batch_y[:, :, 0].mean())
    print('y std:', batch_y[:, :, 0].std())
    print('u mean:', batch_u[:, :, 0].mean())
    print('u std:', batch_u[:, :, 0].std())
    print('e mean:', batch_e[:, :, 0].mean())
    print('e std:', batch_e[:, :, 0].std())

    plt.figure(figsize=(7, 5))
    # plt.plot(batch_input[0,:,0])
    Ts = 1e-2
    T = batch_u.shape[1] * Ts  # ts*self.seq_len# * 2
    t = np.arange(0, T, Ts)

    for i in range(0, batch_u.shape[0]):
        plt.subplot(311)
        plt.plot(t, batch_u[i, :, 0], c='tab:blue', alpha=0.2)
        # plt.legend(['$e_v$'])
        plt.ylabel("$u_L$")
        plt.tick_params('x', labelbottom=False)

        plt.subplot(312)
        plt.plot(t, batch_y[i, :, 0], c='tab:blue', alpha=0.2)
        # plt.legend(['$u$'])
        plt.ylabel("$y_L$")
        plt.xlabel("$t$ [s]")

        plt.subplot(313)
        plt.plot(t, batch_e[i, :, 0], c='tab:blue', alpha=0.2)
        # plt.legend(['$y$'])
        plt.ylabel("$e_v$")
        plt.tick_params('x', labelbottom=False)

    plt.show()