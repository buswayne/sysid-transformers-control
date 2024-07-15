import copy
import math
import time

import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset, Dataset
import control  # pip install python-control, pip install slycot (optional)
from lti import drss_matrices, dlsim
from simple_example_1 import simulate_simple_example_1
import matplotlib.pyplot as plt
from utils import prbs, random_signal
from control.matlab import *
from scipy.interpolate import interp1d
from u_estimate import u_estimate

import warnings

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.xmargin'] = 0

# Disable all user warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

# Re-enable user warnings
warnings.filterwarnings("default")


class SimpleExample1Dataset(Dataset):
    def __init__(self, seq_len=1e6, normalize=False, dtype="float32", signal='white noise', use_prefilter=False, fixed=True):
        super(SimpleExample1Dataset, self).__init__()
        self.seq_len = seq_len
        self.dtype = dtype
        self.normalize = normalize
        self.signal = signal
        self.use_prefilter = use_prefilter
        self.fixed = fixed
        self.data = None
        self.ts = 1e-2
        self.T = 20
        self.t = np.arange(0, self.T, self.ts)

        if self.fixed:
            self.data = self._generate_data()

    def __len__(self):
        return len(self.data[0]) if self.fixed else 32

    def _generate_data(self):
        data_y = []
        data_u = []
        data_e = []

        for _ in range(32):  # Adjust the range based on the desired size of the fixed dataset
            y, u, e = self._generate_sample()
            data_y.append(y)
            data_u.append(u)
            data_e.append(e)

        return torch.stack(data_y), torch.stack(data_u), torch.stack(data_e)

    def _generate_sample(self):
        # np.random.seed(42)

        t = self.t

        choice = self.signal

        if choice == "white noise":
            n_steps = np.random.randint(2, 50)
            u = np.random.normal(0, 1000, t.shape)
            f = interp1d(t[::n_steps], u[::n_steps], kind='next',
                         bounds_error=False,
                         fill_value=0.0)
        elif choice == "prbs":
            u = prbs(len(t))
        elif choice == "random":
            u = random_signal(len(t))

        u, y = simulate_simple_example_1(t, u, perturbation=0.0)

        s = tf('s')
        z = tf([1, 0], [1], dt=self.ts)
        tau = 1
        M = 1 / (1 + (tau / (2 * np.pi)) * s)
        M = c2d(M, self.ts, 'matched')
        W = 1 / (1 + (0.1 * tau / (2 * np.pi)) * s)
        W = c2d(W, self.ts, 'matched')

        L = M

        M_proper = z * M

        if self.use_prefilter:
            u_L = lsim(L, u, t)[0]
            y_L = lsim(L, y, t)[0]
        else:
            u_L = u
            y_L = y

        r_v = lsim(M_proper ** (-1), y_L, t)[0]
        r_v = np.insert(r_v[:-1], 0, 0)

        e_v = (r_v - y_L).reshape(-1, 1)

        u_L = u_L.reshape(-1, 1)

        if self.normalize:
            if self.signal == 'white noise' and not self.use_prefilter:
                e_std = 6.15
                u_std = 1000
            elif self.signal == 'white noise' and self.use_prefilter:
                e_std = 4.5
                u_std = 177
                y_std = 25
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

            e_v = e_v / e_std
            u_L = u_L / u_std
            y_L = y_L / y_std

        e_v = e_v.astype(self.dtype)
        u_L = np.insert(u_L, 0, 0)
        u_L = u_L[:-1].astype(self.dtype)
        y_L = y_L.astype(self.dtype)
        r_v = r_v.astype(self.dtype)

        start_idx = np.random.randint(0, len(e_v) - self.seq_len)
        e_v = e_v[start_idx:start_idx + self.seq_len]
        u_L = u_L[start_idx:start_idx + self.seq_len]
        y_L = y_L[start_idx:start_idx + self.seq_len]
        r_v = r_v[start_idx:start_idx + self.seq_len]

        e = e_v.reshape(-1, 1)
        u = u_L.reshape(-1, 1)
        y = y_L.reshape(-1, 1)
        r = r_v.reshape(-1, 1)

        return torch.tensor(y), torch.tensor(u), torch.tensor(e)

    def __getitem__(self, idx):
        if self.fixed:
            return self.data[0][idx], self.data[1][idx], self.data[2][idx]
        else:
            return self._generate_sample()


if __name__ == "__main__":
    train_ds = SimpleExample1Dataset(seq_len=500, normalize=True, signal='white noise', use_prefilter=True, fixed=True)
    train_dl = DataLoader(train_ds, batch_size=32)
    batch_y, batch_u, batch_e = next(iter(train_dl))

    print(batch_y.shape)
    print(batch_y.dtype)
    # print(batch_u.shape)
    # print(batch_y[3,8,0])

    # batch_y, batch_u, batch_e = next(iter(train_dl))
    # print(batch_y[3, 8, 0])
    # print('y mean:', batch_y[:, :, 0].mean())
    # print('y std:', batch_y[:, :, 0].std())
    # print('u mean:', batch_u[:, :, 0].mean())
    # print('u std:', batch_u[:, :, 0].std())
    # print('e mean:', batch_e[:, :, 0].mean())
    # print('e std:', batch_e[:, :, 0].std())

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
        # plt.legend(['$y$'])
        plt.ylabel("$y_L$")
        plt.tick_params('x', labelbottom=False)

        plt.subplot(313)
        plt.plot(t, batch_e[i, :, 0], c='tab:blue', alpha=0.2)
        # plt.legend(['$u$'])
        plt.ylabel("$e_v$")
        plt.xlabel("$t$ [s]")
    plt.show()
