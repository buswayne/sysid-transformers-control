import copy
import math
import time

import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
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
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.rcParams['axes.grid']=True
plt.rcParams['axes.xmargin']=0

# Disable all user warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

# Your code goes here

# Re-enable user warnings
warnings.filterwarnings("default")

class SimpleExample1Dataset(IterableDataset):
    def __init__(self, seq_len=1e6, normalize=False, dtype="float32", return_y=False, signal='white noise', use_prefilter=False, noisy=False):
        super(SimpleExample1Dataset).__init__()
        self.seq_len = seq_len # plus one since one time stamp is discarted due to u(t-1)
        self.dtype = dtype
        self.normalize = normalize
        self.return_y = return_y

        # Call the function to generate data
        self.ts = 1e-2
        self.T = 20#ts*self.seq_len# * 2
        self.t = np.arange(0, self.T, self.ts)
        # self.n_steps = np.random.randint(2, 50)
        # self.u_s = np.array([0])  # optional offset, set to zero ftb
        # self.u = np.zeros(self.t.shape)
        # self.u = prbs(len(self.t)) + self.u_s[0]

        self.signal = signal
        self.use_prefilter = use_prefilter
        self.noisy = noisy


        t = self.t

        # self.u = np.random.normal(0, 1000, t.shape)

        # choices = ['white noise', 'prbs', 'random']

        # choice = np.random.choice(choices)

        # choice = self.signal  # "random"
        #
        # if choice == "white noise":
        #     n_steps = np.random.randint(2, 50)
        #     u = np.random.normal(0, 1000, t.shape)
        #     f = interp1d(t[::n_steps], u[::n_steps], kind='next',
        #                  bounds_error=False,
        #                  fill_value=0.0)
        #     # u = f(t)
        # elif choice == "prbs":
        #     u = prbs(len(t))
        # elif choice == "random":
        #     u = random_signal(len(t))

    def __iter__(self):

        # np.random.seed(42)

        t = self.t


        n_context = self.seq_len

        while True:  # infinite dataset

            u = np.random.normal(0, 1000, t.shape)  # self.u

            # System
            u, y = simulate_simple_example_1(t, u, perturbation=0.2)

            if self.noisy:
                y += np.random.normal(0, 2, t.shape)

            #### New
            s = tf('s')
            z = tf([1, 0], [1], dt=self.ts)
            tau = 1  # s
            M = 1 / (1 + (tau / (2 * np.pi)) * s)
            M = c2d(M, self.ts, 'tustin')
            # M = M*(1 + 1e-2*(tau/(2*np.pi))*s) # add a high freq zero for inversion
            # W = 1 / (1 + (0.1 * tau / (2 * np.pi)) * s)
            # W = c2d(W, self.ts, 'tustin')


            # integrator = (self.ts / 2) * (z + 1) / (z - 1)
            # u = lsim(integrator**-1, u, t)[0] # input to be learned without modifying anything
            # first append a zero for -1
            # u = np.diff(u, prepend=0)
            # U = u_estimate(u, self.ts)

            L = M#minreal(minreal((1 - M) * M, verbose=False) * minreal(W * U ** -1, verbose=False), verbose=False)

            if self.use_prefilter:
                u_L = lsim(L, u, t)[0]
                y_L = lsim(L, y, t)[0]
            else:
                u_L = u
                y_L = y

            # u_L = u
            # y_L = y

            r_v = lsim(M**(-1), y_L, t)[0]
            e_v = (r_v - y_L).reshape(-1, 1)  # must be 2d

            u_L = u_L.reshape(-1, 1)
            # e_v_integral = np.cumsum(e_v).reshape(-1,1)

            if self.normalize:
                if self.signal == 'white noise' and not self.use_prefilter:
                    e_std = 6.15
                    u_std = 1000
                elif self.signal == 'white noise' and self.use_prefilter:
                    e_std = 5
                    u_std = 177
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

            e_v = e_v[1:].astype(self.dtype)
            # e_v = np.cumsum(e_v)
            # e_v_integral = e_v_integral.astype(self.dtype)
            # u_L = np.insert(u_L, 0, 0)
            u_L = u_L[:-1].astype(self.dtype)
            y_L = y_L[1:].astype(self.dtype)
            r_v = r_v[1:].astype(self.dtype)

            # lunghezza contesto 5
            start_idx = 0
            # start_idx = np.random.randint(0, len(e_v)-n_context)

            e_v = e_v[start_idx:start_idx + n_context] # this is e(t), ... e(t+N)
            u_L = u_L[start_idx:start_idx + n_context] # this is u(t-1), ... u(t+N-1)
            y_L = y_L[start_idx:start_idx + n_context]
            r_v = r_v[start_idx:start_idx + n_context]

            # input_vector = np.stack((e_1,e_2),axis=1)
            e_v = e_v.reshape(-1, 1)
            u_L = u_L.reshape(-1, 1)
            y_L = y_L.reshape(-1, 1)
            r_v = r_v.reshape(-1, 1)

            if self.return_y:
                yield torch.tensor(y_L), torch.tensor(u_L), torch.tensor(e_v)
            else:
                yield torch.tensor(u_L), torch.tensor(e_v)#, torch.tensor(r_v)


if __name__ == "__main__":
    train_ds = SimpleExample1Dataset(seq_len=300, normalize=True, return_y=True, signal='white noise', use_prefilter=True, noisy=False)
    train_dl = DataLoader(train_ds, batch_size=32)
    batch_y, batch_u, batch_e = next(iter(train_dl))

    print(batch_u.shape)

    print('y mean:',batch_y[:, :, 0].mean())
    print('y std:', batch_y[:, :, 0].std())
    print('u mean:',batch_u[:, :, 0].mean())
    print('u std:', batch_u[:, :, 0].std())
    print('e mean:',batch_e[:, :, 0].mean())
    print('e std:', batch_e[:, :, 0].std())

    plt.figure(figsize=(7,5))
    #plt.plot(batch_input[0,:,0])
    Ts = 1e-2
    T = batch_u.shape[1]*Ts  # ts*self.seq_len# * 2
    t = np.arange(0, T, Ts)

    for i in range(0,batch_u.shape[0]):
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