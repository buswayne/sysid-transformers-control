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
    def __init__(self, seq_len=1e6, normalize=False, dtype="float32", signal='white noise', use_prefilter=False):
        super(SimpleExample1Dataset).__init__()
        self.seq_len = seq_len
        self.dtype = dtype
        self.normalize = normalize

        # Call the function to generate data
        self.ts = 1e-2
        self.T = 20#ts*self.seq_len# * 2
        self.t = np.arange(0, self.T, self.ts)

        self.signal = signal
        self.use_prefilter = use_prefilter

    def __iter__(self):

        np.random.seed(42)

        n_context = self.seq_len

        while True:  # infinite dataset

            t = self.t

            # choices = ['white noise', 'prbs', 'random']

            # choice = np.random.choice(choices)

            choice = self.signal#"random"

            if choice == "white noise":
                n_steps = np.random.randint(2, 50)
                u = np.random.normal(0, 1000, t.shape)
                f = interp1d(t[::n_steps], u[::n_steps], kind='next',
                             bounds_error=False,
                             fill_value=0.0)
                # u = f(t)
            elif choice == "prbs":
                u = prbs(len(t))
            elif choice == "random":
                u = random_signal(len(t))

            # u = np.nan_to_num(u)
            #print(np.isnan(u).sum())
            # System
            u, y = simulate_simple_example_1(t, u, perturbation=0.0)    # u(1), .. u(T), y(1), .. y(T)


            #### New
            s = tf('s')
            z = tf([1, 0], [1], dt=self.ts)
            tau = 1  # s
            M = 1 / (1 + (tau / (2 * np.pi)) * s)
            M = c2d(M, self.ts, 'tustin')
            # M = M*(1 + 1e-2*(tau/(2*np.pi))*s) # add a high freq zero for inversion
            W = 1 / (1 + (0.1 * tau / (2 * np.pi)) * s)
            W = c2d(W, self.ts, 'tustin')


            # integrator = (self.ts / 2) * (z + 1) / (z - 1)
            # u = lsim(integrator**-1, u, t)[0] # input to be learned without modifying anything
            # first append a zero for -1
            # u = np.diff(u, prepend=0)
            # U = u_estimate(u, self.ts)

            L = M#minreal(minreal((1 - M) * M, verbose=False) * minreal(W * U ** -1, verbose=False), verbose=False)

            # print('M', M)
            # print('W', W)
            # print('U', U)
            # print('L', L)

            # M_proper = z * M

            if self.use_prefilter:
                u_L = lsim(L, u, t)[0]
                y_L = lsim(L, y, t)[0]
            else:
                u_L = u
                y_L = y

            # u_L = u
            # y_L = y

            r_v = lsim(M**(-1), y_L, t)[0] # r(1), ... r(T)
            # r_v = np.insert(r_v[:-1], 0, 0)


            # # Desired variable to be controlled is x1 = \theta. Let's compute virtual error
            # s = tf('s')
            # tau = 0.5  # s
            # M = 1 / (1 + (tau / (2 * np.pi)) * s)
            # M = M * (1 + 1e-2 * (tau / (2 * np.pi)) * s)  # add a high freq zero for inversion
            # get virtual error
            # r_v = lsim(M ** (-1), y, t)[0]

            e_v = (r_v - y_L).reshape(-1, 1)  # e(1), ... e(T)

            u_L = u_L.reshape(-1, 1)
            r_v = r_v.reshape(-1, 1)
            # e_v_integral = np.cumsum(e_v).reshape(-1,1)

            if self.normalize:
                if self.signal == 'white noise' and not self.use_prefilter:
                    r_std = 33
                    e_std = 6.15
                    u_std = 1000
                    y_std = 26.8
                elif self.signal == 'white noise' and self.use_prefilter:
                    r_std = 26.8
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

                r_v = r_v / r_std
                e_v = e_v / e_std#2.49#4.8320#2.22  # mean 0, std 10
                u_L = u_L / u_std#
                y_L = y_L / y_std


            e_v = e_v.astype(self.dtype)
            # e_v = np.cumsum(e_v)
            # e_v_integral = e_v_integral.astype(self.dtype)
            # u_L = np.insert(u_L, 0, 0) # shift by one
            u_L = u_L.astype(self.dtype)
            # y_L = np.insert(y_L, 0, 0) # shift by one
            y_L = y_L.astype(self.dtype)
            r_v = r_v.astype(self.dtype)

            # lunghezza contesto 5
            # start_idx = 0
            start_idx = np.random.randint(0, len(e_v)-n_context)

            # start_idx = 0
            e_v = e_v[start_idx:start_idx + n_context]
            u_L = u_L[start_idx:start_idx + n_context]
            y_L = y_L[start_idx:start_idx + n_context]
            r_v = r_v[start_idx:start_idx + n_context]

            # e_1 = e_v[1:].flatten()  #
            # e_2 = e_v[:-1].flatten()  #

            # input_vector = np.stack((e_1,e_2),axis=1)
            e = e_v.reshape(-1, 1)
            u = u_L.reshape(-1, 1)
            y = y_L.reshape(-1, 1)
            r = r_v.reshape(-1, 1)

            yield torch.tensor(y), torch.tensor(u), torch.tensor(r_v)#, torch.tensor(r_v)


if __name__ == "__main__":
    # train_ds = WHDataset(nx=2, seq_len=32, mag_range=(0.5, 0.96),
    #                      phase_range=(0, math.pi / 3),
    #                      system_seed=42, data_seed=445, fixed_system=False)
    # start = time.time()
    train_ds = SimpleExample1Dataset(seq_len=500, normalize=True, signal='white noise', use_prefilter=True)
    train_dl = DataLoader(train_ds, batch_size=64)
    batch_y, batch_u, batch_r = next(iter(train_dl))

    print(batch_r.shape)

    # print(batch_output.shape)
    # print(batch_input.shape)
    print('y mean:',batch_y[:, :, 0].mean())
    print('y std:', batch_y[:, :, 0].std())
    print('u mean:',batch_u[:, :, 0].mean())
    print('u std:', batch_u[:, :, 0].std())
    print('e mean:',batch_r[:, :, 0].mean())
    print('e std:', batch_r[:, :, 0].std())

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
        # plt.legend(['$y$'])
        plt.ylabel("$y_L$")
        plt.tick_params('x', labelbottom=False)

        plt.subplot(313)
        plt.plot(t, batch_r[i, :, 0], c='tab:blue', alpha=0.2)
        # plt.legend(['$u$'])
        plt.ylabel("$r_v$")
        plt.xlabel("$t$ [s]")
    plt.show()