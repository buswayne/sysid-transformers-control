import math
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import matplotlib.pyplot as plt
import control  # pip install python-control, pip install slycot (optional)
from control.matlab import *
from lti import drss_matrices, dlsim



class WHDataset(IterableDataset):
    def __init__(self, nx=5, nu=1, ny=1, seq_len=600, random_order=True,
                 strictly_proper=True, normalize=True, dtype="float32",
                 fixed_system=False, system_seed=0, data_seed=None,  return_y=False, fixed_u = False, return_system = False,
                  tau =1, **mdlargs):  #system and data seed = None
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
        self.return_y = return_y
        self.tau = tau
        self.fixed_u = fixed_u
        self.return_system = return_system

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
        n_skip = 200

        Ts = 1e-2


        if self.fixed_system:  # same model at each step, generate only once!
            w1 = self.system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
            b1 = self.system_rng.normal(size=(1, n_hidden)) * 1.0
            w2 = self.system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
            b2 = self.system_rng.normal(size=(1, n_out)) * 1.0

            G1 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=self.strictly_proper,
                               rng=self.system_rng,
                               **self.mdlargs)

            G2 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=False,
                               rng=self.system_rng,
                               **self.mdlargs)

            A1 = [*G1[0]]
            B1 = [*G1[1]]
            C1 = [*G1[2]]
            D1 = [*G1[3]]
            A2 = [*G2[0]]
            B2 = [*G2[1]]
            C2 = [*G2[2]]
            D2 = [*G2[3]]


        while True:  # infinite dataset

            if not self.fixed_system:  # different model for different instances!
                w1 = self.system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
                b1 = self.system_rng.normal(size=(1, n_hidden)) * 1.0
                w2 = self.system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
                b2 = self.system_rng.normal(size=(1, n_out)) * 1.0

                G1 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                   inputs=1,
                                   outputs=1,
                                   strictly_proper=self.strictly_proper,
                                   rng=self.system_rng,
                                   **self.mdlargs)

                G2 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                   inputs=1,
                                   outputs=1,
                                   strictly_proper=False,
                                   rng=self.system_rng,
                                   **self.mdlargs)

            #u = np.random.randn(self.seq_len + n_skip, 1)  # input to be improved (filtered noise, multisine, etc)
            #u = self.data_rng.normal(size=(self.seq_len + n_skip, 1))
            #u = self.data_rng.normal(size=(self.seq_len, 1))
            #print(u.shape)

            if self.fixed_u:
                np.random.seed(0)
            #t = np.arange(0, self.seq_len * Ts, Ts)
            #u = np.random.normal(0, 1000, t.shape)
            u = self.data_rng.normal(size=(self.seq_len + n_skip, 1))
            u = u.reshape(-1, 1)

            # G1
            y1 = dlsim(*G1, u)
            y1 = (y1 - y1[n_skip:].mean(axis=0)) / (y1[n_skip:].std(axis=0) + 1e-6)
            #y1 = (y1 - y1[:].mean(axis=0)) / (y1[:].std(axis=0) + 1e-6)

            # F
            y2 = nn_fun(y1)

            # G2
            y3 = dlsim(*G2, y2)

            u = u[n_skip:]
            y = y3[n_skip:]
            #u = u[:]
            #y = y3[:]

            s = tf('s')
            tau = self.tau # s
            M = 1 / (1 + (tau / (2 * np.pi)) * s)
            M = c2d(M, Ts, 'matched')
            M_proper = z*M
            # get virtual error
            t = np.arange(0, len(y) * Ts, Ts)
            r_v = lsim(M_proper**(-1), y,t)[0]
            r_v = r_v.reshape(-1, 1)
            e_v = (r_v - y).reshape(-1, 1)


            u = u.reshape(-1, 1)


            u = u[:-1].astype(self.dtype)
            y= y[1:].astype(self.dtype)
            r_v = r_v[1:].astype(self.dtype)
            e_v = e_v[1:].astype(self.dtype)

            start_idx = 0
            e_v = e_v[start_idx:start_idx + n_context]
            u = u[start_idx:start_idx + n_context]
            y = y[start_idx:start_idx + n_context]

            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)
                e_v = (e_v - e_v.mean(axis=0)) / (e_v.std(axis=0) + 1e-6)
                u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-6)

            output_vector = u.reshape(-1, 1)
            y = y.reshape(-1, 1)
            input_vector = e_v.reshape(-1, 1)


            if self.return_y and not self.return_system :
                yield torch.tensor(output_vector), torch.tensor(input_vector), torch.tensor(y)
            elif self.return_system :
                yield torch.tensor(output_vector), torch.tensor(input_vector), torch.tensor(y), torch.tensor(w1), torch.tensor(b1), \
                    torch.tensor(w2), torch.tensor(b2),  torch.tensor(A1), torch.tensor(B1),  torch.tensor(C1),  torch.tensor(D1), \
                    torch.tensor(A2),  torch.tensor(B2),  torch.tensor(C2),  torch.tensor(D2)
            else :
                yield torch.tensor(output_vector), torch.tensor(input_vector)




if __name__ == "__main__":

    train_ds = WHDataset(seq_len=500, return_y= True, fixed_system=True, return_system=True, tau = 1)
    train_dl = DataLoader(train_ds, batch_size=1)
    batch_u, batch_e_v, batch_y, w1, b1, w2, b2, A1, B1, C1, D1, A2, B2, C2, D2 = next(iter(train_dl))
    #batch_u, batch_e_v, batch_y = next(iter(train_dl))

    print(batch_e_v[:, :, 0].mean())  #without normalization e_v has mean 6.5 and 1.71 std on fixed system
    print(batch_e_v[:, :, 0].std())
    print(batch_u[:, :, 0].mean())
    print(batch_u[:, :, 0].std())
    print(batch_y[:, :, 0].mean())
    print(batch_y[:, :, 0].std())

    print(w1)
    print(A1)
    #print('G2 :', G2)

    test_fixed_working = 1
    if test_fixed_working :
        train_ds2 = WHDataset(seq_len=500, return_y=True, fixed_system=True, return_system=True, tau=1)
        train_dl2 = DataLoader(train_ds2, batch_size=1)
        batch_u, batch_e_v, batch_y, w11, b1, w2, b2, A11, B1, C1, D1, A2, B2, C2, D2 = next(iter(train_dl))

        print(batch_e_v[:, :, 0].mean())  # without normalization e_v has mean 6.5 and 1.71 std on fixed system
        print(batch_e_v[:, :, 0].std())
        print(batch_u[:, :, 0].mean())
        print(batch_u[:, :, 0].std())
        print(batch_y[:, :, 0].mean())
        print(batch_y[:, :, 0].std())

        print(w11)
        print(A11)

    plt.figure()
    Ts = 1e-2
    T = batch_u.shape[1]*Ts  # ts*self.seq_len# * 2
    t = np.arange(0, T, Ts)
    fig = plt.figure(figsize=(10, 8))
    for i in range(0,batch_u.shape[0]):

        plt.subplot(211)
        plt.plot(t, batch_u[i, :, 0], c='tab:blue', alpha=0.4)
        plt.legend(['$u$'], prop={'size': 15}, loc = 'upper right')
        plt.xlim(0, 5)
        plt.ylim(-5, 5)
        plt.grid()
        plt.subplot(212)
        plt.plot(t, batch_y[i, :, 0], c='tab:blue', alpha=0.4)
        plt.legend(['$y$'], prop={'size': 15}, loc = 'upper right')
        plt.xlabel("$t$ [s]")
        plt.xlim(0, 5)
        plt.ylim(-5, 5)
        plt.grid()
        #plt.subplot(313)
        #plt.plot(t, batch_input[i, :, 0], c='tab:blue', alpha=0.2)
        #plt.legend(['$e_v$'], prop={'size': 15}, loc = 'upper right')
        #plt.xlabel("$t$ [s]")
    plt.show()

