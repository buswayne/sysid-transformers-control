import math
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import matplotlib.pyplot as plt
import control  # pip install python-control, pip install slycot (optional)
from control.matlab import *
from lti import drss_matrices, dlsim
from wh_simulate import simulate_wh



class WHDataset_fixedG(IterableDataset):
    def __init__(self, nx=5, nu=1, ny=1, seq_len=600, random_order=True,
                 strictly_proper=True, normalize=True, dtype="float32",
                 fixed_system=False, system_seed=0, data_seed=None,  return_y=False, fixed_u = False, return_system = False,
                  tau =1, **mdlargs):  #system and data seed = None
        super(WHDataset_fixedG).__init__()
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
        dtype = "float32"

        ts = 1e-2
        T = 5
        t = np.arange(0, T, ts)
        data_rng = np.random.default_rng(0)
        seq_len = 500
        n_skip = 200
        u = data_rng.normal(size=(seq_len + n_skip, 1))
        u = u.reshape(-1, 1)
        u, y, A11, B11, C11, D11, A22, B22, C22, D22, w11, b11, w22, b22 = simulate_wh(t, u)


        A1 = np.array([[0.57261285, 0.26499255], [-0.20746851, 1.05757019]])
        A1 = [np.array(row) for row in A1]
        B1 = np.array([[-2.25014117], [0.]])
        B1 = [np.array(row) for row in B1]
        C1 = [np.array(np.array([-0.58164084, 0.]))]
        #C1 = [np.array(row) for row in C1]
        D1 = [np.array(np.array([0.]))]
        #D1 = [np.array(row) for row in D1]



        G1 = (A1[0], B1[0], C1[0], D1[0])

        #G1 = (
        #    np.array(G1[0][0]),  # Convert first element
        #    np.array(G1[1][0]),  # Convert second element
        #    np.array(G1[2][0]),  # Convert third element
        #    np.array(G1[3][0])  # Convert fourth element
        #)

        A2 = [np.array([ 0.69790484,  0.46806399, -0.20050822,  0.32306574,  0.45940802]), np.array([ 0.05236874,  0.84211994, -0.00675783,  0.19006836,  0.30179567]), np.array([0.01714812, 0.0126176 , 0.76679674, 0.07035186, 0.17832423]), np.array([-0.04389574, -0.29700311,  0.11064575,  0.41333607, -0.33421139]), np.array([ 0.0489118 ,  0.20662678, -0.02640519,  0.14054648,  0.80200039])]
        B2 = [np.array([-0.]), np.array([-1.18571981]), np.array([-2.39823287]), np.array([0.]), np.array([-0.])]
        C2 = [np.array([-0.53000841, -0.        ,  0.        , -0.04980097,  0.        ])]
        D2 = [np.array([-0.])]
        w1 =[
        [0.20955037],
        [-0.22017477],
        [1.06737108],
        [0.17483353],
        [-0.89278229],
        [0.60265842],
        [2.17333341],
        [1.57846827],
        [-1.17289206],
        [-2.10903579],
        [-1.03879077],
        [0.06887663],
        [-3.87505129],
        [-0.36465277],
        [-2.07651825],
        [-1.22044559],
        [-0.9070983],
        [-0.52716693],
        [0.68605089],
        [1.73752228],
        [-0.21422444],
        [2.27743912],
        [-1.10865779],
        [0.58585012],
        [1.50578364],
        [0.15668716],
        [-1.23916542],
        [-1.53620896],
        [-0.76287638],
        [0.36699187],
        [-1.68269697],
        [-0.34862596]
        ]
        w1 = np.array(w1)

        b1 = np.array([[-0.15922501,  0.54084558,  0.21465912,  0.35537271, -0.65382861,
                        -0.12961363,  0.78397547,  1.49343115, -1.25906553,  1.51392377,
                         1.34587542,  0.7813114 ,  0.26445563, -0.31392281,  1.45802068,
                         1.96025832,  1.80163487,  1.31510376,  0.35738041, -1.20831863,
                        -0.00445413,  0.65647494, -1.28836146,  0.39512206,  0.42986369,
                         0.69604272, -1.18411797, -0.66170257, -0.43643525, -1.16980191,
                         1.73936788, -0.49591073]])
        w2 = np.array([[0.05815416, -0.0457096, 0.2799211, 0.23340905, 0.11196198, -0.38952919,
                        0.00919751, 0.12085979, 0.17747701, -0.10923157, 0.32208915, -0.23342142,
                        -0.11694274, 0.16529505, 0.00867171, 0.35397634, 0.0333258, -0.11193396,
                        -0.06674443, -0.1928892, -0.22586408, 0.11144206, 0.10273657, 0.22884783,
                        -0.13339672, 0.29859483, -0.05080345, 0.27831869, -0.07650645, -0.13001631,
                        0.04415623, 0.18233687]])
        b2 = np.array([[0.16100958]])

        w1 = w1[0]
        w1 = np.array(w1)
        w1 = w1.flatten()
        w1 = w1.reshape(-1, 1)
        b1 = b1[0]
        w2 = w2[0]
        w2 = np.array(w2)
        b2 = b2[0]



        G2 = (A2, B2, C2, D2)
        G2 = (
            np.array(G2[0][0]),  # Convert first element
            np.array(G2[1][0]),  # Convert second element
            np.array(G2[2][0]),  # Convert third element
            np.array(G2[3][0])  # Convert fourth element
        )




        while True:  # infinite dataset
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
            z = tf([1, 0], [1], dt=Ts)
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


            if self.return_y  :
                yield torch.tensor(output_vector), torch.tensor(input_vector), torch.tensor(y)
            else :
                yield torch.tensor(output_vector), torch.tensor(input_vector)




if __name__ == "__main__":

    train_ds = WHDataset_fixedG(seq_len=500, return_y= True, tau = 1)
    train_dl = DataLoader(train_ds, batch_size=1)
    batch_u, batch_e_v, batch_y = next(iter(train_dl))

    print(batch_u.shape)
    print(batch_e_v.shape)
    print(batch_e_v[:, :, 0].mean())  #without normalization e_v has mean 6.5 and 1.71 std on fixed system
    print(batch_e_v[:, :, 0].std())
    print(batch_u[:, :, 0].mean())
    print(batch_u[:, :, 0].std())
    print(batch_y[:, :, 0].mean())
    print(batch_y[:, :, 0].std())



    test_fixed_working = 0
    if test_fixed_working :
        train_ds2 = WHDataset_old(seq_len=500, return_y=True, fixed_system=True, return_system=True, tau=1)
        train_dl2 = DataLoader(train_ds2, batch_size=1)
        batch_u, batch_e_v, batch_y, w11, b1, w2, b2, A11, B1, C1, D1, A2, B2, C2, D2 = next(iter(train_dl))

        print(batch_e_v[:, :, 0].mean())  # without normalization e_v has mean 6.5 and 1.71 std on fixed system
        print(batch_e_v[:, :, 0].std())
        print(batch_u[:, :, 0].mean())
        print(batch_u[:, :, 0].std())
        print(batch_y[:, :, 0].mean())
        print(batch_y[:, :, 0].std())



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

