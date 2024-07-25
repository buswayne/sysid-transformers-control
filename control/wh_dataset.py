import math
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset, Dataset
import matplotlib.pyplot as plt
import control  # pip install python-control, pip install slycot (optional)
from control.matlab import *
from lti import drss_matrices, dlsim
from wh_simulate import simulate_wh
from wh_simulate import fixed_wh_system



class WHDataset(IterableDataset):
    def __init__(self, nx=5, nu=1, ny=1, seq_len=500,
                  normalize=True, dtype="float32",
                 fixed=False, data_seed=None, return_y=False, use_prefilter = False,
                  use_e_int = False,
                 tau=1, **mdlargs):  # system and data seed = None
        super(WHDataset).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.seq_len = seq_len
        self.dtype = dtype
        self.normalize = normalize
        self.data_seed = data_seed
        self.data_rng = np.random.default_rng(data_seed)  # source of randomness for model generation
        self.fixed= fixed  # same model at each iteration (classical identification)
        self.mdlargs = mdlargs
        self.return_y = return_y
        self.tau = tau
        self.use_prefilter = use_prefilter
        self.data = None
        self.ts = 1e-2
        #self.T = 20
        self.T = self.seq_len * self.ts
        self.t = np.arange(0, self.T, self.ts)
        self.wh_system_fixed = fixed_wh_system()
        self.use_e_int = use_e_int

        ##UNUSED ARGUMENTS
        # self.strictly_proper = strictly_proper
        # self.random_order = random_order  # random number of states from 1 to nx
        # self.system_seed = system_seed
        # self.system_rng = np.random.default_rng(system_seed)  # source of randomness for model generation

        # with this check i can see that it generates different batches, still using the same system
        #G1 = self.wh_system_fixed['G1']
        #print(G1)

        if self.fixed:
            self.data = self._generate_data()

    def __len__(self):
        return len(self.data[0]) if self.fixed else 32


    def _generate_data(self):
        data_y = []
        data_u = []
        data_e = []

        if self.return_y:
            for _ in range(32):
                # Adjust the range based on the desired size of the fixed dataset
              y, u, e = self._generate_sample()
              data_y.append(y)
              data_u.append(u)
              data_e.append(e)

            return torch.stack(data_y), torch.stack(data_u), torch.stack(data_e)
        else :
            for _ in range(32):
              u, e = self._generate_sample()
              data_u.append(u)
              data_e.append(e)
            return torch.stack(data_u), torch.stack(data_e)


    def _generate_sample(self):
        # np.random.seed(42)

        t = self.t
        n_skip = 200
        # not sure about the utility of n_skip, keeping it for now
        u = self.data_rng.normal(size=(self.seq_len + n_skip, 1))
        u = u.reshape(-1, 1)

        #u, y, A1, B1, C1, D1, A2, B2, C2, D2, w1, b1, w2, b2 = simulate_wh(t, u)
        u, y = simulate_wh(self.wh_system_fixed, u)

        u = u[n_skip:]
        y = y[n_skip:]



        s = tf('s')
        z = tf([1, 0], [1], dt=self.ts)
        M = 1 / (1 + (self.tau / (2 * np.pi)) * s)
        M = c2d(M, self.ts, 'matched')
        W = 1 / (1 + (0.1 * self.tau / (2 * np.pi)) * s)
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
        #r_v = np.insert(r_v[:-1], 0, 0)

        e_v = (r_v - y_L).reshape(-1, 1)
        e_v_integral = np.cumsum(e_v).reshape(-1, 1)

        u_L = u_L.reshape(-1, 1)

        if self.normalize:
            y_L = ( y_L - 6.60 ) / 0.83
            e_v = e_v / 1.5
            e_v_integral = (e_v_integral - 40 ) / 130
           # y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)
           # e_v = (e_v - e_v.mean(axis=0)) / (e_v.std(axis=0) + 1e-6)
           # u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-6)

        u_L = u_L[:-1].astype(self.dtype)
        y_L = y_L[1:].astype(self.dtype)
        r_v = r_v[1:].astype(self.dtype)
        e_v = e_v[1:].astype(self.dtype)
        e_v_integral = e_v_integral[1:].astype(self.dtype)

        start_idx = 0 # that -1 is to be checked, without it created batch_u of shape [1,499,32] and other signals with shape[1,500,32]
        e_v = e_v[start_idx:start_idx + self.seq_len-1]
        u_L = u_L[start_idx:start_idx + self.seq_len-1]
        y_L = y_L[start_idx:start_idx + self.seq_len-1]
        r_v = r_v[start_idx:start_idx + self.seq_len-1]
        e_v_integral = e_v_integral[start_idx : start_idx + self.seq_len-1 ]

        e = e_v.reshape(-1, 1)
        u = u_L.reshape(-1, 1)
        y = y_L.reshape(-1, 1)
        r = r_v.reshape(-1, 1)
        e_v_integral = e_v_integral.reshape(-1,1)

        #print(A2) #i used this to see if it returned the same system at every call
        if self.return_y :
            return torch.tensor(y), torch.tensor(u), torch.tensor(e)
        elif self.use_e_int :
            return torch.tensor(u), torch.tensor(e_v_integral)
        else :
            return torch.tensor(u), torch.tensor(e)

    def __getitem__(self, idx):
        if self.fixed and self.return_y:
            return self.data[0][idx], self.data[1][idx], self.data[2][idx]
        elif self.fixed and not self.return_y :
            return self.data[0][idx], self.data[1][idx]
        else:
            return self._generate_sample()

    def __iter__(self):
        while True:
           yield self._generate_sample()


    #def __iter__(self):
    #    for i in range(len(self)):
    #       yield self.__getitem__(i)


if __name__ == "__main__":
    train_ds = WHDataset(seq_len=500, use_prefilter=False, fixed=False, return_y = False, use_e_int= True)
    train_dl = DataLoader(train_ds, batch_size=32)
    #batch_y, batch_u, batch_e = next(iter(train_dl))
    batch_u, batch_e = next(iter(train_dl))



    print('u mean:', batch_u[:, :, 0].mean())
    print('u std:', batch_u[:, :, 0].std())
    print('e mean:', batch_e[:, :, 0].mean())
    print('e std:', batch_e[:, :, 0].std())

    ##THIS IS TO CHECK THAT IT RE-USES THE SAME SYSTEM AS IN train_ds
    train_ds2 = WHDataset(seq_len=500, use_prefilter=False, fixed=False, return_y=False, use_e_int=True)
    train_dl2 = DataLoader(train_ds2, batch_size=32)
    batch_u2, batch_e2 = next(iter(train_dl))
    print('u2 mean:', batch_u2[:, :, 0].mean())
    print('u2 std:', batch_u2[:, :, 0].std())
    print('e2 mean:', batch_e2[:, :, 0].mean())
    print('e2 std:', batch_e2[:, :, 0].std())

    #print(batch_y.shape)
    #print(batch_y.dtype)
    #print(batch_u.shape)
    #print(batch_e.shape)
    # print(batch_y[3,8,0])


    # batch_y, batch_u, batch_e = next(iter(train_dl))
    # print(batch_y[3, 8, 0])


    plt.figure(figsize=(7, 5))
    # plt.plot(batch_input[0,:,0])
    Ts = 1e-2
    T = batch_u.shape[1] * Ts  # ts*self.seq_len# * 2
    t = np.arange(0, T, Ts)

    for i in range(0, batch_u.shape[0]):
        plt.subplot(211)
        plt.plot(t, batch_u[i, :, 0], c='tab:blue', alpha=0.2)
        # plt.legend(['$e_v$'])
        plt.ylabel("$u$")
        plt.tick_params('x', labelbottom=False)

        #plt.subplot(312)
        #plt.plot(t, batch_y[i, :, 0], c='tab:blue', alpha=0.2)
        # plt.legend(['$y$'])
        #plt.ylabel("$y_L$")
        #plt.tick_params('x', labelbottom=False)

        plt.subplot(212)
        plt.plot(t, batch_e[i, :, 0], c='tab:blue', alpha=0.2)
        # plt.legend(['$u$'])
        plt.ylabel("$e_v$")
        plt.xlabel("$t$ [s]")
    plt.show()