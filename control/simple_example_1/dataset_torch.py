import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import IterableDataset
from control_torch import drss, forced_response, tf2ss, c2d, perturb_matrices, set_seed

import matplotlib.pyplot as plt

class CustomDataset(IterableDataset):
    def __init__(self, seq_len, nx=2, nu=1, ny=1, seed=42, ts=0.01, return_y=False):
        set_seed(42)

        self.seq_len = seq_len + 1
        self.nx = nx
        self.nu = nu
        self.ny = ny

        # define settings
        self.ts = ts
        self.return_y = return_y

        # define nominal model
        self.G_0 = drss(self.nx, self.nu, self.ny, device=device)

        # define model reference
        tau = 1
        M_num = torch.tensor([0.01, 1], device=device, dtype=torch.float32)  # Numerator coefficients
        M_den = torch.tensor([tau/4, 1], device=device, dtype=torch.float32)  # Denominator coefficients
        M = tf2ss(M_num, M_den, device=device)  # M
        M_inv = tf2ss(M_den, M_num, device=device)  # M^-1, num den are inverted
        self.M = c2d(*M, self.ts, device=device)
        self.M_inv = c2d(*M_inv, self.ts, device=device)

    def __iter__(self):

        while True: # dataset is infinite

            # Generate data on-the-fly
            G = perturb_matrices(*self.G_0, percentage=0, device=device)

            u = torch.randn(self.seq_len, self.nu, device=device, dtype=torch.float32)
            # u = torch.ones(self.seq_len, self.nu, device=device, dtype=torch.float32)

            # Simulate forced response using custom GPU function
            y = forced_response(*G, u)

            # Prefilter with M
            u_L = forced_response(*self.M, u)
            y_L = forced_response(*self.M, y)

            # Compute virtual reference
            r_v = forced_response(*self.M_inv, y_L)

            # Compute virtual error
            e_v = r_v - y_L

            # Align to have proper signal
            u_L = u_L[:-1]
            y_L = y_L[1:]
            r_v = r_v[1:]
            e_v = e_v[1:]

            if self.return_y:
                yield y_L, u_L, e_v
            else:
                yield u_L, e_v

class WHDataset(IterableDataset):
    def __init__(self, seq_len, nx=2, nu=1, ny=1, seed=42, ts=0.01, return_y=False):
        set_seed(42)

        self.seq_len = seq_len + 1
        self.nx = nx
        self.nu = nu
        self.ny = ny

        # define settings
        self.ts = ts
        self.return_y = return_y

        # define nominal model
        n_in = 1
        n_out = 1
        n_hidden = 32

        # Generate random weights and biases using PyTorch's random functions and move them to the GPU
        self.w1 = torch.randn(n_hidden, n_in, device=device) / torch.sqrt(
             torch.tensor(n_in, dtype=torch.float32, device=device)) * (5 / 3)
        self.b1 = torch.randn(1, n_hidden, device=device) * 1.0
        self.w2 = torch.randn(n_out, n_hidden, device=device) / torch.sqrt(
             torch.tensor(n_hidden, dtype=torch.float32, device=device))
        self.b2 = torch.randn(1, n_out, device=device) * 1.0
        self.G_01 = drss(self.nx, self.nu, self.ny, device=device)
        self.G_02 = drss(self.nx, self.nu, self.ny, device=device)

        # define model reference
        tau = 1
        M_num = torch.tensor([0.01, 1], device=device, dtype=torch.float32)  # Numerator coefficients
        M_den = torch.tensor([tau/4, 1], device=device, dtype=torch.float32)  # Denominator coefficients
        M = tf2ss(M_num, M_den, device=device)  # M
        M_inv = tf2ss(M_den, M_num, device=device)  # M^-1, num den are inverted
        self.M = c2d(*M, self.ts, device=device)
        self.M_inv = c2d(*M_inv, self.ts, device=device)

    def __iter__(self):

        # A simple ff neural network
        def nn_fun(x):
            out = x @ self.w1.T + self.b1
            out = torch.tanh(out)
            out = out @ self.w2.T + self.b2
            return out

        while True: # dataset is infinite

            # Generate data on-the-fly
            G1 = perturb_matrices(*self.G_01, percentage=0, device=device)
            G2 = perturb_matrices(*self.G_02, percentage=0, device=device)

            u = torch.randn(self.seq_len, self.nu, device=device, dtype=torch.float32)
            # u = torch.ones(self.seq_len, self.nu, device=device, dtype=torch.float32)

            # Simulate forced response using custom GPU function
            y1 = forced_response(*G1, u)

            # F
            y2 = nn_fun(y1)

            # G2
            y = forced_response(*G2, y2)

            # Prefilter with M
            u_L = forced_response(*self.M, u)
            y_L = forced_response(*self.M, y)

            # Compute virtual reference
            r_v = forced_response(*self.M_inv, y_L)

            # Compute virtual error
            e_v = r_v - y_L

            # Align to have proper signal
            u_L = u_L[:-1]
            y_L = y_L[1:]
            r_v = r_v[1:]
            e_v = e_v[1:]

            if self.return_y:
                yield y_L, u_L, e_v
            else:
                yield u_L, e_v

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = CustomDataset(seq_len=500, ts=0.01, seed=42, return_y=True)
    # dataset = WHDataset(seq_len=500, ts=0.01, seed=42, return_y=True)
    dataloader = DataLoader(dataset, batch_size=32)

    batch_y, batch_u, batch_e = next(iter(dataloader))

    print(batch_y.shape)

    ts = 1e-2
    T = batch_u.shape[1] * ts  # ts*self.seq_len# * 2
    t = np.arange(0, T, ts)

    for i in range(0, batch_u.shape[0]):
        plt.subplot(311)
        plt.plot(t, batch_u[i, :, 0].cpu(), c='tab:blue', alpha=0.2)
        # plt.legend(['$e_v$'])
        plt.ylabel("$u_L$")
        plt.tick_params('x', labelbottom=False)

        plt.subplot(312)
        plt.plot(t, batch_y[i, :, 0].cpu(), c='tab:blue', alpha=0.2)
        # plt.legend(['$u$'])
        plt.ylabel("$y_L$")
        plt.xlabel("$t$ [s]")

        plt.subplot(313)
        plt.plot(t, batch_e[i, :, 0].cpu(), c='tab:blue', alpha=0.2)
        # plt.legend(['$y$'])
        plt.ylabel("$e_v$")
        plt.tick_params('x', labelbottom=False)

    plt.show()
