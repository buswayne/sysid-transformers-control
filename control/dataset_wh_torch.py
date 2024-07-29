import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import IterableDataset
from control_torch import drss, forced_response, tf2ss, c2d, perturb_matrices, set_seed, initialize_nn

import matplotlib.pyplot as plt


def nn_fun(x):
    out = x @ w1.transpose() + b1
    out = np.tanh(out)
    out = out @ w2.transpose() + b2
    return out

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
        self.G1_0 = drss(self.nx, self.nu, self.ny, device=device)
        self.G2_0 = drss(self.nx, self.nu, self.ny, device=device) #discrete time matrices
        # Define the sizes MAYBE PUT THIS INIZIALIZATION IN A FUNCTION ( in control_torch)
        w1, b1, w2, b2 = initialize_nn()
        ##as this is in init, everytime CustomDataset is called G_0 is the same

        # define model reference
        tau = 0.1
        M_num = torch.tensor([0.01, 1], device=device, dtype=torch.float32)  # Numerator coefficients
        M_den = torch.tensor([tau/4, 1], device=device, dtype=torch.float32)  # Denominator coefficients
        M = tf2ss(M_num, M_den, device=device)  # M
        M_inv = tf2ss(M_den, M_num, device=device)  # M^-1, num den are inverted
        self.M = c2d(*M, self.ts, device=device)
        self.M_inv = c2d(*M_inv, self.ts, device=device)

    def __iter__(self):

        while True: # dataset is infinite

            # Generate data on-the-fly
            G1 = perturb_matrices(*self.G1_0, percentage=0, device=device)
            G2 = perturb_matrices(*self.G2_0, percentage=0, device=device)

            u = torch.randn(self.seq_len, self.nu, device=device, dtype=torch.float32)
            # u = torch.ones(self.seq_len, self.nu, device=device, dtype=torch.float32)

            # Simulate forced response using custom GPU function
            y1 = forced_response(*G1, u)
            ##do i need a normalization here ??
            y2 = nn_fun(y1)
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
