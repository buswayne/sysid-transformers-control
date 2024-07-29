import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from simple_example_1_mine import simulate_simple_example_1
import matplotlib.pyplot as plt
from control.matlab import *
from scipy.interpolate import interp1d

class SimpleExample1Dataset(IterableDataset):
    def __init__(self, seq_len=1e6, normalize=False, dtype=torch.float32, return_y=False, device='cpu',tau=1, ts=1e-2):
        super(SimpleExample1Dataset).__init__()
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.normalize = normalize
        self.return_y = return_y
        self.device = torch.device(device)
        self.tau = tau
        self.ts = ts

    def __iter__(self):
        # Call the function to generate data
        ts = 1e-2
        T = 20  # ts*self.seq_len# * 2
        t = np.arange(0, T, ts)

        n_context = self.seq_len

        while True:  # infinite dataset
            # prbs instead
            # random
            n_steps = np.random.randint(2, 50)
            u = np.random.normal(0, 1000, t.shape)

            f = interp1d(t[::n_steps], u[::n_steps], kind='next', bounds_error=False, fill_value=0.0)
            u = np.nan_to_num(u)

            # System ,  THIS STILL RELIES ON LSIM , PERFORMED WITH CPU
            x, u, y = simulate_simple_example_1(t, u, perturbation=0.2)

            # Desired variable to be controlled is x1 = \theta. Let's compute virtual error
            s = tf('s')  # s
            M = 1 / (1 + (self.tau / (2 * np.pi)) * s)
            M = c2d(M, self.ts, 'tustin')
            # get virtual error     relies on lsmi
            r_v = lsim(M ** (-1), y, t)[0]
            e_v = (r_v - y).reshape(-1, 1)  # must be 2d
            u = u.reshape(-1, 1)
            e_v_integral = np.cumsum(e_v).reshape(-1, 1)

            e_v = e_v.astype(np.float32)
            e_v_integral = e_v_integral.astype(np.float32)
            u = np.insert(u, 0, 1e-6)
            u = u[:-1].astype(np.float32)
            y = y.astype(np.float32)

            # lunghezza contesto 5
            start_idx = 0
            e_v = e_v[start_idx:start_idx + n_context]
            e_v_integral = e_v_integral[start_idx:start_idx + n_context]
            u = u[start_idx:start_idx + n_context]
            y = y[start_idx:start_idx + n_context]

            if self.normalize:
                e_v = e_v / 10.2  # mean 0, std 10
                u = u / 1000  # mean 0, std 17
                e_v_integral = e_v_integral / 220

            #input_vector = np.concatenate((e_v, e_v_integral), axis=1)ù
            input_vector = e_v.reshape(-1,1)
            output_vector = u.reshape(-1, 1)
            y = y.reshape(-1, 1)

            # Convert numpy arrays to tensors and transfer to the GPU
            input_vector_tensor = torch.tensor(input_vector, dtype=self.dtype).to(self.device)
            output_vector_tensor = torch.tensor(output_vector, dtype=self.dtype).to(self.device)
            y_tensor = torch.tensor(y, dtype=self.dtype).to(self.device)

            if self.return_y:
                yield output_vector_tensor, input_vector_tensor, y_tensor
            else:
                yield output_vector_tensor, input_vector_tensor


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available())
    train_ds = SimpleExample1Dataset(seq_len=500, normalize=True, return_y=True, device=device)
    train_dl = DataLoader(train_ds, batch_size=32)

    batch_output, batch_input, y = next(iter(train_dl))

    print(batch_output.shape)
    print(batch_input.shape)
    print(batch_output[:, :, 0].mean())
    print(batch_output[:, :, 0].std())
    print(batch_input[:, :, 0].mean())
    print(batch_input[:, :, 0].std())

    plt.figure()
    Ts = 1e-2
    T = batch_input.shape[1] * Ts  # ts*self.seq_len# * 2
    t = np.arange(0, T, Ts)
    fig = plt.figure(figsize=(10, 8))
    for i in range(0, batch_output.shape[0]):
        plt.subplot(211)
        plt.plot(t, batch_output[i, :, 0].cpu().numpy(), c='tab:blue', alpha=0.2)
        plt.legend(['$u$'], prop={'size': 15}, loc='upper right')
        plt.subplot(212)
        plt.plot(t, batch_input[i, :, 0].cpu().numpy(), c='tab:blue', alpha=0.2)
        plt.legend(['$e_v$'], prop={'size': 15}, loc='upper right')
        plt.xlabel("$t$ [s]")
        # plt.subplot(313)
        # plt.plot(t, batch_input[i, :, 0], c='tab:blue', alpha=0.2)
        # plt.legend(['$e_v$'], prop={'size': 15}, loc = 'upper right')
        # plt.xlabel("$t$ [s]")
    plt.show()