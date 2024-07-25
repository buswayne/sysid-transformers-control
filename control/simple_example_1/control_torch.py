import torch

class DiscreteTransferFunction(torch.nn.Module):
    def __init__(self, b, a, dt=1.0):
        super(DiscreteTransferFunction, self).__init__()
        self.b = torch.tensor(b, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)
        self.dt = dt

    def forward(self, r):
        # Ensure the input is a tensor and has the correct dtype
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float32)
        else:
            r = r.type(torch.float32)

        y = torch.zeros_like(r)
        # Compute the output using the difference equation
        for t in range(len(r)):
            for i in range(len(self.b)):
                if t - i >= 0:
                    y[t] += self.b[i] * r[t - i]
            for j in range(1, len(self.a)):
                if t - j >= 0:
                    y[t] -= self.a[j] * y[t - j]

        y = torch.cat((torch.tensor([0]), y[:-1]))

        return y * self.dt