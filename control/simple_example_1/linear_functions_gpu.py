import numpy as np
import torch
import control
import matplotlib.pyplot as plt
import time
from control.matlab import *

def normalize(num, den, device='cuda'):
    """Normalize numerator/denominator of a continuous-time transfer function on GPU.

    Parameters
    ----------
    num: torch.Tensor
        Numerator of the transfer function. Can be a 2-D tensor to normalize
        multiple transfer functions.
    den: torch.Tensor
        Denominator of the transfer function. At most 1-D tensor.

    Returns
    -------
    num: torch.Tensor
        The numerator of the normalized transfer function. At least a 1-D
        tensor. A 2-D tensor if the input `num` is a 2-D tensor.
    den: torch.Tensor
        The denominator of the normalized transfer function.

    Notes
    -----
    Coefficients for both the numerator and denominator should be specified in
    descending exponent order (e.g., ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``).
    """
    num = num.to(device)
    den = den.to(device)

    den = torch.atleast_1d(den)
    num = torch.atleast_2d(num)

    if den.ndimension() != 1:
        raise ValueError("Denominator polynomial must be rank-1 tensor.")
    if num.ndimension() > 2:
        raise ValueError("Numerator polynomial must be rank-1 or rank-2 tensor.")
    if torch.all(den == 0):
        raise ValueError("Denominator must have at least one nonzero element.")

    # Trim leading zeros in denominator, leave at least one
    den = torch.cat([den[torch.nonzero(den, as_tuple=True)[0][0]:],
                     torch.zeros(max(0, len(den) - len(den[torch.nonzero(den, as_tuple=True)[0][0]:])), device=device)])

    # Normalize transfer function
    den_0 = den[0]
    num = num / den_0
    den = den / den_0

    # Count numerator columns that are all zero
    leading_zeros = 0
    for col in num.T:
        if torch.allclose(col, torch.tensor(0.0, device=device), atol=1e-14):
            leading_zeros += 1
        else:
            break

    # Trim leading zeros of numerator
    if leading_zeros > 0:
        # Make sure at least one column remains
        if leading_zeros == num.shape[1]:
            leading_zeros -= 1
        num = num[:, leading_zeros:]

    # Squeeze first dimension if singular
    if num.shape[0] == 1:
        num = num.squeeze(0)

    return num, den


def tf2ss(num, den, device='cuda'):
    """
    Convert a transfer function to state-space representation using canonical form.

    Args:
        num (torch.Tensor): Numerator coefficients.
        den (torch.Tensor): Denominator coefficients.
        device (str): Device to run on ('cuda' or 'cpu').

    Returns:
        Tuple: State-space matrices (A, B, C, D).
    """
    device = torch.device(device)

    num, den = normalize(num, den)  # Normalize the input

    num = num.reshape(-1, 1)
    den = den.reshape(-1, 1)

    M = num.shape[0]
    K = den.shape[0]

    if M > K:
        raise ValueError("Improper transfer function. `num` is longer than `den`.")

    if M == 0 or K == 0:  # Null system
        return (torch.zeros((0, 0), dtype=torch.float32, device=device),
                torch.zeros((0, 0), dtype=torch.float32, device=device),
                torch.zeros((0, 0), dtype=torch.float32, device=device),
                torch.zeros((0, 0), dtype=torch.float32, device=device))

    # Pad numerator to have same number of columns as denominator
    num = torch.cat((torch.zeros((num.shape[0], K - M), dtype=num.dtype, device=device), num), dim=1)

    if num.shape[-1] > 0:
        D = num[0].unsqueeze(1)  # Create 2D tensor for D
    else:
        D = torch.tensor([[0]], dtype=torch.float32, device=device)

    if K == 1:
        D = D.reshape(num.shape)
        return (torch.zeros((1, 1), dtype=torch.float32, device=device),
                torch.zeros((1, D.shape[1]), dtype=torch.float32, device=device),
                torch.zeros((D.shape[0], 1), dtype=torch.float32, device=device),
                D)

    # Create A matrix
    A = torch.zeros((K - 1, K - 1), dtype=torch.float32, device=device)
    A[0, :] = -den[1:] / den[0]
    A[1:, :-1] = torch.eye(K - 2, dtype=torch.float32, device=device)

    # Create B matrix
    B = torch.eye(K - 1, 1, dtype=torch.float32, device=device)

    # Create C matrix
    C = num[1:] - torch.outer(num[0].reshape(-1), den[1:].reshape(-1))

    # Ensure D is in the correct shape
    D = D.reshape(C.shape[0], B.shape[1])

    return A, B, C, D

def c2d(A, B, C, D, dt, device='cuda'):
    """
    Convert continuous-time state-space matrices to discrete-time using the bilinear transform.

    Args:
        A (torch.Tensor): Continuous-time state matrix of shape (n, n).
        B (torch.Tensor): Continuous-time input matrix of shape (n, m).
        C (torch.Tensor): Continuous-time output matrix of shape (p, n).
        D (torch.Tensor): Continuous-time feedthrough matrix of shape (p, m).
        T (float): Sampling period.
        device (str): Device to run on ('cuda' or 'cpu').

    Returns:
        A_d (torch.Tensor): Discrete-time state matrix.
        B_d (torch.Tensor): Discrete-time input matrix.
        C_d (torch.Tensor): Discrete-time output matrix.
        D_d (torch.Tensor): Discrete-time feedthrough matrix.
    """
    # Ensure matrices are on the correct device
    A = A.to(device)
    B = B.to(device)
    C = C.to(device)
    D = D.to(device)

    alpha = 0.5

    # Compute I - alpha * dt * a
    I = torch.eye(A.size(0), device=device)
    ima = I - alpha * dt * A

    # Compute ad and bd by solving linear systems
    I_alpha = I + (1.0 - alpha) * dt * A
    A_d = torch.linalg.solve(ima, I_alpha)
    B_d = torch.linalg.solve(ima, dt * B)

    # Compute cd and dd
    C_d = torch.linalg.solve(ima.T, C.T).T
    D_d = D + alpha * torch.matmul(C, B_d)

    return A_d, B_d, C_d, D_d

def custom_drss(nx, nu, ny, stricly_proper=False, device="cuda:0", dtype=torch.float32):
    """
    Generate random state-space matrices for a discrete-time linear system.
    Args:
        nx: Number of states
        nu: Number of inputs
        ny: Number of outputs
        device: Device to store tensors
        dtype: Data type of tensors
    Returns:
        A, B, C, D: State-space matrices
    """
    A = torch.randn(nx, nx, device=device, dtype=dtype) * 0.1
    B = torch.randn(nx, nu, device=device, dtype=dtype)
    C = torch.randn(ny, nx, device=device, dtype=dtype)
    D = torch.randn(ny, nu, device=device, dtype=dtype)
    if stricly_proper:
        D = D * 0.0

    # Ensure A is stable
    L_complex = torch.linalg.eigvals(A)
    max_eigval = torch.max(torch.abs(L_complex))
    if max_eigval >= 1:
        A = A / (max_eigval + 1.1)

    return A, B, C, D


def custom_forced_response(A, B, C, D, u, x0=None):
    """
    Simulate the forced response of a discrete-time linear system.
    Args:
        A, B, C, D: State-space matrices
        u: Input sequence (T, nu)
        x0: Initial state (nx,)
    Returns:
        y: Output sequence (T, ny)
    """
    T, nu = u.shape
    nx = A.shape[0]
    ny = C.shape[0]

    # Convert x0 to tensor if it's not None
    if x0 is None:
        x0 = torch.zeros(nx, device=u.device, dtype=u.dtype)
    else:
        x0 = torch.tensor(x0, device=u.device, dtype=u.dtype)
        if x0.shape[0] != nx:
            raise ValueError(f"Initial state x0 must have {nx} elements, but got {x0.shape[0]}")

    x = x0
    y = torch.zeros(T, ny, device=u.device, dtype=u.dtype)  # Preallocate tensor for outputs

    # Compute the initial output
    y[0] = C @ x0 + D @ u[0]

    for t in range(1, T):  # Start from the second sample to avoid duplicating the initial output
        x = A @ x + B @ u[t]
        y[t] = C @ x + D @ u[t]

    return y


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nx, nu, ny = 1, 1, 1
seq_len = 10
T = 1

Ts = 0.01  # Sampling period in seconds
t = np.arange(0, T, Ts)
# Generate random state-space matrices on GPU
# A, B, C, D = custom_drss(nx, nu, ny, device=device)

# Parameters
K = 1
settling_time = 1  # Desired settling time in seconds
tau = settling_time / 4  # Continuous-time time constant

# Convert to state-space (continuous-time)
num = torch.tensor([0.01, 1], device=device, dtype=torch.float32)  # Numerator coefficients
den = torch.tensor([tau, 1], device=device, dtype=torch.float32)  # Denominator coefficients


A_c, B_c, C_c, D_c = tf2ss(den, num, device=device)

print(A_c, B_c, C_c, D_c)

# Convert to discrete-time state-space
A, B, C, D = c2d(A_c, B_c, C_c, D_c, Ts, device=device)

# u = torch.randn(seq_len, nu, device=device, dtype=torch.float32)
# Create a step signal as input
u = torch.ones(len(t), nu, device=device, dtype=torch.float32)

# Simulate forced response using custom GPU function
start = time.time()
y_gpu = custom_forced_response(A, B, C, D, u)
print(time.time() - start)

# Convert tensors to numpy arrays for comparison with control library
A_np = A.cpu().numpy()
B_np = B.cpu().numpy()
C_np = C.cpu().numpy()
D_np = D.cpu().numpy()
u_np = u.cpu().numpy()

# Simulate forced response using control library on CPU
sys = control.StateSpace(A_np, B_np, C_np, D_np, 1)
M_inv = control.matlab.tf([0.01, 1], [tau, 1])**-1
start = time.time()
_, y_control = control.forced_response(M_inv, T=t, U=u_np.T)
print(time.time() - start)

# Transpose y_control to match the shape (T, ny)
y_control = y_control.T

# Compare the results
y_gpu_np = y_gpu.cpu().numpy().flatten()


print("Difference between custom GPU response and control library response:")
print(np.max(np.abs(y_gpu_np - y_control)))

# Check if the outputs are close enough
tolerance = 1e-6
if np.allclose(y_gpu_np, y_control, atol=tolerance):
    print("The outputs are close enough within the tolerance.")
else:
    print("The outputs are not close enough.")

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(t, y_gpu_np, label='Custom GPU Response', alpha=0.7)
plt.plot(t, y_control, label='Control Library Response', alpha=0.7, linestyle='dashed')
plt.xlabel('Time Step')
plt.ylabel('Output')
plt.title('Comparison of System Responses')
plt.legend()
plt.show()