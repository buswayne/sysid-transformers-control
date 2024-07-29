import numpy as np
import torch
import control
import matplotlib.pyplot as plt


def custom_drss(nx, nu, ny, device="cuda:0", dtype=torch.float32):
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
    D = torch.randn(ny, nu, device=device, dtype=dtype) * 0.0

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

if __name__ == "__main__":
# Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nx, nu, ny = 5, 1, 1
    seq_len = 10

    # Generate random state-space matrices on GPU
    A, B, C, D = custom_drss(nx, nu, ny, device=device)

    # u = torch.randn(seq_len, nu, device=device, dtype=torch.float32)
    # Create a step signal as input
    u = torch.ones(seq_len, nu, device=device, dtype=torch.float32)

    # Simulate forced response using custom GPU function
    y_gpu = custom_forced_response(A, B, C, D, u, x0=np.zeros(5))

    # Convert tensors to numpy arrays for comparison with control library
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()
    C_np = C.cpu().numpy()
    D_np = D.cpu().numpy()
    u_np = u.cpu().numpy()

    # Simulate forced response using control library on CPU
    sys = control.StateSpace(A_np, B_np, C_np, D_np, 1)
    _, y_control = control.forced_response(sys, T=None, U=u_np.T, X0=np.zeros(5))

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
    plt.plot(y_gpu_np, label='Custom GPU Response', alpha=0.7)
    plt.plot(y_control, label='Control Library Response', alpha=0.7, linestyle='dashed')
    plt.xlabel('Time Step')
    plt.ylabel('Output')
    plt.title('Comparison of System Responses')
    plt.legend()
    plt.show()