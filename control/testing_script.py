import numpy as np

# Original servo_motor function
def original_servo_motor(x, u, data):
    theta = x[0].item()
    omega = x[1].item()
    I = x[2].item()

    xdot = (np.dot(np.array([[0, 1, 0], [0, -data['b'] / data['J'], data['K'] / data['J']], [0, -data['K'] / data['L'], -data['R'] / data['L']]]) +
                   np.array([[0, 1, 0], [(data['m'] * data['g'] * data['l'])/data['J'], 0, 0], [0, 0, 0]]) * np.sin(theta) / theta,
                   np.array([[theta], [omega], [I]])) +
            np.dot(np.array([[0], [0], [1 / data['L']]]), u.item()))

    return xdot

# Modified servo_motor function
def modified_servo_motor(x, u, data):
    theta = x[0].item()
    omega = x[1].item()
    I = x[2].item()

    # State derivative expression
    M1_00 = 0
    M1_01 = 1
    M1_02 = 0

    M1_10 = 0
    M1_11 = -data['b'] / data['J']
    M1_12 = data['K'] / data['J']

    M1_20 = 0
    M1_21 = -data['K'] / data['L']
    M1_22 = -data['R'] / data['L']

    M2_00 = 0
    M2_01 = 1 * np.sin(theta) / theta
    M2_02 = 0

    M2_10 = ((data['m'] * data['g'] * data['l']) / data['J'] ) * np.sin(theta) / theta
    M2_11 = 0
    M2_12 = 0

    M2_20 = 0
    M2_21 = 0
    M2_22 = 0

    M3_00 = theta
    M3_01 = omega
    M3_02 = I

    # Matrix multiplication using explicit element-wise operations
    xdot_0 = (M1_00 + M2_00) * M3_00 + (M1_01 + M2_01) * M3_01 + (M1_02 + M2_02) * M3_02
    xdot_1 = (M1_10 + M2_10) * M3_00 + (M1_11 + M2_11) * M3_01 + (M1_12 + M2_12) * M3_02
    xdot_2 = (M1_20 + M2_20) * M3_00 + (M1_21 + M2_21) * M3_01 + (M1_22 + M2_22) * M3_02 + (u.item() / data['L'])

    xdot = np.array([[xdot_0], [xdot_1], [xdot_2]])

    return xdot

# Problem data, numeric constants
data = {'g': 9.8, 'R': 9.5, 'L': 0.84E-3, 'K': 53.6E-3, 'J': 2.2E-4, 'b': 6.6E-5, 'm': 0.07, 'l': 0.042}

# Initial conditions
x = np.array([1e-3, 1e-3, 1e-3])
u = np.array([1e-3])

# Call both functions
original_output = original_servo_motor(x, u, data).astype(np.float64)
modified_output = modified_servo_motor(x, u, data).astype(np.float64)

print(original_output)
print(modified_output)

# Check if outputs match
outputs_match = np.allclose(original_output, modified_output, atol=1e-4)
print("Outputs match:", outputs_match)
