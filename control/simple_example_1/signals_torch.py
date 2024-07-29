import torch
import matplotlib.pyplot as plt


def steps_sequence(T, Ts, min_val, max_val, min_duration, max_duration, device='cuda'):
    # Ensure device is set correctly
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Calculate the total number of samples
    total_samples = int(T / Ts)

    # Initialize tensor for the final sequence
    steps_sequence = torch.empty(total_samples, device=device)

    current_index = 0

    while current_index < total_samples:
        # Determine remaining samples
        remaining_samples = total_samples - current_index

        # Randomly determine the duration for the current step in samples
        step_duration_sec = torch.randint(int(min_duration / Ts), int(max_duration / Ts) + 1, (1,), device=device).item()

        # Ensure we don't exceed the remaining samples
        step_duration = min(step_duration_sec, remaining_samples)

        # Generate a random step value between min_val and max_val
        step_value = (max_val - min_val) * torch.rand(1, device=device) + min_val

        # Fill the corresponding part of the sequence with the step value
        steps_sequence[current_index:current_index + step_duration] = step_value

        # Update the current index
        current_index += step_duration

    return steps_sequence

if __name__ == '__main__':
    # Example usage
    T = 20  # Total time
    Ts = 0.01  # Sampling time
    min_val = -20  # Minimum step value
    max_val = 20  # Maximum step value
    min_duration = 2  # Minimum duration for each step (in samples)
    max_duration = 5  # Maximum duration for each step (in samples)

    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    random_steps = steps_sequence(T, Ts, min_val, max_val, min_duration, max_duration, device)

    # Create the time array
    t = torch.arange(0, T, Ts, device=device)

    # Plotting the steps sequence
    plt.figure(figsize=(12, 6))
    plt.plot(t.cpu(), random_steps.cpu(), drawstyle='steps-post')
    plt.xlabel('Time (s)')
    plt.ylabel('Step Value')
    plt.title('Random Steps Sequence')
    plt.grid(True)
    plt.show()