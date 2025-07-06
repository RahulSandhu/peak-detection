import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Use custom style
plt.style.use("../../config/matplotlib/mhedas.mplstyle")

# File paths
raw_file_path = "../data/signals/raw/sample_03.txt"
ground_truth_file_path = "../data/signals/ground_truth/sample_03.txt"

# Load raw signal data
raw_data = np.loadtxt(raw_file_path, delimiter=",")
t = raw_data[:, 0]
signal = raw_data[:, 1]

# Load ground truth signal data
ground_truth_data = np.loadtxt(ground_truth_file_path, delimiter=",")
gt_t = ground_truth_data[:, 0]
gt_sig = ground_truth_data[:, 1]

# Extract ground truth times
gt_t = t[np.where(gt_sig > 0)]

# Define parameters
fs = 10
win_dur = 500
win_size = win_dur * fs

# Initialize lists for detected peaks
peak_t = []
peak_v = []

# Calculate the number of windows
num_windows = (len(t) + win_size - 1) // win_size

# Single window for demonstration
demo_window_idx = 2

# Define figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Loop over each window
for i in range(num_windows):
    # Define window range
    start_idx = i * win_size
    end_idx = min((i + 1) * win_size, len(t))
    win_sig = signal[start_idx:end_idx]
    win_t = t[start_idx:end_idx]

    # Check if there are any ground truth peaks in the current window
    gt_idx_window = np.where((gt_t >= win_t[0]) & (gt_t <= win_t[-1]))[0]

    if len(gt_idx_window) == 0:
        # No ground truth peaks in this window, skip it
        continue
    elif len(gt_idx_window) == 1:
        # One ground truth peak: use half max_v as height and ignore distance
        max_v = np.max(win_sig)
        min_h = 0.25 * max_v
        peaks, _ = find_peaks(win_sig, height=min_h, prominence=0.15 * max_v)
    else:
        # Two or more ground truth peaks: use height, distance, and prominence
        max_v = np.max(win_sig)
        min_h = 0.25 * max_v
        mean_diff = np.mean(np.diff(gt_t[gt_idx_window]))
        min_d = max(int(mean_diff * fs), 1)
        peaks, _ = find_peaks(
            win_sig, height=min_h, distance=min_d, prominence=0.15 * max_v
        )

    # Append detected peaks to the lists
    peak_t.extend(win_t[peaks])
    peak_v.extend(win_sig[peaks])

    # If iteration is on the selected window
    if i == demo_window_idx:
        # Window plot
        axes[0].plot(win_t, win_sig, label="Raw Signal")
        axes[0].plot(
            win_t,
            gt_sig[start_idx:end_idx],
            label="Ground Truth Signal",
            linestyle="--",
        )
        axes[0].scatter(
            win_t[peaks], win_sig[peaks], color="red", label="Detected Peaks", zorder=5
        )
        axes[0].set_title(f"Peak Detection in Window {i + 1}")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Signal Amplitude")
        axes[0].set_xlim(min(win_t), max(win_t))
        axes[0].legend()

# Ground truth and SciPy detected peaks
axes[1].stem(
    t[gt_sig > 0],
    gt_sig[gt_sig > 0],
    linefmt="orange",
    markerfmt="o",
    basefmt=" ",
    label="Ground Truth Signal",
)
axes[1].stem(
    peak_t,
    peak_v,
    linefmt="red",
    markerfmt="o",
    basefmt=" ",
    label="Detected Peaks",
)
axes[1].set_title("Ground Truth and Detected Peaks")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Signal Amplitude")
axes[1].legend()

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG file
output_path = "../images/scipy_method_plot.png"
plt.savefig(output_path, dpi=300)

# Display the plot
plt.show()
