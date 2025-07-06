import matplotlib.pyplot as plt
import numpy as np

# Use custom style
plt.style.use("../../config/matplotlib/mhedas.mplstyle")

# File paths
raw_file_path = "../data/signals/raw/sample_01.txt"
ground_truth_file_path = "../data/signals/ground_truth/sample_01.txt"
smoothed_file_path = "../data/signals/custom_method/smoothed/sample_01.txt"
baseline_corrected_file_path = "../data/signals/custom_method/baseline/sample_01.txt"
filtered_file_path = "../data/signals/custom_method/filtered/sample_01.txt"
custom_peaks_file_path = "../data/peaks/custom_peaks/sample_01.txt"

# Load raw signal data
raw_data = np.loadtxt(raw_file_path, delimiter=",")
t = raw_data[:, 0]
signal = raw_data[:, 1]

# Load ground truth signal data
ground_truth_data = np.loadtxt(ground_truth_file_path, delimiter=",")
gt_sig = ground_truth_data[:, 1]

# Load smoothed signal
smoothed_sig = np.loadtxt(smoothed_file_path, delimiter=",")[:, 1]

# Load baseline signal
baseline_sig = np.loadtxt(baseline_corrected_file_path, delimiter=",")[:, 1]

# Load filtered signal
filtered_sig = np.loadtxt(filtered_file_path, delimiter=",")[:, 1]

# Load custom peaks
custom_peaks = np.loadtxt(custom_peaks_file_path, delimiter=",")
peak_t = custom_peaks[:, 0]
peak_v = custom_peaks[:, 1]

# Define figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Raw and Smoothed Signals
axes[0, 0].plot(t, signal, label="Raw Signal")
axes[0, 0].plot(t, smoothed_sig, label="Smoothed Signal")
axes[0, 0].set_title("Raw and Smoothed Signals")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_xlim(t.min(), t.max())
axes[0, 0].set_ylabel("Signal Amplitude")
axes[0, 0].legend()

# Smoothed and Baseline Signals
axes[0, 1].plot(t, smoothed_sig, label="Smoothed Signal")
axes[0, 1].plot(t, baseline_sig, label="Baseline")
axes[0, 1].set_title("Smoothed and Baseline Signals")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_xlim(t.min(), t.max())
axes[0, 1].set_ylabel("Signal Amplitude")
axes[0, 1].legend()

# Filtered Signal
axes[1, 0].plot(t, filtered_sig, label="Filtered Signal")
axes[1, 0].set_title("Filtered Signal")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_xlim(t.min(), t.max())
axes[1, 0].set_ylabel("Signal Amplitude")
axes[1, 0].legend()

# Ground truth and custom peaks
axes[1, 1].stem(
    t[gt_sig > 0],
    gt_sig[gt_sig > 0],
    linefmt="orange",
    markerfmt="o",
    basefmt=" ",
    label="Ground Truth",
)
axes[1, 1].stem(
    peak_t,
    peak_v,
    linefmt="red",
    markerfmt="o",
    basefmt=" ",
    label="Custom Peaks",
)
axes[1, 1].set_title("Ground Truth and Detected Peaks")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Signal Amplitude")
axes[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG file
output_path = "../images/custom_method_plot.png"
plt.savefig(output_path, dpi=300)

# Display the plot
plt.show()
