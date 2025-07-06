import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Use custom style
plt.style.use("../../config/matplotlib/mhedas.mplstyle")

# File paths
raw_path = "../data/signals/raw/sample_01.txt"
ground_truth_path = "../data/signals/ground_truth/sample_01.txt"
ref_peak_path = "../data/signals/ref_peak.mat"

# Load raw signal data
raw_data = np.loadtxt(raw_path, delimiter=",")
t = raw_data[:, 0]
signal = raw_data[:, 1]

# Load ground truth signal data
ground_truth_data = np.loadtxt(ground_truth_path, delimiter=",")
gt_t = ground_truth_data[:, 0]
gt_sig = ground_truth_data[:, 1]

# Load reference peak data
ref_peak_data = loadmat(ref_peak_path)
ref_peak_t = ref_peak_data["tref"].flatten()
ref_peak_sig = ref_peak_data["xref"].flatten()

# Define figure
plt.figure(figsize=(14, 6))

# Raw Signal and Ground Truth
plt.subplot(1, 2, 1)
plt.plot(t, signal, label="Raw Signal")
plt.plot(gt_t, gt_sig, label="Ground Truth Signal", linestyle="--")
plt.title("Raw Signal and Ground Truth Signal")
plt.xlabel("Time (s)")
plt.ylabel("Signal Amplitude")
plt.xlim(min(t), max(t))
plt.ylim(min(signal) - 0.1, max(signal) + 0.1)
plt.legend()

# Reference Peak
plt.subplot(1, 2, 2)
plt.plot(ref_peak_t, ref_peak_sig, label="Reference Peak")
plt.title("Reference Peak")
plt.xlabel("Arbitrary Units")
plt.ylabel("Signal Amplitude")
plt.xlim(min(ref_peak_t), max(ref_peak_t))
plt.ylim(min(ref_peak_sig) - 0.1, max(ref_peak_sig) + 0.1)
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG file
output_path = "../images/raw_signal_and_ref_peak_plot.png"
plt.savefig(output_path, dpi=300)

# Display the plot
plt.show()
