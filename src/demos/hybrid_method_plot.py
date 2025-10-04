import matplotlib.pyplot as plt
import numpy as np

# Use custom style
plt.style.use("../../config/matplotlib/mhedas.mplstyle")

# File paths
raw_file_path = "../data/signals/raw/sample_01.txt"
ground_truth_file_path = "../data/signals/ground_truth/sample_01.txt"
filtered_file_path = "../data/signals/hybrid_method/filtered/sample_01.txt"
convolved_file_path = "../data/signals/hybrid_method/convolved/sample_01.txt"
hybrid_peaks_file_path = "../data/peaks/hybrid_peaks/sample_01.txt"


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

# Load hybrid filtered signal
filtered_sig = np.loadtxt(filtered_file_path, delimiter=",")[:, 1]

# Load convolved signal
conv_sig = np.loadtxt(convolved_file_path)
height = np.max(conv_sig) * 0.01

# Load hybrid peaks
hybrid_peaks = np.loadtxt(hybrid_peaks_file_path, delimiter=",")
peak_t = hybrid_peaks[:, 0]
peak_v = hybrid_peaks[:, 1]

# Compute Power Spectral Density (PSD)
freqs = np.fft.rfftfreq(len(signal), d=np.mean(np.diff(t)))
psd = np.abs(np.fft.rfft(signal)) ** 2

# Define figure
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Power Spectral Density (PSD)
ax1.plot(freqs, psd, label="PSD")
ax1.set_xlim(-0.001, 0.015)
ax1.set_title("Power Spectral Density")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Power")
ax1.legend()

# Raw and Filtered Signal
ax2.plot(t, signal, label="Raw Signal")
ax2.plot(t, filtered_sig, label="Filtered Signal")
ax2.set_title("Raw and Filtered Signal")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Signal Amplitude")
ax2.set_xlim(min(t), max(t))
ax2.legend()

# Convolved signal
ax3.plot(conv_sig, label="Convolved Signal")
ax3.axhline(y=height, color="black", linestyle="--", label="Threshold")
ax3.set_title("Convolved Signal")
ax3.set_xlabel("Sample")
ax3.set_ylabel("Signal Amplitude")
ax3.set_xlim(0, len(conv_sig) - 1)
ax3.legend()

# Ground Truth and Detected Peaks
ax4.stem(
    gt_t,
    gt_sig[gt_sig > 0],
    linefmt="orange",
    markerfmt="o",
    basefmt=" ",
    label="Ground Truth Peaks",
)
ax4.stem(
    peak_t,
    peak_v,
    linefmt="red",
    markerfmt="o",
    basefmt=" ",
    label="Detected Peaks",
)
ax4.set_title("Ground Truth and Detected Peaks")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Signal Amplitude")
ax4.legend()

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG file
output_path = "../images/hybrid_method_plot.png"
plt.savefig(output_path, dpi=300)

# Display the plot
plt.show()
