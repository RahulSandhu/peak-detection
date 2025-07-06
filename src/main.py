import os

import numpy as np
from scipy.io import loadmat

from analysis.metrics import metrics
from processing.custom_method import custom_method
from processing.hybrid_method import hybrid_method
from processing.scipy_method import scipy_method

# 1. Load signals

# Load MAT file and extract data
data = loadmat("../data/signals/raw/data.mat")
t = data["t"].squeeze()
raw_sig = data["X"]
gt_sig = data["GT"]

# Load reference peak data
ref_peak_data = loadmat("../data/signals/ref_peak.mat")
ref_peak = ref_peak_data["xref"].flatten()

# Initialize storage dictionary
signals = {}

# Ensure output directories exist
os.makedirs("../data/signals/raw/", exist_ok=True)
os.makedirs("../data/signals/ground_truth/", exist_ok=True)

# Process and save signals
for i, (raw, gt) in enumerate(zip(raw_sig, gt_sig), start=1):
    # Extract sample name
    name = f"sample_{i:02d}"

    # Combine time and signals
    raw_data = np.column_stack((t, raw))
    gt_data = np.column_stack((t, gt))

    # File paths
    raw_path = f"../data/signals/raw/{name}.txt"
    gt_path = f"../data/signals/ground_truth/{name}.txt"

    # Save to files
    np.savetxt(raw_path, raw_data, delimiter=",", fmt="%.6f")
    np.savetxt(gt_path, gt_data, delimiter=",", fmt="%.6f")

    # Update dictionary
    signals[name] = {"time": t, "raw": raw, "gt": gt}

# 2. SciPy method

# Ensure output directories exist
os.makedirs("../data/peaks/scipy_peaks/", exist_ok=True)
os.makedirs("../data/metrics/scipy_metrics/", exist_ok=True)

# Process each signal using scipy_peakdet
for name, sig in signals.items():
    # Extract values
    raw = sig["raw"]
    t = sig["time"]
    gt = sig["gt"]

    # Detect SciPy peak detection
    peak_t, peak_v = scipy_method(raw, t, gt, fs=10, win_dur=500, th1=0.25, th2=0.15)

    # Save peak data
    peaks_path = f"../data/peaks/scipy_peaks/{name}.txt"
    np.savetxt(peaks_path, np.column_stack((peak_t, peak_v)), delimiter=",", fmt="%.6f")

    # Compute and save metrics
    gt_t, gt_v = t[gt > 0], gt[gt > 0]
    met = metrics(gt_t, gt_v, peak_t, peak_v, tol=0.5)
    met_arr = np.array([list(met.values())])
    met_path = f"../data/metrics/scipy_metrics/{name}.txt"
    np.savetxt(
        met_path,
        met_arr,
        delimiter=",",
        fmt="%.6f",
        header="Sensitivity,Specificity,Time_Accuracy,MAE_Intensity",
    )

# 3. Hybrid method

# Ensure output directories exist
os.makedirs("../data/signals/hybrid_method/filtered/", exist_ok=True)
os.makedirs("../data/signals/hybrid_method/convolved/", exist_ok=True)
os.makedirs("../data/peaks/hybrid_peaks/", exist_ok=True)
os.makedirs("../data/metrics/hybrid_metrics/", exist_ok=True)

# Process each signal
for name, sig in signals.items():
    # Extract values
    raw = sig["raw"]
    t = sig["time"]
    gt = sig["gt"]

    # Apply hybrid peak detection
    filtered_sig, conv_sig, peak_t, peak_v = hybrid_method(
        raw, t, ref_peak, fs=10, order=1, lc=0.01, hc=0.1, th=0.01
    )

    # Save intermediate results
    np.savetxt(
        f"../data/signals/hybrid_method/filtered/{name}.txt",
        np.column_stack((t, filtered_sig)),
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        f"../data/signals/hybrid_method/convolved/{name}.txt",
        conv_sig,
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        f"../data/peaks/hybrid_peaks/{name}.txt",
        np.column_stack((peak_t, peak_v)),
        delimiter=",",
        fmt="%.6f",
    )

    # Ground truth and metrics computation
    gt_t, gt_v = t[gt > 0], gt[gt > 0]
    hybrid_met = metrics(gt_t, gt_v, peak_t, peak_v, tol=0.5)
    hybrid_met_arr = np.array([list(hybrid_met.values())])
    np.savetxt(
        f"../data/metrics/hybrid_metrics/{name}.txt",
        hybrid_met_arr,
        delimiter=",",
        fmt="%.6f",
        header="Sensitivity,Specificity,Time_Accuracy,MAE_Intensity",
    )

# 4. Custom method

# Ensure output directories exist
os.makedirs("../data/signals/custom_method/smoothed", exist_ok=True)
os.makedirs("../data/signals/custom_method/baseline", exist_ok=True)
os.makedirs("../data/signals/custom_method/filtered", exist_ok=True)
os.makedirs("../data/peaks/custom_peaks", exist_ok=True)
os.makedirs("../data/metrics/custom_metrics", exist_ok=True)

# Process each signal
for name, sig in signals.items():
    # Extract values
    raw = sig["raw"]
    t = sig["time"]
    gt = sig["gt"]

    # Apply custom peak detection
    smoothed_sig, baseline_sig, filtered_sig, peak_t, peak_v = custom_method(
        raw, t, win_len=151, poly_order=3, lam=1e8, pen=0.001, max_iter=50, th=0.1
    )

    # Save intermediate results
    np.savetxt(
        f"../data/signals/custom_method/smoothed/{name}.txt",
        np.column_stack((t, smoothed_sig)),
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        f"../data/signals/custom_method/baseline/{name}.txt",
        np.column_stack((t, baseline_sig)),  # type: ignore[arg-type]
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        f"../data/signals/custom_method/filtered/{name}.txt",
        np.column_stack((t, filtered_sig)),
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        f"../data/peaks/custom_peaks/{name}.txt",
        np.column_stack((peak_t, peak_v)),
        delimiter=",",
        fmt="%.6f",
    )

    # Ground truth and metrics computation
    gt_t, gt_v = t[gt > 0], gt[gt > 0]
    custom_met = metrics(gt_t, gt_v, peak_t, peak_v, tol=0.5)
    custom_met_arr = np.array([list(custom_met.values())])
    np.savetxt(
        f"../data/metrics/custom_metrics/{name}.txt",
        custom_met_arr,
        delimiter=",",
        fmt="%.6f",
        header="Sensitivity,Specificity,Time_Accuracy,MAE_Intensity",
    )
