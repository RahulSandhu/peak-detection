import numpy as np
from scipy.signal import find_peaks


def scipy_method(sig, t, gt_sig, fs, win_dur, th1, th2):
    """
    Detect peaks in a signal using sliding windows.

    Parameters:
        sig (array): Input signal.
        t (array): Time vector for the signal.
        gt_sig (array): Ground truth signal for peak detection.
        fs (float): Sampling frequency (Hz).
        win_dur (int): Window size in seconds.
        th1 (float): Threshold factor for peak height.
        th2 (float): Threshold factor for peak prominence.

    Returns:
        tuple: Lists of detected peak times and peak values.
    """
    # Initialize lists for detected peaks
    peak_t = []
    peak_v = []

    # Calculate window duration in samples
    win_size = fs * win_dur

    # Extract ground truth times
    gt_t = t[np.where(gt_sig > 0)]

    # Calculate the number of windows
    num_windows = (len(t) + win_size - 1) // win_size

    # Loop over each window
    for i in range(num_windows):
        # Define window range
        start_idx = i * win_size
        end_idx = min((i + 1) * win_size, len(t))
        win_sig = sig[start_idx:end_idx]
        win_t = t[start_idx:end_idx]

        # Check if there are any ground truth peaks in the current window
        gt_idx_window = np.where((gt_t >= win_t[0]) & (gt_t <= win_t[-1]))[0]

        if len(gt_idx_window) == 0:
            # No ground truth peaks in this window, skip it
            continue
        elif len(gt_idx_window) == 1:
            # One ground truth peak: use half max_v as height and ignore distance
            max_v = np.max(win_sig)
            min_h = th1 * max_v
            peaks, _ = find_peaks(win_sig, height=min_h, prominence=th2 * max_v)
        else:
            # Two or more ground truth peaks: use height, distance, and prominence
            max_v = np.max(win_sig)
            min_h = th1 * max_v
            mean_diff = np.mean(np.diff(gt_t[gt_idx_window]))
            min_d = max(int(mean_diff * fs), 1)
            peaks, _ = find_peaks(
                win_sig, height=min_h, distance=min_d, prominence=th2 * max_v
            )

        # Append detected peaks to the lists
        peak_t.extend(win_t[peaks])
        peak_v.extend(win_sig[peaks])

    return peak_t, peak_v
