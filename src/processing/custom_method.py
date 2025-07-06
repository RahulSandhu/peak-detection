import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# Define Savitzky-Golay filter
def sgolay(sig, win_len, poly_order):
    """
    Apply Savitzky-Golay filter to smooth a signal.

    Parameters:
        sig (array): Input signal values to smooth.
        win_len (int): Length of the sliding window (must be odd).
        poly_order (int): Polynomial order for fitting within the window.

    Returns:
        smoothed_sig (array): Smoothed signal.

    Reference:
        Adapted version from
        https://gist.github.com/krvajal/1ca6adc7c8ed50f5315fee687d57c3eb
    """
    # Calculate half window size to center the sliding window
    half_win = (win_len - 1) // 2

    # Create the polynomial matrix for the window
    poly_mat = np.array(
        [[k**i for i in range(poly_order + 1)] for k in range(-half_win, half_win + 1)]
    )

    # Compute the coefficients
    coeff = np.linalg.pinv(poly_mat)[0]

    # Generate padding at the start to handle edge cases
    pad_start = sig[0] - np.abs(sig[1 : half_win + 1][::-1] - sig[0])

    # Generate padding at the end to handle edge cases
    pad_end = sig[-1] + np.abs(sig[-half_win - 1 : -1][::-1] - sig[-1])

    # Combine the original signal with the padded values
    padded_sig = np.concatenate((pad_start, sig, pad_end))

    # Apply the filter coefficients to the padded signal using convolution
    smoothed_sig = np.convolve(coeff[::-1], padded_sig, mode="valid")

    return smoothed_sig


# Define ALS baseline removal
def als(sig, lam, pen, max_iter):
    """
    Remove the baseline from a signal using Asymmetric Least Squares (ALS).

    Parameters:
        sig (array): Input signal for baseline correction.
        lam (float): Smoothing parameter for baseline estimation.
        pen (float): Penalty parameter controlling baseline asymmetry.
        max_iter (int): Maximum number of iterations for convergence.

    Returns:
        baseline (array): The computed baseline of the signal.

    Reference:
        Adapted version from
        https://nirpyresearch.com/two-methods-baseline-correction-spectral-data/
    """
    # Length of the signal
    n = len(sig)

    # Create second-order difference matrix for smoothing
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), format="csc")  # type: ignore[arg-type]

    # Initialize weights and baseline
    w = np.ones(n)
    baseline = np.zeros(n)

    # Iteratively update weights and compute the baseline
    for _ in range(max_iter):
        # Create diagonal weight matrix
        W = diags(w, 0, shape=(n, n), format="csc")

        # Compute system matrix for least squares solution
        Z = W + lam * D.T @ D

        # Solve the system to update the baseline
        baseline = spsolve(Z, w * sig)

        # Update weights based on current baseline
        w = pen * (sig > baseline) + (1 - pen) * (sig < baseline)

    return baseline


# Define peak detection
def peakdet(t, sig, th):
    """
    Detect local maxima and minima in a signal.

    Parameters:
        t (array): Time points corresponding to the signal.
        sig (array): Signal values to analyze for peaks.
        th (float): Threshold for peak detection.

    Returns:
        max_peaks (list): List of tuples (time, value) for detected maxima.
        min_peaks (list): List of tuples (time, value) for detected minima.

    Reference:
        Adapted version from
        https://billauer.co.il/blog/2009/01/peakdet-matlab-octave/
    """
    # Initialize lists for maxima and minima
    max_peaks = []
    min_peaks = []

    # Set initial values for minimum and maximum
    min_val, max_val = float("inf"), float("-inf")
    min_t, max_t = None, None

    # Start by looking for a maximum
    finding_max = True

    # Iterate through time and signal values
    for time, val in zip(t, sig):
        # Update maximum if a new higher value is found
        if val > max_val:
            max_val, max_t = val, time

        # Update minimum if a new lower value is found
        if val < min_val:
            min_val, min_t = val, time

        # Check for a peak based on the current mode (finding max or min)
        if finding_max:
            if val < max_val - th:
                # Record maximum and switch to finding a minimum
                max_peaks.append((max_t, max_val))
                min_val, min_t = val, time
                finding_max = False
        else:
            if val > min_val + th:
                # Record minimum and switch to finding a maximum
                min_peaks.append((min_t, min_val))
                max_val, max_t = val, time
                finding_max = True

    return max_peaks, min_peaks


def custom_method(sig, t, win_len, poly_order, lam, pen, max_iter, th):
    """
    Perform smoothing, baseline removal, and peak detection on a signal.

    Parameters:
        sig (array): Input signal.
        t (array): Time vector.
        win_len (int): Window length for Savitzky-Golay smoothing.
        poly_order (int): Polynomial order for Savitzky-Golay smoothing.
        lam (float): Smoothing parameter for ALS baseline removal.
        pen (float): Penalty parameter for ALS baseline removal.
        max_iter (int): Maximum iterations for ALS baseline removal.
        th (float): Threshold for peak detection.

    Returns:
        tuple: Smoothed signal, baseline signal, filtered signal, and detected
        peak times, peak values,
    """
    # Smooth the input signal using Savitzky-Golay filter
    smoothed_sig = sgolay(sig, win_len, poly_order)

    # Remove the baseline using ALS method
    baseline_sig = als(smoothed_sig, lam, pen, max_iter)

    # Subtract the baseline from the smoothed signal to get the filtered signal
    filtered_sig = smoothed_sig - baseline_sig

    # Detect peaks in the filtered signal
    max_peaks, _ = peakdet(t, filtered_sig, th)

    # Extract peak times and values from the detected peaks
    peak_t = [p[0] for p in max_peaks]
    peak_v = [p[1] for p in max_peaks]

    return smoothed_sig, baseline_sig, filtered_sig, peak_t, peak_v
