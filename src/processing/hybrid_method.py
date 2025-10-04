import numpy as np
from scipy.signal import butter, convolve, filtfilt, find_peaks


def hybrid_method(sig, t, ref, fs, order, lc, hc, th):
    """
    Hybrid method for signal preprocessing and peak detection employing matched
    filtering and SciPy findpeaks.

    Parameters:
        sig (array): Input signal.
        t (array): Time values corresponding to the signal.
        ref (array): Reference peak window for matched filtering.
        fs (float): Sampling frequency.
        order (int): Order of the Butterworth filter.
        lc (float): Low cutoff frequency for the band-pass filter (Hz).
        hc (float): High cutoff frequency for the band-pass filter (Hz).
        th (float): Threshold factor to determine the peak detection threshold.

    Returns:
        tuple: Filtered signal, convolved signal, peak times, and peak amplitudes.
    """
    # Band-pass Butterworth filter
    b, a = butter(order, [lc / (fs / 2), hc / (fs / 2)], btype="band")  # type: ignore[arg-type]

    # Apply the filter to the signal
    filtered_sig = filtfilt(b, a, sig)

    # Set all negative values to zero
    filtered_sig[filtered_sig < 0] = 0

    # Apply the matched filter
    conv_sig = convolve(filtered_sig, ref, mode="full")

    # Detect peaks in the convolution signal
    peaks_conv, _ = find_peaks(conv_sig, height=np.max(conv_sig) * th)

    # Extract the detected peaks with corrected indices
    detected_peaks_conv_ind = peaks_conv - int(np.ceil(len(ref) / 2))
    peaks_t = t[detected_peaks_conv_ind]
    peaks_v = filtered_sig[detected_peaks_conv_ind]

    return filtered_sig, conv_sig, peaks_t, peaks_v
