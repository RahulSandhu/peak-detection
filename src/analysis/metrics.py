import numpy as np


def metrics(gt_t, gt_a, det_t, det_a, tol):
    """
    Compute sensitivity, specificity, time accuracy, and MAE of peak
    intensities.

    Parameters:
        - gt_t: Ground truth times.
        - gt_a: Ground truth amplitudes.
        - det_t: Detected times.
        - det_a: Detected amplitudes.
        - tol: Tolerance for time matching.

    Returns:
        - Dictionary with sensitivity, specificity, time_accuracy, and
          MAE_intensity.
    """

    # Preallocate arrays for matched indices
    tp_idx, matched_idx = [], []

    # Match detected peaks to ground truth peaks within tolerance
    for i, gt in enumerate(gt_t):
        diff = np.abs(det_t - gt)
        closest = np.argmin(diff)
        if diff[closest] <= tol:
            tp_idx.append(i)
            matched_idx.append(closest)

    # Calculate sensitivity and specificity
    tp = len(tp_idx)
    fn = len(gt_t) - tp
    fp = len(det_t) - len(set(matched_idx))
    sens = tp / (tp + fn) if tp + fn > 0 else 0
    spec = tp / (tp + fp) if tp + fp > 0 else 0

    # Compute time accuracy (mean absolute error of matched times)
    time_err = [np.abs(gt_t[i] - det_t[matched_idx[j]]) for j, i in enumerate(tp_idx)]
    time_acc = np.mean(time_err) if time_err else np.nan

    # Compute mean absolute error of amplitudes for matched peaks
    amp_err = [np.abs(gt_a[i] - det_a[matched_idx[j]]) for j, i in enumerate(tp_idx)]
    mae_amp = np.mean(amp_err) if amp_err else np.nan

    return {
        "sensitivity": sens,
        "specificity": spec,
        "time_accuracy": time_acc,
        "mae_intensity": mae_amp,
    }
