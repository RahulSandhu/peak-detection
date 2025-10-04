import os

import matplotlib.pyplot as plt
import numpy as np

# Use custom style
plt.style.use("../../config/matplotlib/mhedas.mplstyle")

# Define directories and metrics
metrics_dirs = {
    "custom_metrics": "../data/metrics/custom_metrics/",
    "scipy_metrics": "../data/metrics/scipy_metrics/",
    "hybrid_metrics": "../data/metrics/hybrid_metrics/",
}

metrics_dicts = {
    "custom_metrics": {
        key: []
        for key in ["sensitivity", "specificity", "time_accuracy", "mae_intensity"]
    },
    "scipy_metrics": {
        key: []
        for key in ["sensitivity", "specificity", "time_accuracy", "mae_intensity"]
    },
    "hybrid_metrics": {
        key: []
        for key in ["sensitivity", "specificity", "time_accuracy", "mae_intensity"]
    },
}

titles = ["Sensitivity", "Specificity", "Time Accuracy", "MAE Intensity"]

# Load all metrics
for metric_type, metric_dir in metrics_dirs.items():
    for file_name in sorted(os.listdir(metric_dir)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(metric_dir, file_name)
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            for i, key in enumerate(metrics_dicts[metric_type].keys()):
                metrics_dicts[metric_type][key].append(data[0, i])

# Define metrics figure
fig_metrics, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Plot each metric
for i, metric in enumerate(
    ["sensitivity", "specificity", "time_accuracy", "mae_intensity"]
):
    # Extract values
    ax = axes[i]
    values_custom = metrics_dicts["custom_metrics"][metric]
    values_scipy = metrics_dicts["scipy_metrics"][metric]
    values_hybrid = metrics_dicts["hybrid_metrics"][metric]

    # Compute mean and standard deviation
    mean_custom = np.nanmean(values_custom)
    std_custom = np.nanstd(values_custom)
    mean_scipy = np.nanmean(values_scipy)
    std_scipy = np.nanstd(values_scipy)
    mean_hybrid = np.nanmean(values_hybrid)
    std_hybrid = np.nanstd(values_hybrid)

    # Define x-values
    x_values = range(1, len(values_custom) + 1)

    # Plot values
    ax.plot(
        x_values,
        values_custom,
        color="red",
        marker="o",
        alpha=0.7,
        label=f"Custom Metrics: {mean_custom:.3f} ± {std_custom:.3f}",
    )
    ax.plot(
        x_values,
        values_scipy,
        color="blue",
        marker="x",
        alpha=0.7,
        label=f"Scipy Metrics: {mean_scipy:.3f} ± {std_scipy:.3f}",
    )
    ax.plot(
        x_values,
        values_hybrid,
        color="green",
        marker="s",
        alpha=0.7,
        label=f"Hybrid Metrics: {mean_hybrid:.3f} ± {std_hybrid:.3f}",
    )

    # Plot shading for mean ± std
    ax.fill_between(
        x_values,
        mean_custom - std_custom,
        mean_custom + std_custom,
        color="red",
        alpha=0.2,
    )
    ax.fill_between(
        x_values,
        mean_scipy - std_scipy,
        mean_scipy + std_scipy,
        color="blue",
        alpha=0.2,
    )
    ax.fill_between(
        x_values,
        mean_hybrid - std_hybrid,
        mean_hybrid + std_hybrid,
        color="green",
        alpha=0.2,
    )

    # Plot mean lines
    ax.axhline(mean_custom, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(mean_scipy, color="blue", linestyle="--", linewidth=1.5)
    ax.axhline(mean_hybrid, color="green", linestyle="--", linewidth=1.5)

    # Customize plot
    ax.set_title(titles[i])
    ax.set_xlabel("Signal Index")
    ax.set_ylabel("Metric Value (s)" if metric == "time_accuracy" else "Metric Value")
    ax.legend(loc="best")

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG file
output_path = "../images/metrics_plot.png"
plt.savefig(output_path, dpi=300)

# Display the plot
plt.show()
