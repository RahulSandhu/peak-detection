<div align="justify">

# Peak Detection in Synthetic Noisy Signals

This repository contains a complete workflow for evaluating and comparing
multiple peak detection techniques on synthetic noisy signals. The project
includes custom signal processing pipelines, experimental results, plots, and a
presentation analyzing three approaches: SciPy-based, hybrid matched filtering,
and a fully custom method.

## 🚀 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/RahulSandhu/peak-detection
   cd peak-detection
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Dataset

The dataset consists of synthetic noisy signals with corresponding ground truth
peak annotations. The data is organized as follows:

- `data/raw/`: Synthetic noisy signals
- `data/ground_truth/`: Reference peak annotations
- `data/ref_peak.mat`: Template for matched filtering (hybrid method)

## 📊 Results

According to the final performance comparison, the **custom method** showed the
most consistent results across all metrics, outperforming both the SciPy and
hybrid approaches in overall reliability.

All quantitative evaluations and detected peaks from each approach are
available in:

- `results/metrics/`: Performance metrics for all methods
- `results/peaks/`: Detected peaks from each approach

## 🎓 Acknowledgments

- Developed as part of the Biomedical Sensors and Signal Processing course of
  the Master in Health Data Science at Universitat Rovira i Virgili (URV)

</div>
