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

## 💻 Source

* `src/main.py`: Entry point to execute all methods and generate results
* `src/processing/`: Contains the SciPy, hybrid, and custom peak detection methods
* `src/analysis/`: Computes metrics and generates comparison plots
* `src/demos/`: (Optional) Exploratory scripts

## 📁 Data

* `data/raw/`: Synthetic noisy signals
* `data/ground_truth/`: Reference peak annotations
* `data/ref_peak.mat`: Template for matched filtering (hybrid method)

## 📊 Results

* `results/metrics/`: Quantitative evaluation of all methods
* `results/peaks/`: Detected peaks from each approach
* **Summary:** According to the final performance comparison in the study, the
**custom method** showed the most consistent results across all metrics,
outperforming both the SciPy and hybrid approaches in overall reliability.

## 🎓 Acknowledgments

* Developed as part of the Health Data Science Master’s program at Universitat
Rovira i Virgili (URV)

</div>
