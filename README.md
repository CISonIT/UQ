# Warehouse Order Picking - Uncertainty Quantification & Reliability Analysis

**MSc-level project analyzing warehouse order picking performance under uncertainty**

[![License](https://img.shields.io/badge/License-Academic-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7+-green.svg)](https://python.org)
[![LaTeX](https://img.shields.io/badge/LaTeX-Document-orange.svg)](warehouse_report/main.pdf)

## Overview

This project applies advanced **Uncertainty Quantification (UQ)** and **Reliability Analysis** methods to assess warehouse order picking performance. We answer the critical question: *"What's the probability we'll miss our Service Level Agreement (SLA)?"*

### Key Results

- **Failure Probability**: ~7.3% (orders exceeding 20-minute SLA)
- **Reliability Index (β)**: 1.44
- **Critical Factors**: 
  - Order lines: 51% contribution to failure
  - Picking time per line: 41% contribution

## What's Included

### 📄 Documentation
- **[main.pdf](warehouse_report/main.pdf)** - Complete 21-page technical report
- **[ENHANCED_PROJECT_SUMMARY.md](ENHANCED_PROJECT_SUMMARY.md)** - Quick project overview
- **[PYTHON_README.md](warehouse_report/PYTHON_README.md)** - Python implementation guide

### 🐍 Python Simulation
- **Monte Carlo Simulation** (100,000 samples)
- **FOSM** (First-Order Second-Moment) method
- **FORM** (First-Order Reliability Method)
- **Sensitivity Analysis**
- Professional visualizations (300 DPI)

### 📊 Visualizations
- 9-panel comprehensive UQ analysis
- Tornado diagram (sensitivity ranking)
- Distribution comparisons (PDF & CDF)
- FORM analysis in standard normal space

### 📚 Course Materials
- Full set of UQ lecture notes (UQGdansk series)
- Assignment descriptions
- Fundamentals summary

## Quick Start

### View the Report
```bash
cd warehouse_report
evince main.pdf  # or your preferred PDF viewer
```

### Run the Simulation
```bash
cd warehouse_report
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python uq_simulation.py
```

**Runtime**: ~30-60 seconds (100k MC samples)

### Recompile the Document
```bash
cd warehouse_report
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

## Project Structure

```
UQ/
├── warehouse_report/          # Main project directory
│   ├── main.tex              # LaTeX source (21 pages)
│   ├── main.pdf              # ✨ Final PDF report
│   ├── uq_simulation.py      # Python implementation
│   ├── requirements.txt      # Dependencies
│   ├── *.png                 # Visualizations (4 figures)
│   └── *.csv                 # Results data
│
├── *.pdf                     # Course materials
├── ENHANCED_PROJECT_SUMMARY.md
└── README.md                 # This file
```

## Methodology

### Mathematical Model
```
Y = N × t_p + D × t_w
```

Where:
- **N**: Number of order lines (Normal, μ=20, CoV=0.20)
- **t_p**: Picking time per line (Lognormal, μ=30s, CoV=0.15)
- **D**: Walking distance (Normal, μ=300m, CoV=0.10)
- **t_w**: Walking time per meter (Lognormal, μ=1.2 s/m, CoV=0.10)

**SLA Threshold**: 1200 seconds (20 minutes)

### Analysis Methods

1. **FOSM**: Fast analytical approximation
2. **Monte Carlo**: Brute-force sampling (100k samples)
3. **FORM**: Find most probable failure point
4. **Sensitivity**: Identify critical variables

## Academic Context

**Suitable for:**
- ✅ MSc Industrial Engineering coursework
- ✅ Production Management courses
- ✅ Reliability Analysis courses
- ✅ Operations Research applications
- ✅ Portfolio demonstrations

**Features:**
- 40 peer-reviewed references (EJOR, IJPR, Structural Safety)
- Proper mathematical formulations
- Professional visualizations
- Complete reproducibility

## Dependencies

### Python
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- seaborn >= 0.11.0

### LaTeX
- pdflatex (TeX Live, MiKTeX, or MacTeX)
- Standard packages (amsmath, hyperref, graphicx, etc.)

## Results Summary

| Method | Mean [s] | Std Dev [s] | P_f [%] | β |
|--------|----------|-------------|---------|---|
| FOSM | 960 | 158 | - | 1.77 |
| Monte Carlo | 960 | 159 | 7.26 | 1.46 |
| FORM | - | - | 7.51 | 1.44 |

## Managerial Insights

**Where to focus improvements:**
1. **Order batching** - Reduce variability in order sizes
2. **Picking standardization** - Voice-directed or light-guided systems
3. **Ergonomics & training** - Improve picker consistency

**Lower priority:**
- Walking distance optimization (only 3.7% impact)
- Walking speed improvements (only 3.9% impact)

## Author

**Jakub** - MSc Student  
*Industrial Engineering / Production Management*

## License

Academic use only. Please cite appropriately if used in coursework or research.

## Acknowledgments

- Course materials from UQ Gdansk series
- Python scientific computing stack (NumPy, SciPy, Matplotlib)
- LaTeX community

---

**Last Updated**: February 2026  
**Status**: ✅ Complete & Production-Ready
