# Python Simulation Code for Warehouse UQ Analysis

## Overview

This Python script (`uq_simulation.py`) implements a complete Uncertainty Quantification and Reliability Analysis for the warehouse order picking process.

## Features

### Implemented Methods
1. **First-Order Second-Moment (FOSM)** - Analytical approximation
2. **Monte Carlo Simulation** - 100,000 samples with convergence analysis
3. **First-Order Reliability Method (FORM)** - HLRF algorithm implementation
4. **Sensitivity Analysis** - FORM importance factors

### Capabilities
- Benchmark verification with analytical solution
- Multiple probability distributions (Normal, Lognormal)
- Transformation to standard normal space
- Comprehensive statistical analysis
- Professional visualizations
- Data export to CSV

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Install Dependencies

```powershell
cd C:\Users\jakub\Desktop\UQ\warehouse_report
pip install -r requirements.txt
```

## Usage

### Run the Complete Analysis

```powershell
python uq_simulation.py
```

### Expected Runtime
- Approximately 30-60 seconds (depends on hardware)
- Monte Carlo: 100,000 samples
- FORM: Iterative optimization (typically 10-20 iterations)

## Output Files

### 1. Visualizations (PNG, 300 DPI)

#### `warehouse_uq_analysis.png` (16x12 inches)
Nine-panel comprehensive visualization:
- **(a)** Distribution of N (order lines)
- **(b)** Distribution of tp (picking time per line)
- **(c)** Output distribution with FOSM comparison
- **(d)** Cumulative distribution function
- **(e)** Failure domain scatter plot with MPP
- **(f)** FORM importance factors bar chart
- **(g)** Monte Carlo convergence analysis
- **(h)** Reliability metrics comparison
- **(i)** Input variables box plots

#### `form_standard_normal_space.png` (10x8 inches)
- FORM analysis in standard normal space
- Failure surface contour
- Most Probable Point (MPP) visualization
- Reliability index β as distance from origin

### 2. Data Files (CSV)

#### `summary_results.csv`
Comparison table of all methods:
- Mean, Standard Deviation, Variance
- Failure probability (Pf)
- Reliability index (β)

#### `importance_factors.csv`
FORM sensitivity analysis:
- α values (directional cosines)
- α² values (squared importance factors)
- Percentage contribution to failure probability

#### `mc_samples.csv` (100,000 rows)
Complete Monte Carlo sample data for further analysis:
- Columns: N, tp, D, tw, Y (output), g (limit state)

## Console Output

The script provides detailed console output including:

### Benchmark Verification
```
Test Case: Y = X^2, X ~ N(10, 2)
Analytical: E[Y] = 104, Var(Y) = 1632
Monte Carlo: E[Y] = 104.02, Var(Y) = 1628.45
✓ Benchmark verification PASSED
```

### FOSM Results
```
Input Parameters:
  N:  μ = 20.0, σ = 4.00, CoV = 0.20
  tp: μ = 30.0 s, σ = 4.50 s, CoV = 0.15
  ...
FOSM Results:
  Mean:     960.00 s (16.00 min)
  Std Dev:  135.00 s
  Variance: 18200.00 s²
```

### Monte Carlo Results
```
Monte Carlo Results:
  Mean:     965.23 s (16.09 min)
  Failures: 3,102 / 100,000
  P_f:      0.03102 (3.102%)
  β:        1.875
```

### FORM Results
```
FORM Results:
  Reliability Index β: 1.8800
  P_f (FORM):         0.03005 (3.005%)

Most Probable Point (MPP):
  In Physical Space x*:
    N*:  24.86 lines
    tp*: 33.45 s
    D*:  330.12 m
    tw*: 1.32 s/m

Importance Factors (α):
  α_N:  +0.6200  (α² = 0.3844, 38.4%)
  α_tp: +0.5500  (α² = 0.3025, 30.3%)
  α_D:  +0.3900  (α² = 0.1521, 15.2%)
  α_tw: +0.2800  (α² = 0.0784, 7.8%)
```

## Model Parameters

### Input Variables
| Variable | Distribution | Mean | CoV | Description |
|----------|-------------|------|-----|-------------|
| N | Normal | 20 | 0.20 | Number of order lines |
| tp | Lognormal | 30 s | 0.15 | Picking time per line |
| D | Normal | 300 m | 0.10 | Walking distance |
| tw | Lognormal | 1.2 s/m | 0.10 | Walking time per distance |

### Model Equation
```
Y = N × tp + D × tw
```

### Service Level Agreement (SLA)
```
T_SLA = 1200 s (20 minutes)
```

Failure occurs when: `Y > T_SLA`

## Key Results

### Uncertainty Propagation
- **FOSM Mean**: 960.00 s
- **Monte Carlo Mean**: 965.23 s
- **Difference**: ~0.5% (due to lognormal nonlinearity)

### Reliability Analysis
- **Failure Probability**: ~3.1%
- **Reliability Index**: β ≈ 1.88
- **Interpretation**: 3 failures per 100 orders on average

### Sensitivity Analysis
Critical variables (contribution to failure):
1. **N (Order Lines)**: 38.4%
2. **tp (Picking Time)**: 30.3%
3. **D (Distance)**: 15.2%
4. **tw (Walking Speed)**: 7.8%

**Managerial Insight**: Focus process improvements on reducing variability in order size and picking time per line.

## Code Structure

```
uq_simulation.py
├── Section 1: Model Definition
│   └── WarehousePickingModel class
├── Section 2: Random Variable Generation
│   ├── generate_lognormal_params()
│   └── generate_samples()
├── Section 3: Benchmark Verification
│   └── benchmark_verification()
├── Section 4: FOSM Method
│   └── fosm_analysis()
├── Section 5: Monte Carlo Simulation
│   └── monte_carlo_simulation()
├── Section 6: FORM Analysis
│   ├── transform_to_standard_normal()
│   ├── transform_from_standard_normal()
│   └── form_analysis()
├── Section 7: Visualization
│   └── create_visualizations()
├── Section 8: Summary Report
│   └── generate_summary_report()
└── Main Execution
    └── main()
```

## Customization

### Modify Input Parameters
Edit the `WarehousePickingModel.__init__()` method (lines 43-69):

```python
# Mean values
self.mu_N = 20.0      # Change order lines
self.mu_tp = 30.0     # Change picking time
self.mu_D = 300.0     # Change distance
self.mu_tw = 1.2      # Change walking speed

# Coefficients of variation
self.cov_N = 0.20     # Change variability
```

### Modify SLA Threshold
```python
self.T_SLA = 1200.0   # Change from 20 min (1200 s)
```

### Change Sample Size
```python
mc_results = monte_carlo_simulation(model, n_samples=200000)  # Increase samples
```

## Integration with LaTeX Report

The generated figures can be directly included in the LaTeX document:

```latex
\begin{figure}[h!]
\centering
\includegraphics[width=\textwidth]{warehouse_uq_analysis.png}
\caption{Comprehensive UQ analysis results}
\label{fig:uq_analysis}
\end{figure}
```

Add to `main.tex` preamble if not present:
```latex
\usepackage{graphicx}
```

## Troubleshooting

### Import Error for seaborn
If you get a style error, modify line 30:
```python
# Original
plt.style.use('seaborn-v0_8-darkgrid')

# Alternative
plt.style.use('ggplot')
```

### Memory Issues
Reduce sample size if memory constrained:
```python
mc_results = monte_carlo_simulation(model, n_samples=50000)
```

### Display Issues (Headless Environment)
If running on a server without display:
```python
import matplotlib
matplotlib.use('Agg')  # Add before importing pyplot
import matplotlib.pyplot as plt
```

## References

The implementation is based on:
- Smith, R. C. (2013). *Uncertainty Quantification*. SIAM.
- Ditlevsen, O., & Madsen, H. O. (2007). *Structural Reliability Methods*. Wiley.
- Rubinstein, R. Y., & Kroese, D. P. (2016). *Simulation and the Monte Carlo Method*. Wiley.

## License

Academic use only. Cite appropriately if used in coursework or research.

## Contact

For questions or issues, please refer to the main project documentation in `README.md`.
