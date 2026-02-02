# Warehouse UQ Analysis

MSc project - uncertainty quantification for warehouse order picking. Analyzes how likely we are to miss the 20-min SLA.

**Result:** ~7.3% failure probability. Order size and picking time are the main factors.

## Quick Start

### Run the simulation
```bash
cd warehouse_report
python3 uq_simulation.py
```

That's it. Script will:
1. Install dependencies automatically if needed
2. Run Monte Carlo (100k samples, ~30-60 sec)
3. Generate 4 figures (PNG)
4. Export results to CSV
5. Print summary to terminal

### Generate PDF report
```bash
cd warehouse_report
pdflatex main.tex
pdflatex main.tex
```

Produces 21-page report with all analysis.

## What's included

- **Python simulation** - FOSM, Monte Carlo, FORM methods
- **LaTeX report** - Full writeup with 33 references
- **Visualizations** - Tornado diagram, distributions, FORM plots
- **Data** - CSV exports of all results

## Model

Simple linear model:
```
Picking Time = (# items × time per item) + (distance × walking speed)
```

Inputs are random variables (Normal/Lognormal). SLA threshold = 20 minutes.

## Methods

- **FOSM** - Analytical approximation
- **Monte Carlo** - 100k samples
- **FORM** - Most probable failure point
- **Sensitivity** - Which variables matter most

## Results

| Method | P_f | β |
|--------|-----|---|
| Monte Carlo | 7.3% | 1.46 |
| FORM | 7.5% | 1.44 |

Variables ranked by impact:
1. Order size (51%)
2. Picking time (41%)
3. Walking distance (4%)
4. Walking speed (4%)

## Dependencies

Python packages (auto-installed):
- numpy, scipy, matplotlib, pandas, seaborn

LaTeX (for PDF):
- pdflatex with standard packages

---

**Jakub** | MSc Industrial Engineering | Feb 2026
