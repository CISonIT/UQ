# Warehouse UQ Analysis

My MSc project on uncertainty quantification for warehouse order picking. Basically trying to answer: how likely are we to miss the SLA?

## What is this?

Applied UQ and reliability methods to warehouse picking. Used FOSM, Monte Carlo (100k samples), and FORM to analyze failure probability.

**Bottom line:** ~7.3% chance of missing the 20-min SLA. Order size and picking time per item are what really matter.

## What's in here

- LaTeX report (21 pages) - compile with `pdflatex main.tex`
- Python simulation - runs all the analysis
- Visualizations - tornado diagram, distributions, FORM plots
- Data exports - CSV files with results

## Running it

### Simulation
```bash
cd warehouse_report
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python uq_simulation.py
```

Takes about 30-60 sec to run 100k Monte Carlo samples.

### LaTeX doc
```bash
cd warehouse_report
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

## Structure

```
warehouse_report/
├── main.tex              # LaTeX source
├── uq_simulation.py      # Python code
├── *.png                 # Figures
└── *.csv                 # Results
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
