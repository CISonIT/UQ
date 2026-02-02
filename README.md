# Warehouse UQ Analysis

Uncertainty quantification for warehouse order picking. Probability of missing 20-min SLA.

## Usage

```bash
cd warehouse_report
python3 uq_simulation.py
```

Generates figures and CSV results. LaTeX source in `main.tex`.

## Results

Failure probability: 7.3%

Main factors: order size (51%), picking time (41%)
