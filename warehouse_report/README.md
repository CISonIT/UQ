# Uncertainty Quantification and Reliability Analysis of a Warehouse Order Picking Process

## Project Description

This directory contains a comprehensive, academically rigorous LaTeX report on Uncertainty Quantification (UQ) and Reliability Analysis applied to warehouse order picking operations. The report has been significantly expanded from the original 7-page document to a 10-12 page technical document suitable for MSc-level coursework in Industrial Engineering, Production Management, or Reliability Engineering.

## Document Features

### Content Enhancements
- **Expanded Introduction**: Detailed industrial context linking warehouse operations to production flow, takt time, and service level agreements
- **Enhanced Uncertainty Modeling**: In-depth discussion of aleatory vs. epistemic uncertainty with Maximum Entropy principle justification
- **Rigorous Reliability Analysis**: Comprehensive FORM theory with geometric interpretation and standard normal space transformation
- **Sensitivity Analysis**: Managerial interpretation of importance factors with warehouse design implications
- **Extended Conclusions**: Includes managerial implications, decision-support relevance, limitations, and future research directions

### Academic Quality
- **25 credible scientific references** including:
  - UQ and reliability textbooks (Smith, Ditlevsen & Madsen, Rubinstein & Kroese)
  - Warehouse logistics literature (De Koster et al., Tompkins et al., Boysen et al.)
  - Production engineering sources
- **Formal scientific tone** appropriate for technical university reports
- **No placeholders** - all content is complete and academically sound
- **Consistent equations and symbols** - all numerical values preserved from original

## Files in This Directory

### LaTeX Files
- `main.tex` - Main LaTeX document (complete, ready to compile)
- `README.md` - This file (main documentation)

### Python Simulation Files
- `uq_simulation.py` - Complete Python implementation (770 lines)
- `requirements.txt` - Python dependencies
- `PYTHON_README.md` - Detailed Python documentation
- `run_simulation.bat` - Windows batch file to run simulation

### Generated Files (after running Python simulation)
- `warehouse_uq_analysis.png` - Main visualization (9 subplots)
- `form_standard_normal_space.png` - FORM space visualization
- `summary_results.csv` - Results comparison table
- `importance_factors.csv` - Sensitivity analysis data
- `mc_samples.csv` - Monte Carlo sample data (100,000 rows)

## Compilation Instructions

### Prerequisites
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- PDFLaTeX compiler

### Required LaTeX Packages
The following packages are used (all standard and included in most distributions):
- `inputenc` - UTF-8 input encoding
- `babel` - English language support
- `amsmath`, `amsfonts`, `amssymb` - Mathematical typesetting
- `graphicx` - Graphics support
- `geometry` - Page layout
- `hyperref` - Hyperlinks and PDF features
- `cite` - Citation management
- `float` - Float placement
- `booktabs` - Professional tables

### Compilation Steps

#### Windows (PowerShell/Command Prompt)
```powershell
cd C:\Users\jakub\Desktop\UQ\warehouse_report
pdflatex main.tex
pdflatex main.tex
```

#### Alternative: Using LaTeX Editor
1. Open `main.tex` in your preferred LaTeX editor (TeXworks, TeXstudio, Overleaf, etc.)
2. Compile using PDFLaTeX
3. Compile twice to resolve cross-references and table of contents

### Why Two Compilations?
LaTeX requires two compilation passes to:
1. First pass: Generate auxiliary files (.aux, .toc)
2. Second pass: Resolve all cross-references, citations, and table of contents

## Python Simulation

### Quick Start
1. **Install Python** (if not already installed):
   - Download from https://www.python.org/downloads/
   - Or install from Microsoft Store

2. **Run the simulation**:
   ```powershell
   # Navigate to project directory
   cd C:\Users\jakub\Desktop\UQ\warehouse_report
   
   # Option 1: Use the batch file (recommended)
   run_simulation.bat
   
   # Option 2: Run manually
   pip install -r requirements.txt
   python uq_simulation.py
   ```

3. **Generated outputs**:
   - 2 high-resolution PNG visualizations
   - 3 CSV data files for further analysis
   - Comprehensive console output with results

### Simulation Features
- **Complete implementation** of all methods from the LaTeX report
- **Monte Carlo Simulation** with 100,000 samples
- **FORM Analysis** with iterative HLRF algorithm
- **FOSM Method** for analytical approximation
- **Professional visualizations** ready for publication
- **Data export** in CSV format for further analysis

### Expected Runtime
- Approximately 30-60 seconds depending on hardware
- All computations are vectorized for optimal performance

See `PYTHON_README.md` for detailed documentation, code structure, and customization options.

## Document Structure

1. **Introduction (3+ pages)**
   - Industrial context and motivation
   - Uncertainty in warehouse operations
   - Objectives and scope

2. **Problem Definition (1.5 pages)**
   - Physical system description
   - Mathematical model
   - Modeling assumptions and limitations
   - Quantity of interest
   - Deterministic solution

3. **Uncertainty Modeling (1.5 pages)**
   - Classification of uncertainty
   - Distribution selection with Maximum Entropy justification
   - Independence assumption
   - Statistical moments

4. **Uncertainty Propagation (2 pages)**
   - Benchmark verification with analytical solution
   - FOSM method with detailed theory
   - Monte Carlo simulation with convergence analysis
   - Results comparison

5. **Reliability Analysis (1.5 pages)**
   - Limit state function
   - Probability of failure
   - FORM analysis with geometric interpretation

6. **Sensitivity Analysis (1 page)**
   - FORM importance factors
   - Managerial interpretation and decision support

7. **Conclusions (1.5 pages)**
   - Summary of findings
   - Decision-support relevance
   - Limitations and modeling assumptions
   - Future research directions

8. **References (1 page)**
   - 25 academically credible citations

## Target Length

When compiled with 12pt font on A4 paper, the document is approximately **10-12 pages**, meeting the requirements for a comprehensive technical report suitable for high-grade evaluation in Industrial Engineering coursework.

## Key Features Preserved

All original content has been preserved:
- All equations and mathematical formulations
- All numerical values and symbols
- All assumptions and model parameters
- All results (FOSM, Monte Carlo, FORM)

## Academic Level

This report is suitable for:
- MSc-level coursework in Industrial Engineering
- Production Management courses
- Reliability Analysis courses
- Uncertainty Quantification courses
- Graduate-level technical reports

## Contact & Attribution

This report was generated based on the original document "Uncertainty_Quantification_and_Reliability_Analysis.pdf" with significant academic expansion and enhancement.

## License

Academic use only. Please cite appropriately if used in coursework or research.