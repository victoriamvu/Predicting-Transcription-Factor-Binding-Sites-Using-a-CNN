# Notebooks Directory

This directory contains Jupyter notebooks for data exploration, visualization, and analysis of results.

## Files

```
notebooks/
├── README.md             # This file
├── exploratory.ipynb     # Data exploration notebook
└── results.ipynb         # Results visualization notebook
```

## Notebook Descriptions

### exploratory.ipynb

This notebook explores the raw data to gain insights before model building.

**Contents:**
- JASPAR motif visualization
- Analysis of ChIP-seq peak distributions
- Investigation of sequence characteristics
- Data quality assessment
- Preliminary feature analysis

**Sections:**
1. Loading JASPAR motifs
2. Visualizing position weight matrices
3. Exploring ChIP-seq peak characteristics
4. Sequence composition analysis
5. GC content and other features
6. Sample size and class balance considerations

**Usage:**
Run this notebook before implementing `src/data.py` to understand data characteristics and inform design decisions.

### results.ipynb

This notebook analyzes and visualizes results after model training and evaluation.

**Contents:**
- Model performance analysis
- Comparison across different TFs
- Visualization of learned motifs
- Interpretation of model predictions
- Feature importance analysis

**Sections:**
1. Loading model evaluation results
2. Visualization of ROC and PR curves
3. Comparison of model performance across TFs
4. Visualization of learned motifs vs. JASPAR motifs
5. Analysis of influential sequence patterns
6. Case studies of correctly/incorrectly predicted sites

**Usage:**
Run this notebook after training and evaluating models to interpret results and gain biological insights.

## Running Notebooks

To run these notebooks:

```bash
# From the project root directory
jupyter notebook notebooks/
```

## Development Notes

- Notebooks should import functions from the `src` directory
- Keep data processing code in `src` modules, use notebooks primarily for visualization
- When adding new notebooks, update this README
- Use consistent plotting styles across notebooks
- Include markdown cells with explanations for better readability
- Save key figures to `notebooks/figures/` directory for use in reports
