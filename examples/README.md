# Examples

This directory contains example scripts and advanced usage patterns for the Dynamic Influence-Based Clustering framework.

## Files

- **`examples_paper_experiments.py`**: Comprehensive experiment runner that reproduces all experiments from the paper. This is a complex script with many configuration options for systematic evaluation across multiple datasets, influence methods, and baselines.

## Usage

### Paper Experiments Reproduction

To reproduce the experiments from the TNNLS paper:

```bash
cd examples/
python examples_paper_experiments.py --dataset energy_data --output_dir ../results/
```

This script includes:
- All baseline comparisons (k-Shape, DTW K-medoids, standard clustering)
- Multiple influence methods (SHAP, LIME, Integrated Gradients, Spearman)
- Comprehensive evaluation metrics
- Statistical significance testing
- Parallel processing support

### For Regular Usage

For simpler experiments, use the main `run_experiments.py` script in the root directory:

```bash
# Simple experiment
python run_experiments.py --dataset your_data.csv --influence spearman

# Compare all methods
python run_experiments.py --dataset your_data.csv --compare_all
```

## Note

The examples in this directory are more complex and designed for research purposes. For everyday usage, the scripts in the main directory (`demo.py`, `run_experiments.py`) are more appropriate.
