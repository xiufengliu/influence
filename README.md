# Dynamic Influence-Based Clustering Framework

This project implements the Dynamic Influence-Based Clustering Framework for energy consumption analysis as described in the paper. The framework transforms raw energy consumption data into an influence space using explainable machine learning (XML) methods, performs clustering in this space, and analyzes temporal transitions between clusters.

## Overview

Energy consumption patterns are inherently dynamic, influenced by temporal, contextual, and behavioral factors. Traditional clustering approaches rely on static representations of raw data, often failing to capture the evolving relationships between features and their impact on energy usage. This framework addresses these limitations by:

1. Transforming raw data into an influence space derived from feature importance explanations
2. Performing clustering in this influence space to discover interpretable subgroups
3. Analyzing temporal transitions to track subgroup evolution over time
4. Detecting anomalies in consumption patterns

## Features

- **Influence Space Transformation**: Convert raw features to influence representations using SHAP, LIME, or Spearman methods
- **Dynamic Clustering**: Perform clustering with temporal and contextual constraints
- **Transition Analysis**: Track cluster evolution over time using Markov chain models
- **Anomaly Detection**: Identify rare or unexpected transitions in consumption patterns
- **Visualization**: Visualize clusters, transitions, and temporal evolution in PDF format for publication-ready figures

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-github-username/dynamic-influence-clustering.git
   cd dynamic-influence-clustering
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the framework with default settings:

```bash
python main.py --dataset industrial_site1 --influence shap --clustering kmeans
```

### TNNLS Experiments

Run comprehensive experiments for TNNLS submission:

```bash
python run_tnnls_experiments.py --datasets energy_data steel_industry --experiments clustering_quality temporal_analysis
```

### Command Line Arguments for Main Script

- `--dataset`: Dataset to use (building_genome, industrial_site1, industrial_site2, industrial_site3, energy_data, steel_industry)
- `--influence`: Influence method (shap, lime, spearman)
- `--clustering`: Clustering algorithm (kmeans, hierarchical, spectral)
- `--n_clusters`: Number of clusters (default: 3)
- `--output_dir`: Directory to save results (default: data/results/{dataset})

### Command Line Arguments for TNNLS Experiments

- `--datasets`: Datasets to use for experiments (default: energy_data, steel_industry)
- `--experiments`: Experiments to run (default: all experiments)
- `--influence_methods`: Influence methods to evaluate (default: shap, lime, spearman)
- `--clustering_algorithms`: Clustering algorithms to evaluate (default: kmeans, hierarchical, spectral)
- `--n_clusters_list`: Number of clusters to evaluate (default: 3, 5, 7)
- `--output_dir`: Directory to save results (default: data/results/tnnls_experiments)
- `--random_seeds`: Random seeds for reproducibility (default: 42, 123, 456, 789, 101112)
- `--n_jobs`: Number of parallel jobs (default: -1, use all available cores)
- `--verbose`: Enable verbose output (default: True)

### Examples

```bash
# Run main script
python main.py --dataset building_genome --influence shap --clustering kmeans --n_clusters 5 --output_dir results/building_genome_shap_kmeans

# Run TNNLS experiments
python run_tnnls_experiments.py --datasets energy_data --experiments clustering_quality case_studies --influence_methods shap --clustering_algorithms kmeans
```

## Project Structure

```
influence/
├── data/
│   ├── raw/                  # Raw energy consumption datasets
│   ├── processed/            # Preprocessed datasets
│   └── results/              # Results from experiments
├── src/
│   ├── preprocessing/        # Data preprocessing modules
│   ├── models/               # Predictive models
│   ├── influence/            # Influence space transformation
│   ├── clustering/           # Clustering algorithms
│   ├── temporal/             # Temporal analysis
│   └── utils/                # Utility functions
├── experiments/              # Experiment scripts
│   └── tnnls/                # TNNLS-quality experiments
│       ├── ablation_studies/     # Ablation studies
│       ├── case_studies/         # Case studies
│       ├── clustering_quality/   # Clustering quality evaluation
│       ├── computational_analysis/ # Computational analysis
│       ├── contextual_coherence/ # Contextual coherence assessment
│       └── temporal_analysis/    # Temporal pattern analysis
├── notebooks/                # Jupyter notebooks for analysis
├── tests/                    # Unit tests
├── main.py                   # Main entry point
├── run_tnnls_experiments.py  # Script for TNNLS experiments
├── config.py                 # Configuration settings
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Core Components

### 1. Data Preprocessing

The framework includes utilities for loading and preprocessing energy consumption data, handling missing values, normalizing features, and aligning temporal data.

### 2. Predictive Modeling

Gradient boosting models are used to predict energy consumption based on input features. These models serve as the foundation for generating influence scores.

### 3. Influence Space Transformation

Three methods are implemented for generating influence scores:
- **SHAP**: Uses Shapley values to quantify feature contributions
- **LIME**: Uses local interpretable model-agnostic explanations
- **Spearman**: Uses Spearman rank correlation coefficients

### 4. Dynamic Clustering

The framework supports three clustering algorithms:
- **K-means**: Centroid-based clustering
- **Hierarchical**: Agglomerative clustering with various linkage criteria
- **Spectral**: Graph-based clustering for complex patterns

### 5. Transition Analysis

Markov chain models are used to analyze cluster transitions over time, providing insights into pattern evolution and stability.

### 6. Anomaly Detection

The framework can identify anomalous transitions between clusters, which may indicate unusual energy consumption patterns.

## Extending the Framework

### Adding New Datasets

1. Place your dataset in the `data/raw/` directory
2. Implement a custom data loader in `src/preprocessing/data_loader.py`
3. Add preprocessing logic in `src/preprocessing/preprocessor.py`

### Adding New Influence Methods

1. Create a new class in the `src/influence/` directory
2. Implement the `generate_influence()` method
3. Update `main.py` to include the new method

### Adding New Clustering Algorithms

1. Create a new class in the `src/clustering/` directory that extends `BaseClustering`
2. Implement the required methods: `fit()` and `predict()`
3. Update `main.py` to include the new algorithm

## Contributing and Development

### Cleaning the Project

Before pushing your code to GitHub, you can use the provided cleaning script to remove temporary files, Python cache files, and other artifacts:

```bash
./clean_project.sh
```

This script will:
- Remove Python cache files (`__pycache__`, `.pyc`, etc.)
- Remove Jupyter notebook checkpoints
- Remove IDE-specific files
- Remove log files
- Clean the results directory
- Check for large files that might need to be handled with Git LFS

### Code Style

This project follows PEP 8 style guidelines. You can use tools like `flake8` and `black` to ensure your code adheres to these standards.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite the original paper:

```
@article{dynamic_influence_clustering,
  title={Dynamic Influence-Based Clustering for Energy Consumption Analysis: A Framework for Subgroup Discovery and Transition Detection},
  author={Ma, Rongfei and Liu, Xiufeng},
  journal={Sustainable Cities and Society},
  year={2023}
}
```
