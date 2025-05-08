# TNNLS-Quality Experiments for Dynamic Influence-Based Clustering Framework

This directory contains comprehensive experiments designed to meet the standards of the IEEE Transactions on Neural Networks and Learning Systems (TNNLS) journal.

## Experiment Overview

The experiments are organized into the following categories:

1. **Clustering Quality Evaluation**: Evaluates the quality of clusters produced by different influence methods and clustering algorithms.
2. **Temporal Pattern Analysis**: Analyzes the temporal evolution of clusters and transitions between them.
3. **Contextual Coherence Assessment**: Evaluates the coherence of clusters within specific contexts.
4. **Ablation Studies**: Analyzes the contribution of different components of the framework.
5. **Case Studies**: Demonstrates the practical utility of the framework in real-world scenarios.
6. **Computational Analysis**: Evaluates the efficiency and scalability of the framework.

## Running the Experiments

To run all experiments, use the main script:

```bash
python run_tnnls_experiments.py
```

### Command Line Arguments

- `--datasets`: Datasets to use for experiments (default: all available datasets)
- `--experiments`: Experiments to run (default: all experiments)
- `--influence_methods`: Influence methods to evaluate (default: shap, lime, spearman)
- `--clustering_algorithms`: Clustering algorithms to evaluate (default: kmeans, hierarchical, spectral)
- `--n_clusters_list`: Number of clusters to evaluate (default: 3, 5, 7)
- `--output_dir`: Directory to save results (default: data/results/tnnls_experiments)
- `--random_seeds`: Random seeds for reproducibility (default: 42, 123, 456, 789, 101112)
- `--n_jobs`: Number of parallel jobs (default: -1, use all available cores)
- `--verbose`: Enable verbose output (default: True)

### Examples

Run only clustering quality experiments:

```bash
python run_tnnls_experiments.py --experiments clustering_quality
```

Run experiments with specific influence methods:

```bash
python run_tnnls_experiments.py --influence_methods shap lime
```

Run experiments with specific datasets:

```bash
python run_tnnls_experiments.py --datasets building_genome industrial_site1
```

## Experiment Details

### 1. Clustering Quality Evaluation

Evaluates clustering quality using multiple metrics:
- Silhouette score (cluster separation)
- Davies-Bouldin index (cluster compactness)
- Calinski-Harabasz index (cluster definition)
- Entropy (cluster homogeneity)

Statistical significance testing is performed to compare different methods.

### 2. Temporal Pattern Analysis

Analyzes temporal patterns using:
- Transition matrices between clusters
- Stationary distribution analysis
- Cluster stability metrics
- Temporal coherence measures

### 3. Contextual Coherence Assessment

Evaluates contextual coherence using:
- Entropy analysis within specific contexts
- Conditional entropy given contextual attributes
- Information gain analysis
- Context-specific cluster distribution

### 4. Ablation Studies

Analyzes the contribution of different components:
- Impact of influence space transformation
- Effect of temporal integration
- Role of contextual alignment

### 5. Case Studies

Demonstrates practical utility through:
- Pattern discovery in energy consumption
- Transition analysis between consumption patterns
- Anomaly detection in consumption patterns

### 6. Computational Analysis

Evaluates efficiency and scalability:
- Time complexity analysis
- Memory requirements
- Scalability with increasing dataset size
- Runtime comparison of different methods

## Results

Results are saved in the specified output directory with the following structure:

```
output_dir/
├── clustering_quality/
│   ├── results/
│   └── visualizations/
├── temporal_analysis/
│   ├── results/
│   └── visualizations/
├── contextual_coherence/
│   ├── results/
│   └── visualizations/
├── ablation_studies/
│   ├── influence_space/
│   ├── temporal_integration/
│   └── contextual_alignment/
├── case_studies/
│   ├── pattern_discovery/
│   ├── transition_analysis/
│   └── anomaly_detection/
└── computational_analysis/
    ├── influence_benchmarks/
    ├── clustering_benchmarks/
    └── end_to_end_benchmarks/
```

Each directory contains CSV files with detailed results and PNG files with visualizations.
