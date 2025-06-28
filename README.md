# Dynamic Influence-Based Clustering Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python framework for dynamic clustering with influence analysis, designed for interpretable analysis of complex time-series data. This framework combines advanced clustering algorithms with influence function analysis to provide insights into how data points affect clustering decisions over time.

## 🚀 Features

- **Dynamic Clustering**: Adaptive clustering algorithms that handle evolving data patterns
- **Multiple Influence Methods**: SHAP, LIME, Integrated Gradients, Hessian-based, and Spearman influence analysis
- **Temporal Analysis**: Built-in tools for analyzing clustering evolution and transitions over time
- **Comprehensive Evaluation**: Multiple metrics for clustering quality, contextual coherence, and temporal stability
- **Flexible Architecture**: Modular design supporting custom clustering algorithms and influence methods
- **Scalable Implementation**: Efficient algorithms with parallel processing support

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd influence

# Install dependencies
pip install -r requirements.txt

# Verify installation
python setup_verification.py
```

### Optional Dependencies

For enhanced time-series analysis:
```bash
pip install tslearn>=0.5.2
```

## 🏃 Quick Start

### Basic Usage

```python
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.models.gradient_boost import GradientBoostModel
from src.influence.shap_influence import ShapInfluence
from src.clustering.dynamic_kmeans import DynamicKMeansClustering

# Load and preprocess data
loader = DataLoader("your_dataset")
data = loader.load_data()

preprocessor = Preprocessor()
X, y, timestamps, contexts = preprocessor.process_data(data)

# Train predictive model
model = GradientBoostModel()
model.fit(X, y)

# Generate influence representations
influence_extractor = ShapInfluence(model)
influence_vectors = influence_extractor.compute_influence(X)

# Perform dynamic clustering
clustering = DynamicKMeansClustering(
    n_clusters=3,
    alpha=1.0,  # cohesion weight
    beta=1.0,   # temporal weight
    gamma=1.0   # contextual weight
)
cluster_labels = clustering.fit_predict(influence_vectors, timestamps, contexts)
```

### Quick Demo

Try the framework with example data:

```bash
# Run complete demo with synthetic data
python demo.py

# Run experiment with your data
python run_experiments.py --dataset your_data.csv --influence shap

# Compare multiple influence methods
python run_experiments.py --dataset your_data.csv --compare_all
```

## 🔧 Configuration

### Command Line Interface

```bash
python main.py --dataset energy_data --influence shap --clustering dynamic_kmeans --n_clusters 5
```

### Available Options

- `--dataset`: Dataset to analyze
- `--influence`: Influence method (shap, lime, integrated_gradients, hessian, spearman)
- `--clustering`: Clustering algorithm (dynamic_kmeans, hierarchical, spectral)
- `--n_clusters`: Number of clusters
- `--output_dir`: Results directory

## 🏗️ Framework Components

### 1. Clustering Algorithms

- **Dynamic K-Means**: Adaptive K-means with temporal and contextual constraints
- **Hierarchical Clustering**: Agglomerative clustering with various linkage criteria
- **Spectral Clustering**: Graph-based clustering for complex pattern detection
- **Time-Series Baselines**: Specialized algorithms for temporal data (k-Shape, DTW-based)

### 2. Influence Methods

- **SHAP**: Shapley value-based feature importance analysis
- **LIME**: Local interpretable model-agnostic explanations
- **Integrated Gradients**: Path-integrated attribution method
- **Hessian Influence**: Second-order gradient-based influence functions
- **Spearman Influence**: Rank correlation-based influence analysis

### 3. Evaluation Framework

- **Clustering Quality**: Silhouette score, Davies-Bouldin index, Calinski-Harabasz index
- **Contextual Coherence**: Domain-specific alignment measures
- **Temporal Analysis**: Transition matrices, stability metrics, anomaly detection

## 📁 Project Structure

```
influence/
├── src/                      # Core framework code
│   ├── clustering/           # Clustering algorithms
│   │   ├── dynamic_kmeans.py
│   │   ├── hierarchical.py
│   │   ├── spectral.py
│   │   └── timeseries_baselines.py
│   ├── influence/            # Influence function implementations
│   │   ├── shap_influence.py
│   │   ├── lime_influence.py
│   │   ├── integrated_gradients_influence.py
│   │   ├── hessian_influence.py
│   │   └── spearman_influence.py
│   ├── models/               # Machine learning models
│   ├── preprocessing/        # Data preprocessing utilities
│   ├── temporal/             # Temporal analysis tools
│   └── utils/                # Utilities and helpers
├── examples/                 # Example scripts and advanced usage
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks for exploration
├── config.py                 # Configuration settings
├── main.py                   # Main execution script
├── run_experiments.py        # Simple experiment runner
└── demo.py                   # Quick demonstration script
```

## 💾 Data Format Requirements

The framework expects CSV data with the following structure:

```csv
timestamp,feature1,feature2,feature3,target
2023-01-01,1.2,3.4,5.6,100.5
2023-01-02,1.3,3.2,5.8,102.1
2023-01-03,1.1,3.6,5.4,98.9
...
```

Required columns:
- **timestamp**: Time index (datetime or sequence number)
- **features**: Numerical feature columns
- **target**: Target variable for predictive modeling (optional)

## 🔬 Advanced Usage

### Custom Clustering Algorithms

Extend the base clustering class:

```python
from src.clustering.base_clustering import BaseClustering

class CustomClustering(BaseClustering):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit_predict(self, X, timestamps=None, contexts=None):
        # Implement your clustering logic
        return cluster_labels
```

### Custom Influence Methods

Implement custom influence functions:

```python
from src.influence.base_influence import BaseInfluence

class CustomInfluence(BaseInfluence):
    def compute_influence(self, X, model=None):
        # Implement your influence computation
        return influence_scores
```

### Adding New Datasets

1. Place dataset in `data/raw/` directory
2. Implement data loader in `src/preprocessing/data_loader.py`
3. Add preprocessing logic if needed

## ⚡ Performance Optimization

### Memory Management
- Use batch processing for large datasets
- Enable parallel processing with `n_jobs` parameter
- Monitor memory usage with influence methods

### Computational Efficiency
- Start with Spearman influence method (fastest)
- Use GPU acceleration where available
- Consider data sampling for initial exploration

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_framework.py

# Check installation
python setup_verification.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Testing requirements
- Pull request process
- Development setup

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Troubleshooting

### Common Issues

1. **Memory Issues**: Use smaller batch sizes or sequential processing
2. **Import Errors**: Verify all dependencies are installed
3. **Performance**: Start with simpler influence methods

### Getting Help

- Check the `examples/` directory for usage patterns
- Review `notebooks/` for detailed analysis examples
- Open an issue for bugs or feature requests

## 📈 Roadmap

- [ ] Additional influence methods
- [ ] Enhanced visualization tools
- [ ] Real-time clustering support
- [ ] Cloud deployment options
- [ ] API interface development

---

**Built with ❤️ for interpretable machine learning and dynamic data analysis**