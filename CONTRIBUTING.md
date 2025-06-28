# Contributing to Dynamic Influence-Based Clustering

Thank you for your interest in contributing to the Dynamic Influence-Based Clustering framework! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Pull Request Process](#pull-request-process)

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We strive to maintain a welcoming environment for all contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your contribution
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Installation

```bash
# Clone your fork
git clone https://github.com/your-username/dynamic-influence-clustering.git
cd dynamic-influence-clustering

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8

# Verify installation
python setup_verification.py
## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Report and fix bugs
- **Feature enhancements**: Improve existing functionality
- **New features**: Add new influence methods, clustering algorithms, or evaluation metrics
- **Documentation**: Improve documentation, examples, or tutorials
- **Performance improvements**: Optimize code for better performance
- **Testing**: Add or improve test coverage

### Reporting Issues

Before creating an issue:

1. Check if the issue already exists
2. Use the latest version of the code
3. Include detailed reproduction steps
4. Provide system information and dependencies

## Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and methods
- Include type hints where appropriate
- Use meaningful variable and function names
- Write unit tests for new functionality

### Code Formatting

Use [Black](https://black.readthedocs.io/) for code formatting:

```bash
# Format all Python files
black src/ tests/ *.py

# Check formatting without making changes
black --check src/ tests/ *.py
```

### Linting

Use [flake8](https://flake8.pycqa.org/) for linting:

```bash
# Run linting
flake8 src/ tests/ *.py
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_clustering.py

# Run with coverage
python -m pytest tests/ --cov=src/

# Run with verbose output
python -m pytest tests/ -v
```

### Writing Tests

- Write tests for all new functionality
- Aim for high test coverage (>80%)
- Use descriptive test names
- Include edge cases and error conditions

## Pull Request Process

1. **Create a feature branch**: Use descriptive branch names (e.g., `feature/new-influence-method`)
2. **Make focused changes**: Keep pull requests reasonably sized
3. **Test thoroughly**: Ensure all tests pass
4. **Format code**: Run Black and flake8
5. **Update documentation**: Add/update docstrings and README
6. **Submit PR**: Provide a clear description of changes

### Pull Request Guidelines

- Reference related issues in the description
- Include tests for new functionality
- Update documentation when needed
- Ensure backward compatibility or document breaking changes

## Specific Contribution Areas

### Adding New Influence Methods

To add a new influence method:

1. Create a new class inheriting from `BaseInfluence`
2. Implement the `compute_influence` method
3. Add comprehensive tests
4. Update configuration and documentation

### Adding New Clustering Algorithms

To add a new clustering algorithm:

1. Create a new class inheriting from `BaseClustering`
2. Implement required methods (`fit`, `predict`, etc.)
3. Add comprehensive tests
4. Update experimental configurations

### Adding New Evaluation Metrics

To add new evaluation metrics:

1. Add methods to `ClusteringEvaluator` class
2. Include appropriate references for the metrics
3. Add comprehensive tests
4. Update documentation

## Questions and Support

If you have questions about contributing:

- Open an issue for bugs or feature requests
- Contact the maintainers for general questions
- Check existing issues and pull requests

Thank you for contributing to Dynamic Influence-Based Clustering!

1. Update the README.md with details of changes if appropriate
2. Update the requirements.txt file if you've added new dependencies
3. Run the clean_project.sh script before committing:
   ```
   ./clean_project.sh
   ```
4. Your pull request will be reviewed by the maintainers

## Adding New Features

### Adding New Datasets

1. Place your dataset in the `data/raw/` directory
2. Implement a custom data loader in `src/preprocessing/data_loader.py`
3. Add preprocessing logic in `src/preprocessing/preprocessor.py`
4. Add tests for your new dataset

### Adding New Influence Methods

1. Create a new class in the `src/influence/` directory
2. Implement the `generate_influence()` method
3. Update `main.py` to include the new method
4. Add tests for your new influence method

### Adding New Clustering Algorithms

1. Create a new class in the `src/clustering/` directory that extends `BaseClustering`
2. Implement the required methods: `fit()` and `predict()`
3. Update `main.py` to include the new algorithm
4. Add tests for your new clustering algorithm

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
