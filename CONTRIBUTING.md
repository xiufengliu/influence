# Contributing to Dynamic Influence-Based Clustering Framework

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Run the tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Development Setup

1. Clone your fork of the repository
2. Install the development dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the tests to make sure everything is working:
   ```
   pytest
   ```

## Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and methods
- Include type hints where appropriate
- Write unit tests for new functionality

## Pull Request Process

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
