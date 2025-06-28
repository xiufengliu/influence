# Makefile for Dynamic Influence-Based Clustering

.PHONY: help install test lint format demo verify clean

help: ## Show this help message
	@echo "Dynamic Influence-Based Clustering - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest black flake8 mypy

verify: ## Verify installation and run basic tests
	python setup_verification.py

demo: ## Run the demo with synthetic data
	python demo.py

test: ## Run all tests
	python -m pytest tests/ -v

test-coverage: ## Run tests with coverage report
	python -m pytest tests/ --cov=src/ --cov-report=html

lint: ## Run linting checks
	flake8 src/ tests/ *.py

format: ## Format code with black
	black src/ tests/ *.py

format-check: ## Check code formatting without making changes
	black --check src/ tests/ *.py

type-check: ## Run type checking with mypy
	mypy src/

experiment-simple: ## Run simple experiment with spearman influence
	python run_experiments.py --dataset energy_data --influence spearman

experiment-compare: ## Compare all influence methods
	python run_experiments.py --dataset energy_data --compare_all

experiment-paper: ## Run comprehensive paper experiments (advanced)
	python examples/examples_paper_experiments.py --dataset energy_data --output_dir results/

clean: ## Clean temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage

clean-results: ## Clean experiment results
	rm -rf results/
	rm -rf data/results/

setup: install verify ## Complete setup: install dependencies and verify

check: format-check lint test ## Run all checks: formatting, linting, and tests

all: clean install verify demo ## Clean, install, verify, and run demo

# Development workflow
dev-setup: install-dev verify ## Setup development environment

dev-check: format lint type-check test ## Run all development checks

# CI/CD targets
ci-test: install test ## CI testing target

ci-check: install format-check lint test ## CI comprehensive check

# Documentation
docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

# Docker targets (future enhancement)
docker-build: ## Build Docker image (placeholder)
	@echo "Docker support not yet implemented"

docker-run: ## Run in Docker container (placeholder)
	@echo "Docker support not yet implemented"

# Information targets
info: ## Show project information
	@echo "Dynamic Influence-Based Clustering Framework"
	@echo "============================================="
	@echo "Python version: $$(python --version)"
	@echo "Project root: $$(pwd)"
	@echo "Dependencies: $$(wc -l < requirements.txt) packages"
	@echo ""
	@echo "Core modules:"
	@find src/ -name "*.py" -not -path "*/test_*" -not -name "__*" | wc -l | xargs echo "  Python files:"
	@find src/ -type d | wc -l | xargs echo "  Directories:"

deps: ## Show dependency information
	@echo "Required dependencies:"
	@cat requirements.txt
	@echo ""
	@echo "Development dependencies:"
	@echo "pytest>=6.2.0"
	@echo "black>=21.0.0"
	@echo "flake8>=3.9.0"
	@echo "mypy>=0.812"
