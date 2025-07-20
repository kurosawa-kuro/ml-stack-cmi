# Minimal Makefile for S5E7 Personality Prediction
.PHONY: install test clean help train-fast-dev train-full-optimized train-max-performance predict-basic predict-gold validate-cv

# Core installation
install:
	pip install -e .

dev-install:
	pip install -e .[dev,optimization,visualization]

# Setup directories
setup:
	mkdir -p data/raw data/processed outputs submissions logs
	touch data/raw/.gitkeep data/processed/.gitkeep outputs/.gitkeep submissions/.gitkeep logs/.gitkeep

# Quick CV validation
validate-cv:
	@echo "Running quick CV validation..."
	PYTHONPATH=. python3 scripts/validate_quick_cv.py

# Training commands

train-fast-dev:
	@echo "Running fast development training..."
	PYTHONPATH=. python3 scripts/train_fast_dev.py

train-full-optimized:
	@echo "Running full optimized training..."
	PYTHONPATH=. python3 scripts/train_full_optimized.py

train-max-performance:
	@echo "Running maximum performance training..."
	PYTHONPATH=. python3 scripts/train_max_performance.py


# Prediction commands

# Basic submission prediction
predict-basic:
	@echo "Running basic submission prediction..."
	PYTHONPATH=. python3 scripts/predict_basic_submission.py

# Gold submission prediction
predict-gold:
	@echo "Running gold submission prediction..."
	PYTHONPATH=. python3 scripts/predict_gold_submission.py

# Code quality - unified with pre-commit hooks
lint:
	black --check src/ tests/ scripts/
	flake8 src/ tests/ scripts/
	mypy src/ tests/ scripts/

# Code formatting - apply black to all files
format:
	black src/ tests/ scripts/

# Auto-fix lint issues
lint-fix: format
	@echo "Code formatted with black"
	@echo "Note: Some lint issues may require manual fixes"

# Basic test
test:
	PYTHONPATH=. pytest tests/ -v


# Clean outputs
clean:
	rm -rf outputs/* submissions/* logs/*
	rm -rf **__pycache__** .pytest_cache .coverage htmlcov
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Help
help:
	@echo "S5E7 Personality Prediction - Minimal Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install              - Install dependencies"
	@echo "  make setup               - Create directories"
	@echo ""
	@echo "Run:"
	@echo "  make validate-cv         - Quick CV validation"
	@echo ""
	@echo "Training:"
	@echo "  make train-fast-dev      - Fast development training"
	@echo "  make train-full-optimized - Full optimized training"
	@echo "  make train-max-performance - Maximum performance training"
	@echo ""
	@echo "Prediction:"
	@echo "  make predict-basic       - Basic submission prediction"
	@echo "  make predict-gold        - Gold submission prediction"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint                - Check code quality (black, flake8, mypy)"
	@echo "  make format              - Format code with black"
	@echo "  make lint-fix            - Format and show results"
	@echo ""
	@echo "Maintenance:"
	@echo "  make test                - Run tests"
	@echo "  make clean               - Clean outputs"

# Default
.DEFAULT_GOAL := help
