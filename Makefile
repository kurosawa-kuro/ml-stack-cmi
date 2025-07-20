# CMI Competition - Kaggle Detect Behavior with Sensor Data
.PHONY: install test clean help setup eda data-check bronze silver gold train-lgb train-cnn evaluate feature-importance ensemble submit

# Core installation
install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements-dev.txt

# Project setup and initialization
setup:
	@echo "🚀 Initializing project environment..."
	mkdir -p data/raw data/processed/bronze data/processed/silver data/processed/gold
	mkdir -p outputs/models outputs/figures outputs/reports outputs/evaluation outputs/submissions
	mkdir -p logs cache
	PYTHONPATH=. python3 scripts/setup/project_setup.py

# Configuration management
config-show:
	@echo "📋 Showing current project configuration..."
	PYTHONPATH=. python3 -c "from src.config import get_project_config; config = get_project_config(); print(f'Phase: {config.phase.value}'); print(f'Primary Algorithm: {config.algorithm_strategy.primary_algorithm}'); print(f'Target CV Score: {config.targets.cv_score}'); print(f'Enabled Algorithms: {config.algorithm_strategy.enabled_algorithms}')"

config-validate:
	@echo "🔍 Validating project configuration..."
	PYTHONPATH=. python3 -c "from src.config import get_project_config; config = get_project_config(); warnings = config.validate_configuration(); print('Configuration validation complete'); [print(f'⚠️ {w}') for w in warnings] if warnings else print('✅ No configuration warnings')"

# Data understanding and quality
eda:
	@echo "📊 Running exploratory data analysis..."
	PYTHONPATH=. python3 scripts/setup/exploratory_analysis.py

data-check:
	@echo "🔍 Checking data quality..."
	PYTHONPATH=. python3 scripts/setup/data_quality_check.py

# Data processing pipeline (Configuration-Driven)
bronze:
	@echo "🥉 Processing Bronze layer (Configuration-Driven)..."
	PYTHONPATH=. python3 scripts/data_processing/bronze_layer_config.py

silver:
	@echo "🥈 Processing Silver layer (Configuration-Driven)..."
	PYTHONPATH=. python3 scripts/data_processing/silver_layer_config.py

gold:
	@echo "🥇 Processing Gold layer (Configuration-Driven)..."
	PYTHONPATH=. python3 scripts/data_processing/gold_layer_config.py

# Legacy data processing (fallback)
bronze-legacy:
	@echo "🥉 Processing Bronze layer (Legacy)..."
	PYTHONPATH=. python3 scripts/data_processing/bronze_layer.py

silver-legacy:
	@echo "🥈 Processing Silver layer (Legacy)..."
	PYTHONPATH=. python3 scripts/data_processing/silver_layer.py

gold-legacy:
	@echo "🥇 Processing Gold layer (Legacy)..."
	PYTHONPATH=. python3 scripts/data_processing/gold_layer.py

# Model training
train-lgb:
	@echo "🌳 Training LightGBM baseline..."
	PYTHONPATH=. python3 scripts/training/train_lightgbm.py

# Configuration-driven training
train-lgb-config:
	@echo "🌳 Training LightGBM baseline (Configuration-Driven)..."
	PYTHONPATH=. python3 scripts/training/train_lightgbm_config.py

train-cnn:
	@echo "🧠 Training 1D CNN..."
	PYTHONPATH=. python3 scripts/training/train_cnn.py

# Evaluation and analysis
evaluate:
	@echo "📈 Evaluating model performance..."
	PYTHONPATH=. python3 scripts/evaluation/model_evaluation.py

feature-importance:
	@echo "🌟 Analyzing feature importance..."
	PYTHONPATH=. python3 scripts/evaluation/feature_analysis.py

validate-cv:
	@echo "✓ Running quick CV validation..."
	PYTHONPATH=. python3 scripts/evaluation/validate_quick_cv.py

# Ensemble and submission
ensemble:
	@echo "🎭 Training ensemble model..."
	PYTHONPATH=. python3 scripts/ensemble/train_ensemble.py

submit:
	@echo "📝 Generating submission file..."
	PYTHONPATH=. python3 scripts/ensemble/generate_submission.py

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

# Workflow shortcuts (common sequences)
week1-baseline:
	@echo "🎯 Running Week 1 baseline workflow (Configuration-Driven)..."
	@echo "📋 Phase: baseline | Algorithm: lightgbm | Target: CV 0.50+"
	$(MAKE) setup
	$(MAKE) config-show
	$(MAKE) eda
	$(MAKE) data-check
	$(MAKE) bronze
	$(MAKE) silver
	$(MAKE) gold
	$(MAKE) train-lgb-config
	$(MAKE) evaluate

week1-baseline-legacy:
	@echo "🎯 Running Week 1 baseline workflow (Legacy)..."
	$(MAKE) setup
	$(MAKE) eda
	$(MAKE) data-check
	$(MAKE) bronze-legacy
	$(MAKE) silver-legacy
	$(MAKE) gold-legacy
	$(MAKE) train-lgb
	$(MAKE) evaluate

week2-deep-learning:
	@echo "🧠 Running Week 2 deep learning workflow..."
	$(MAKE) train-cnn
	$(MAKE) evaluate

week3-final:
	@echo "🏁 Running Week 3 final optimization..."
	$(MAKE) feature-importance
	$(MAKE) ensemble
	$(MAKE) submit

# Help
help:
	@echo "CMI Competition - Kaggle Detect Behavior Commands"
	@echo ""
	@echo "🚀 Setup & Initialization:"
	@echo "  make install              - Install dependencies"
	@echo "  make setup               - Initialize project environment"
	@echo "  make config-show          - Show current configuration"
	@echo "  make config-validate      - Validate configuration"
	@echo ""
	@echo "📊 Data Understanding:"
	@echo "  make eda                 - Exploratory data analysis"
	@echo "  make data-check          - Data quality validation"
	@echo ""
	@echo "⚙️  Data Processing (Medallion - Configuration-Driven):"
	@echo "  make bronze              - Clean and normalize data (Config-Driven)"
	@echo "  make silver              - Feature engineering (Config-Driven)"
	@echo "  make gold                - ML-ready preparation (Config-Driven)"
	@echo "  make bronze-legacy       - Legacy bronze processing"
	@echo "  make silver-legacy       - Legacy silver processing"
	@echo "  make gold-legacy         - Legacy gold processing"
	@echo ""
	@echo "🤖 Model Training:"
	@echo "  make train-lgb           - Train LightGBM baseline"
	@echo "  make train-lgb-config    - Train LightGBM (Configuration-Driven)"
	@echo "  make train-cnn           - Train 1D CNN model"
	@echo ""
	@echo "📈 Evaluation & Analysis:"
	@echo "  make evaluate            - Model evaluation"
	@echo "  make feature-importance  - Feature analysis"
	@echo "  make validate-cv         - Quick CV validation"
	@echo ""
	@echo "🎯 Final Steps:"
	@echo "  make ensemble            - Train ensemble model"
	@echo "  make submit              - Generate submission"
	@echo ""
	@echo "🔄 Week-based Workflows:"
	@echo "  make week1-baseline      - Complete Week 1 pipeline (Config-Driven)"
	@echo "  make week1-baseline-legacy - Week 1 pipeline (Legacy)"
	@echo "  make week2-deep-learning - Week 2 CNN training"
	@echo "  make week3-final         - Week 3 optimization"
	@echo ""
	@echo "🔧 Code Quality:"
	@echo "  make lint                - Check code quality"
	@echo "  make format              - Format code with black"
	@echo "  make test                - Run tests"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  make clean               - Clean all outputs"

# Default
.DEFAULT_GOAL := help
