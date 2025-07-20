# CMI - Detect Behavior with Sensor Data

A comprehensive ML pipeline for the Kaggle CMI competition focused on detecting Body-Focused Repetitive Behaviors (BFRB) from multimodal sensor data.

## 🏆 Competition Overview

- **Competition**: [CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data)
- **Task**: Multi-class + Binary classification of BFRB behaviors
- **Metric**: 0.5 × (Binary F1 + Macro F1)
- **Target**: Bronze Medal (LB 0.60+)
- **Deadline**: August 26, 2025

## 🚀 Quick Start

```bash
# 1. Initialize environment
make setup

# 2. Run complete Week 1 baseline
make week1-baseline

# 3. Or run individual steps
make eda                 # Exploratory data analysis
make data-check          # Data quality validation
make bronze              # Data cleaning
make silver              # Feature engineering
make gold                # ML-ready preparation
make train-lgb           # Train LightGBM baseline
make evaluate            # Model evaluation
```

## 📊 Data Description

### Sensor Modalities (50Hz sampling)
- **IMU**: Accelerometer (acc_x/y/z) + Gyroscope (gyro_x/y/z)
- **ToF**: Time-of-Flight distance sensors (tof_0/1/2/3)
- **Thermopile**: Temperature array sensors (thermopile_0-4)

### Target Labels
- **Binary Task**: BFRB behavior present (1) vs absent (0)
- **Multi-class Task**: Specific BFRB gesture types
- **Evaluation**: Combined F1 score requiring balance between both tasks

## 🏗️ Project Structure

```
ml-stack-cmi/
├── src/
│   ├── data/                    # Medallion architecture pipeline
│   │   ├── bronze.py           # Raw data standardization
│   │   ├── silver.py           # Feature engineering
│   │   └── gold.py             # ML-ready preparation
│   ├── models.py               # Model implementations
│   └── validation.py           # Cross-validation framework
│
├── scripts/                    # Organized executable scripts
│   ├── setup/                  # Project initialization & EDA
│   ├── data_processing/        # Bronze→Silver→Gold pipeline
│   ├── training/               # Model training scripts
│   ├── evaluation/             # Model evaluation & analysis
│   └── ensemble/               # Ensemble & submission
│
├── tests/                      # Test suite (73% coverage)
├── Makefile                    # Workflow automation
├── CLAUDE.md                   # Development guidelines
└── requirements.txt            # Dependencies
```

## 🔄 Medallion Architecture

Our data pipeline follows the Medallion Architecture pattern:

```
🗃️  Raw Sensor Data (50Hz)
     ↓
🥉  Bronze Layer: Quality checks & normalization
     ↓
🥈  Silver Layer: Feature engineering (tsfresh, FFT, domain-specific)
     ↓
🥇  Gold Layer: ML-ready features with GroupKFold splits
```

## 📈 Development Workflow

### Week 1: Foundation & Baseline (Target: LB 0.50+)
```bash
make week1-baseline  # Runs complete pipeline
```
- Data understanding & quality checks
- Bronze/Silver/Gold layer implementation
- LightGBM baseline with tsfresh features

### Week 2: Deep Learning (Target: LB 0.57-0.60)
```bash
make week2-deep-learning  # CNN training workflow
```
- 1D CNN implementation (InceptionTime-inspired)
- Multimodal sensor fusion
- Data augmentation strategies

### Week 3: Final Optimization (Target: LB 0.62+)
```bash
make week3-final  # Ensemble optimization
```
- Feature importance analysis
- Advanced ensemble methods
- Submission generation

## 🛠️ Available Commands

### Data Pipeline
```bash
make bronze              # Clean and normalize sensor data
make silver              # Engineer features (tsfresh, FFT, etc.)
make gold                # Prepare ML-ready datasets
```

### Model Training
```bash
make train-lgb           # Train LightGBM baseline
make train-cnn           # Train 1D CNN model
make ensemble            # Train ensemble model
```

### Evaluation & Analysis
```bash
make evaluate            # Comprehensive model evaluation
make feature-importance  # Analyze feature importance
make validate-cv         # Quick cross-validation
```

### Utilities
```bash
make lint                # Code quality checks
make format              # Auto-format code
make test                # Run test suite
make clean               # Clean outputs
```

## 📋 Key Implementation Details

### Critical Requirements
- **GroupKFold**: Must use participant_id for CV splits (no data leakage!)
- **Multimodal Fusion**: Effective integration of IMU + ToF + Thermopile
- **Balanced Metrics**: Monitor Binary F1 and Macro F1 separately

### Performance Targets
| Week | Method | Expected CV | Expected LB |
|------|--------|-------------|-------------|
| 1 | tsfresh + LightGBM | 0.48-0.52 | 0.47-0.51 |
| 2 | + 1D CNN | 0.55-0.58 | 0.54-0.57 |
| 3 | + Ensemble | 0.62-0.65 | 0.61-0.64 |

## 🔧 Installation

```bash
# Basic installation
pip install -r requirements.txt

# Development installation (includes linting tools)
pip install -r requirements-dev.txt
```

### Core Dependencies
- **Data**: pandas, numpy, duckdb
- **ML**: scikit-learn, lightgbm, xgboost, tensorflow
- **Feature Engineering**: tsfresh
- **Optimization**: optuna
- **Development**: pytest, black, flake8, mypy

## 📝 Development Guidelines

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines, including:
- Script management rules (14 core scripts - do not add unnecessarily)
- Medallion architecture implementation details
- Sensor-specific processing guidelines
- Competition-specific strategies

## 🎯 Success Criteria

- **Bronze Medal**: Achieve LB 0.60+ (top 60% of ~360 teams)
- **Robust CV**: GroupKFold with minimal CV-LB gap
- **Clean Pipeline**: Modular, testable, production-quality code
- **Reproducibility**: Fixed random seeds, versioned outputs

## 📚 References

- Competition Page: [CMI - Detect Behavior](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data)
- Time-series Classification: InceptionTime, ROCKET, tsfresh
- Sensor Fusion: Multi-branch CNNs, Late fusion strategies
- BFRB Background: Body-focused repetitive behaviors in clinical context

## 🤝 Contributing

This project follows strict script management guidelines. Before adding any new scripts:
1. Check if functionality can be added to existing scripts
2. Follow naming conventions (see CLAUDE.md)
3. Place in appropriate folder
4. Update documentation

---

*Built for the Kaggle CMI Competition - Target: Bronze Medal (LB 0.60+)*