# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 【PROJECT OVERVIEW】Kaggle CMI - Detect Behavior with Sensor Data
- **Competition**: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data
- **Problem**: Multi-class + Binary classification of Body-Focused Repetitive Behaviors (BFRB) from multimodal sensor data
- **Metric**: 0.5 × (Binary F1 + Macro F1) - Custom F1 composite score
- **Current Ranking**: TBD (New project)
- **Bronze Target**: ~0.60+ (estimated from ~360 teams, top 60% percentile)
- **Deadline**: August 26, 2025 (~5 weeks remaining)

## 【CRITICAL - CURRENT PROJECT STATE】
### Initial Setup Stage
- **Competition Type**: Time-series multimodal sensor classification
- **Data Scale**: ~200 participants × hundreds of sessions (~1.5 GB)
- **Sensor Channels**: IMU (acc/gyro), ToF distance (4ch), Thermopile temperature (5ch)
- **Sampling Rate**: 50 Hz
- **Bronze Medal Target**: LB ~0.60+ (estimated position 109-216/360 teams)

### Key Challenges
- **Multimodal Fusion**: IMU + ToF + Thermopile sensor integration
- **Class Imbalance**: Binary detection + Multi-class gesture classification
- **Temporal Dependencies**: Sequential behavior patterns at 50Hz
- **Participant Leakage**: Must use GroupKFold(participant_id)

## 【DATA STRUCTURE】Multimodal Sensor Stream

### Raw Data Schema
```
Sensor Data (50 Hz sampling):
├── timestamp                 # Time reference
├── acc_x, acc_y, acc_z      # Accelerometer (IMU)
├── gyro_x, gyro_y, gyro_z   # Gyroscope (IMU)
├── tof_0, tof_1, tof_2, tof_3  # Time-of-Flight distance sensors
├── thermopile_0...4         # Temperature array sensors
├── participant_id           # Subject identifier (GroupKFold key)
├── series_id               # Session identifier
└── label                   # Target: BFRB behavior class
```

### Target Labels
- **Binary Task**: BFRB present (1) vs absent (0)
- **Multi-class Task**: Specific BFRB gesture types
- **Evaluation**: Combined F1 score requiring balance between both tasks

## 【MEDALLION ARCHITECTURE】Adapted for Time-Series

### Data Pipeline for Sensor Processing
```
🗃️  Raw Sensor Data
     │
     ├── DuckDB: data/kaggle_datasets.duckdb
     │   └── 50 Hz multimodal sensor streams
     │
     ↓ [Bronze Processing]
     │
🥉  Bronze Layer (src/data/bronze.py)
     │   └── Sensor Quality Checks & Normalization
     │   └── Missing Value Handling (ToF/Thermopile)
     │   └── Participant-aware splitting
     │
     ↓ [Silver Processing]
     │  
🥈  Silver Layer (src/data/silver.py)
     │   └── Time-series Feature Engineering
     │   └── FFT/Statistical Features (tsfresh)
     │   └── Multimodal Channel Fusion
     │
     ↓ [Gold Processing]
     │
🥇  Gold Layer (src/data/gold.py)
     │   └── ML-Ready Features/Sequences
     │   └── Train/Val split with GroupKFold
     │   └── Normalization & Scaling
```

## 【DEVELOPMENT COMMANDS】

### Core Workflow Commands (Makefile)
```bash
# Setup and Installation
make install              # Install basic dependencies
make dev-install         # Install with dev tools (black, flake8, mypy)
make setup               # Create directory structure

# Code Quality
make lint                # Check code quality (black, flake8, mypy)
make format              # Format code with black
make lint-fix            # Format and show results
make test                # Run tests with pytest

# Training Pipeline
make train-fast-dev      # Fast development training
make train-full-optimized # Full optimized training  
make train-max-performance # Maximum performance training

# Validation and Prediction
make validate-cv         # Quick CV validation
make predict-basic       # Basic submission prediction
make predict-gold        # Gold submission prediction

# Maintenance
make clean               # Clean outputs, cache files
make help                # Show available commands
```

### Direct Script Execution
```bash
# Training scripts (in scripts/ directory)
python scripts/train_fast_dev.py      # Fast development training
python scripts/train_full_optimized.py # Full optimized training
python scripts/train_max_performance.py # Max performance training
python scripts/validate_quick_cv.py    # Quick CV validation

# Prediction scripts
python scripts/predict_basic_submission.py # Basic submission
python scripts/predict_gold_submission.py  # Gold submission
```

## 【PROJECT STRUCTURE】

```
src/
├── data/                    # Medallion architecture data pipeline
│   ├── bronze.py           # Raw data standardization & quality
│   ├── silver.py           # Feature engineering & domain knowledge
│   └── gold.py             # ML-ready data preparation
├── models.py               # LightGBM model implementation
├── validation.py           # Cross-validation framework
└── util/                   # Support utilities
    ├── time_tracker.py     # Performance tracking
    └── notifications.py    # Status notifications

scripts/                    # Executable training/prediction scripts
tests/                      # Comprehensive test suite (73% coverage)
```

## 【DATA MANAGEMENT】

### DuckDB Data Source
- **Database Path**: Referenced in bronze.py (`DB_PATH` variable)
- **Primary Tables**: Raw competition data stored in DuckDB format
- **Medallion Pipeline**: bronze → silver → gold layer processing
- **GroupKFold Strategy**: Participant-aware CV to prevent leakage

### Data Access Pattern
```python
# Bronze layer (entry point)
from src.data.bronze import load_data
train_df, test_df = load_data()

# Silver layer (feature engineering)  
from src.data.silver import load_silver_data
train_silver, test_silver = load_silver_data()

# Gold layer (ML-ready)
from src.data.gold import get_ml_ready_data
X_train, y_train, X_test = get_ml_ready_data()
```

## 【SENSOR PROCESSING GUIDELINES】

### Bronze Layer - Sensor Data Quality
**Core Processing:**
- IMU normalization (Z-score per channel)
- ToF missing value handling (0-fill or masking)
- Thermopile noise reduction
- Timestamp validation (50Hz consistency)
- Participant grouping for CV

### Silver Layer - Feature Engineering
**Time-Series Features:**
- Statistical features (tsfresh integration)
- FFT/frequency domain analysis
- Cross-modal sensor correlations
- Temporal pattern detection

**Domain-Specific Features:**
- Movement patterns (IMU autocorrelation)
- Proximity patterns (ToF distance tracking)
- Thermal signatures (temperature gradients)

### Gold Layer - Model Preparation
**Data Formats:**
- Tabular features for tree-based models (LightGBM/CatBoost)
- Sequence data for deep learning (CNN/RNN)
- GroupKFold cross-validation setup

## 【MODEL IMPLEMENTATION】

### Primary Model: LightGBM
- **Implementation**: `src/models.py`
- **Features**: Supports cross-validation, hyperparameter optimization
- **Integration**: Works with medallion pipeline gold layer output

### Training Configuration
- **CV Strategy**: GroupKFold by participant_id (mandatory)
- **Optimization**: Optuna integration available
- **Metrics**: Binary F1 + Macro F1 composite scoring

## 【TESTING & CODE QUALITY】

### Test Coverage: 73%
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing  
- **Test Files**: Comprehensive test suite in `tests/` directory

### Code Quality Tools
```bash
make lint                # black, flake8, mypy checking
make format              # black code formatting
```

### Configuration
- **Black**: 120 character line length
- **Flake8**: Extended ignore for E203, W503
- **MyPy**: Configured for type checking with imports ignored
- **Pytest**: Coverage reporting with HTML output

## 【IMPLEMENTATION GUIDELINES】

### Critical Success Factors
1. **GroupKFold is Mandatory**: Participant leakage will destroy generalization
2. **Multimodal Fusion**: Don't ignore ToF/Thermal - they contain unique signals
3. **Balanced Metrics**: Monitor Binary F1 and Macro F1 separately
4. **Window Engineering**: Behavior duration varies - try multiple window sizes

### Time-Series Specific Considerations
- **Window Size**: Start with 2-5 second windows (100-250 samples)
- **Overlap**: 50-80% overlap for training, less for inference
- **Normalization**: Per-participant or per-session normalization
- **Augmentation**: Time-based augmentations preserve sensor relationships

### Common Pitfalls to Avoid
- Using standard KFold instead of GroupKFold
- Ignoring sensor-specific preprocessing needs
- Over-focusing on one modality (e.g., IMU only)
- Not handling missing ToF/Thermal values properly

## 【BRONZE MEDAL ROADMAP】5-Week Plan

### Week 1: Foundation & Baseline (Target: LB 0.50+)
1. **EDA & Sensor Understanding**
   - Channel-wise statistics and distributions
   - Label frequency analysis
   - Participant/session data quality

2. **Bronze Layer Implementation**
   - Sensor normalization pipeline
   - Missing value strategies
   - GroupKFold CV setup

3. **Quick Baseline**
   - tsfresh features + LightGBM
   - Simple sliding window approach

### Week 2: Deep Learning Integration (Target: LB 0.57-0.60)
1. **1D CNN Implementation**
   - InceptionTime or similar architecture
   - Single-modal (IMU only) baseline
   
2. **Data Augmentation**
   - Time shifting
   - Rotation augmentation (IMU)
   - Noise injection

3. **GPU Optimization**
   - Batch processing
   - Mixed precision training

### Week 3: Multimodal Fusion (Target: LB 0.62+)
1. **Multi-Branch Architecture**
   - Separate branches for IMU/ToF/Thermal
   - Feature-level fusion
   
2. **Advanced Features**
   - Cross-modal correlations
   - Attention mechanisms
   
3. **Ensemble Strategy**
   - 5-fold × multi-seed
   - Model diversity (CNN + Tree-based)

### Week 4-5: Optimization & Robustness
1. **Feature Selection**
   - Channel importance analysis
   - Inference speed optimization
   
2. **Metric Optimization**
   - Binary vs Multi-class weighting
   - Threshold tuning
   
3. **Private LB Preparation**
   - CV-LB alignment
   - Robust validation strategies

## 【SUCCESS CRITERIA】
- **Bronze Medal**: Achieve LB 0.60+ (top 60% of ~360 teams)
- **Robust CV**: GroupKFold with minimal CV-LB gap
- **Multimodal Integration**: Effective fusion of all sensor types
- **Scalable Pipeline**: Clean, modular code for rapid experimentation
- **Submission**: Valid predictions for all test samples

## 【REFERENCES & RESOURCES】
- Competition Page: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data
- Time-series Classification: InceptionTime, ROCKET, tsfresh
- Sensor Fusion: Multi-branch CNNs, Late fusion strategies
- BFRB Background: Body-focused repetitive behaviors in clinical context