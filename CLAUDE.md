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

## 【BRONZE MEDAL ROADMAP】5-Week Detailed Plan

### Week 1: Foundation & Baseline (Target: LB 0.50+)

#### Day 1-2: Data Understanding & EDA
**Data Structure Analysis:**
- Analyze train.csv, test.csv structure and memory usage
- Understand sensor channel meanings:
  - IMU (accelerometer/gyroscope): Body movement detection
  - ToF: Hand-to-face distance measurement  
  - Thermopile: Temperature distribution (facial contact detection)

**Statistical Analysis:**
- Verify sensor value ranges and distributions
- Analyze missing value patterns (especially ToF/temperature sensors)
- Check participant data volume consistency
- Evaluate behavior label distribution and class imbalance

**Time-Series Visualization:**
- Plot sensor waveform patterns for each behavior class
- Validate 50Hz sampling rate consistency
- Assess noise levels across sensor modalities

#### Day 3-4: Preprocessing Pipeline
**Data Cleaning:**
- Implement anomaly detection and handling strategies
- Missing value handling:
  - IMU: Linear interpolation
  - ToF/Temperature: Zero-fill or forward fill
- Validate timestamp continuity

**Normalization:**
- Z-score normalization for IMU data
- Compare participant-level vs global normalization
- Define sensor-specific normalization strategies

**Segmentation:**
- Fixed window segmentation (e.g., 2-second = 100 samples)
- 50% overlap for training data
- Label assignment strategy (majority vote or center value)

#### Day 5-6: Feature Engineering
**tsfresh Integration:**
- Install and configure tsfresh with ComprehensiveFCParameters
- Monitor computation time and memory usage
- Extract comprehensive time-series features

**Statistical Features:**
- Window-based statistics (mean, std, min/max)
- Cross-sensor correlation features
- Rate of change and jerk (acceleration derivative)

**Frequency Domain:**
- FFT-based spectral features
- Power spectral density analysis
- Dominant frequency component extraction

#### Day 7: Baseline Model Construction
**GroupKFold Implementation:**
- Participant-based group splitting to prevent leakage
- 5-fold configuration with data leak verification
- Validate class distribution across folds

**LightGBM Baseline:**
- Basic parameter configuration
- Separate Binary/Multiclass model construction
- Implement evaluation metrics (F1 score calculation)

**Initial Submission:**
- Generate predictions and create submission.csv
- Submit to Kaggle and verify score

### Week 2: Model Enhancement (Target: LB 0.57-0.60)

#### Day 8-9: 1D CNN Implementation
**Data Preparation:**
- Convert to 3D array format (samples, timesteps, channels)
- Create memory-efficient batch generators
- Implement train/validation data loaders

**InceptionTime Architecture:**
- Implement Inception modules with residual connections
- Multi-scale convolution layers
- Configure for multimodal sensor input

**Training Configuration:**
- Loss function selection (consider focal loss for imbalance)
- Learning rate scheduler implementation
- Early stopping with validation monitoring

#### Day 10-11: Data Augmentation
**Time-Series Augmentation:**
- Temporal shifting (±0.5 seconds)
- Gaussian noise injection
- Speed variation (1.1x, 0.9x scaling)

**Sensor-Specific Augmentation:**
- IMU: Rotation transformations (handle left/right hand differences)
- ToF: Distance scaling (individual differences)
- Temperature: Offset addition for thermal variations

#### Day 12-13: Hybrid Prediction Strategy
**Two-Stage Prediction:**
- Stage 1: Binary classification (behavior present/absent)
- Stage 2: Multiclass classification for detected behaviors
- Threshold optimization for stage transition

**Feature Specialization:**
- Binary-focused: Overall movement presence indicators
- Multiclass-focused: Detailed behavioral pattern features

#### Day 14: Intermediate Evaluation
**Error Analysis:**
- Detailed confusion matrix analysis
- Identify misclassification patterns
- Evaluate per-participant performance variations

**Improvement Identification:**
- Target specific behavior classes for accuracy improvement
- Feature importance analysis
- Overfitting detection and mitigation

### Week 3: Multimodal Integration (Target: LB 0.62+)

#### Day 15-16: Multi-Branch Architecture
**Sensor-Specific Branches:**
- IMU branch: Movement detection specialization
- ToF branch: Distance pattern analysis
- Temperature branch: Contact detection focus

**Fusion Strategies:**
- Compare early vs late fusion approaches
- Implement attention mechanisms for sensor weighting
- Learn optimal branch weight combinations

#### Day 17-18: Ensemble Construction
**Model Diversity:**
- tsfresh + LightGBM baseline
- Multiple 1D CNN configurations
- Additional tree-based models (XGBoost/CatBoost)

**Ensemble Methods:**
- Simple averaging across models
- CV score-based weighted averaging
- Consider stacking for meta-learning

#### Day 19-20: Final Optimization
**Hyperparameter Tuning:**
- Optuna-based automatic optimization
- Learning rate and regularization tuning
- Ensemble weight optimization

**Test Time Augmentation (TTA):**
- Time-shift TTA for robust predictions
- Multi-prediction averaging
- Confidence-based prediction weighting

#### Day 21: Final Submission Preparation
**Code Optimization:**
- Remove unnecessary code and optimize runtime
- Verify reproducibility across runs
- Document final implementation approach

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