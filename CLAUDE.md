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
### Configuration-Driven Development
**Strategy Control**: Project phase and algorithms are controlled via configuration files and environment variables:
- **Config File**: `config/project_config.yaml` - Main project settings
- **Environment**: `.env` - Environment-specific variables
- **Dynamic Loading**: `src/config/strategy.py` - Phase-aware strategy system

**Current Phase**: `${PROJECT_PHASE}` (baseline/optimization/ensemble)

### BASELINE PHASE - Single Algorithm Focus
**When PROJECT_PHASE=baseline**: This project implements a **configuration-driven single algorithm approach** for bronze medal achievement.

**Configuration-Controlled Strategy**:
- **Primary Algorithm**: `${PRIMARY_ALGORITHM}` (from config)
- **Ensemble Mode**: `${ENSEMBLE_ENABLED}` (disabled in baseline)
- **Target Score**: `${TARGET_CV_SCORE}` (phase-specific)
- **Algorithm List**: `${ENABLED_ALGORITHMS}` (single in baseline)

**Benefits of Configuration-Driven Approach**:
- **Flexible Phase Management**: Switch strategies via config change only
- **Environment Separation**: Different settings for dev/prod
- **Team Coordination**: Shared strategy without code conflicts
- **CI/CD Ready**: Automated pipeline configuration

### Competition Overview
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

### Raw Data Schema (EDA Verified)
```
Sensor Data (50 Hz sampling):
├── timestamp                 # Time reference
├── acc_x, acc_y, acc_z      # Accelerometer (IMU) - 0% missing
├── rot_w, rot_x, rot_y, rot_z  # Rotation quaternions - 0.64% missing
├── thm_1, thm_2, thm_3, thm_4  # Thermopile sensors - 1-2% missing  
├── thm_5                    # Thermopile sensor 5 - 5.79% missing (critical)
├── tof_1..4_v0..63         # ToF distance arrays (64ch each) - 1% missing
├── tof_5_v0..63            # ToF distance array 5 - 5.24% missing (critical)
├── participant_id          # Subject identifier (81 participants, GroupKFold key)
├── series_id              # Session identifier (8,151 sequences)
└── label                  # Target: BFRB behavior class (18 gestures)
```

### EDA Findings - Data Quality
**Reference**: [EDA Results](docs/project-info/EDA結果.md)
- **Train Data**: 574,945 rows, 81 participants, 8,151 sequences
- **Test Data**: 107 rows, 2 participants, 2 sequences  
- **Average Sequence Duration**: 1.41 seconds (70.5 samples at 50Hz)
- **Critical Missing Sensors**: thm_5 (5.79%) and tof_5 (5.24%) - 94% co-occurrence
- **Problematic Participants**: SUBJ_044680 and SUBJ_016552 with 100% missing for sensors 5

### Target Labels Distribution
- **Binary Task**: Target 59.84% vs Non-Target 40.16%
- **Multi-class Task**: 18 gesture classes with 5.9:1 imbalance ratio
- **Most Frequent**: "Text on phone" (10.17%)
- **Least Frequent**: "Pinch knee/leg skin" (1.71%)
- **Evaluation**: 0.5 × (Binary F1 + Macro F1) composite score

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
make install              # Install dependencies
make setup               # Initialize project environment

# Data Understanding & Quality
make eda                 # Exploratory data analysis
make data-check          # Data quality validation

# Data Processing Pipeline
make bronze              # Bronze layer: Clean and normalize
make silver              # Silver layer: Feature engineering
make gold                # Gold layer: ML-ready preparation

# Model Training
make train-lgb           # Train LightGBM baseline
make train-cnn           # Train 1D CNN model

# Evaluation & Analysis
make evaluate            # Comprehensive model evaluation
make feature-importance  # Feature importance analysis
make validate-cv         # Quick CV validation

# Final Steps
make ensemble            # Train ensemble model
make submit              # Generate submission file

# Week-based Workflows (Shortcuts)
make week1-baseline      # Complete Week 1 pipeline
make week2-deep-learning # Week 2 CNN training
make week3-final         # Week 3 optimization

# Code Quality & Maintenance
make lint                # Check code quality
make format              # Format code with black
make test                # Run tests with pytest
make clean               # Clean all outputs
make help                # Show all commands
```

### Direct Script Execution
```bash
# Setup and initialization
python scripts/setup/project_setup.py         # Project initialization
python scripts/setup/exploratory_analysis.py  # Comprehensive EDA  
python scripts/setup/data_quality_check.py    # Data quality validation

# Data processing pipeline (Medallion Architecture)
python scripts/data_processing/bronze_layer.py  # Bronze layer processing
python scripts/data_processing/silver_layer.py  # Silver layer features
python scripts/data_processing/gold_layer.py    # Gold layer ML-ready

# Model training
python scripts/training/train_lightgbm.py     # LightGBM baseline
python scripts/training/train_cnn.py          # 1D CNN deep learning

# Evaluation and analysis
python scripts/evaluation/model_evaluation.py  # Model evaluation
python scripts/evaluation/feature_analysis.py # Feature analysis

# Ensemble and submission
python scripts/ensemble/train_ensemble.py      # Ensemble training
python scripts/ensemble/generate_submission.py # Generate submission
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

scripts/                    # Organized executable scripts (see SCRIPT MANAGEMENT)
├── setup/                  # Project initialization & data understanding
├── data_processing/        # Medallion architecture pipeline
├── training/              # Model training scripts
├── evaluation/            # Model evaluation & analysis
└── ensemble/              # Ensemble & submission generation

tests/                      # Comprehensive test suite (73% coverage)
```

## 【OUTPUT FOLDER MANAGEMENT】

### Standardized Output Directory Structure
All project outputs (reports, submissions, models, etc.) should be organized in hierarchical folders at the project root level. This ensures consistent output management and easy tracking of results.

### Output Directory Structure
```
outputs/                    # All outputs at project root (git-ignored)
├── submissions/           # Kaggle submission files
│   ├── baseline/         # Initial submissions
│   ├── experiments/      # Experimental submissions
│   └── final/            # Final competition submissions
├── reports/              # Analysis and evaluation reports
│   ├── eda/             # EDA visualizations and reports
│   ├── model_evaluation/ # Model performance reports
│   └── feature_analysis/ # Feature importance reports
├── models/              # Trained model artifacts
│   ├── lightgbm/        # LightGBM models
│   ├── deep_learning/   # CNN/RNN models
│   └── ensemble/        # Ensemble models
├── figures/             # Plots and visualizations
│   ├── training/        # Training curves
│   ├── validation/      # CV results
│   └── analysis/        # Analysis plots
└── logs/                # Training and execution logs
    ├── training/        # Model training logs
    ├── evaluation/      # Evaluation logs
    └── errors/          # Error logs for debugging
```

### Output Management Guidelines
1. **Fixed Root Location**: All outputs MUST be saved under `outputs/` at project root
2. **Hierarchical Organization**: Use subdirectories to categorize outputs by type and purpose
3. **Timestamping**: Include timestamps in filenames for versioning (e.g., `submission_20240115_143022.csv`)
4. **Descriptive Naming**: Use clear, descriptive names that indicate content and parameters
5. **Git Ignore**: The entire `outputs/` directory should be in `.gitignore` to avoid version control bloat

### Implementation Examples
```python
# Correct: Save to standardized output location
submission_path = "outputs/submissions/baseline/submission_lgb_cv0.612_20240115.csv"
report_path = "outputs/reports/eda/sensor_analysis_20240115.html"
model_path = "outputs/models/lightgbm/lgb_fold3_score0.615.pkl"

# Incorrect: Scattered outputs
# ❌ "submission.csv"  # Root directory
# ❌ "scripts/outputs/report.html"  # Script subdirectory
# ❌ "../results/model.pkl"  # Outside project
```

### Benefits of Centralized Output Management
- **Easy Cleanup**: Single `make clean` command can clear all outputs
- **Better Organization**: Clear separation between code and outputs
- **Experiment Tracking**: Historical outputs preserved with timestamps
- **Collaboration**: Team members know exactly where to find results
- **Backup/Archive**: Easy to backup or archive the entire outputs folder

## 【SCRIPT MANAGEMENT GUIDELINES】

### Core Script Set (14 Scripts) - DO NOT ADD UNNECESSARILY
The current script set is **complete and sufficient** for Bronze Medal achievement (LB 0.60+).

#### Established Scripts by Category:
```
setup/
├── project_setup.py         # Environment initialization
├── exploratory_analysis.py  # Comprehensive EDA
└── data_quality_check.py    # Data validation

data_processing/
├── bronze_layer.py         # Data cleaning & normalization
├── silver_layer.py         # Feature engineering
└── gold_layer.py           # ML-ready preparation

training/
├── train_lightgbm.py       # LightGBM baseline
├── train_cnn.py            # 1D CNN deep learning
└── train_baseline.py       # Generic training

evaluation/
├── model_evaluation.py     # Comprehensive evaluation
├── feature_analysis.py     # Feature importance
└── validate_quick_cv.py    # Quick validation

ensemble/
├── train_ensemble.py       # Ensemble training
└── generate_submission.py  # Submission generation
```

### Script Addition Rules
1. **AVOID creating new scripts** - The current set covers the complete ML pipeline
2. **If absolutely necessary**, follow these strict guidelines:
   - **Derive from existing names**: e.g., `train_lightgbm_v2.py`, `feature_analysis_advanced.py`
   - **Place in correct folder**: Maintain the 5-folder structure
   - **Document justification**: Explain why existing scripts are insufficient
   - **Update this section**: Add the new script to the appropriate category above

### Naming Conventions for Derivatives
```
Original Script          →  Allowed Derivatives
train_lightgbm.py       →  train_lightgbm_v2.py, train_lightgbm_optimized.py
feature_analysis.py     →  feature_analysis_sensor.py, feature_analysis_temporal.py
model_evaluation.py     →  model_evaluation_detailed.py, model_evaluation_compare.py
```

### Why This Restriction?
- **Complexity Management**: More scripts = harder maintenance
- **Bronze Medal Focus**: Current scripts achieve LB 0.60+ target
- **Quality > Quantity**: Better to optimize existing scripts than add new ones
- **Team Efficiency**: Clear, limited script set improves collaboration

### Examples of What NOT to Do
❌ **BAD**: Creating `train_xgboost.py`, `train_catboost.py`, `train_random_forest.py`
✅ **GOOD**: Add XGBoost/CatBoost to existing `train_ensemble.py`

❌ **BAD**: Creating `eda_sensors.py`, `eda_labels.py`, `eda_participants.py`
✅ **GOOD**: Enhance existing `exploratory_analysis.py` with additional functions

❌ **BAD**: Creating `submit_to_kaggle.py`, `format_submission.py`
✅ **GOOD**: Use existing `generate_submission.py` with parameters

### When Script Addition IS Justified
Only consider new scripts for:
1. **Fundamentally different approaches**: e.g., `train_transformer.py` for attention-based models
2. **Competition-specific requirements**: e.g., `handle_test_time_adaptation.py` if required
3. **Major version rewrites**: e.g., `train_lightgbm_v2.py` with completely new approach

Always ask: "Can this be a function/parameter in an existing script?" before creating new files.

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
**Core Processing (EDA-Informed):**
- IMU normalization (Z-score per channel) - High reliability sensors
- **Critical Missing Value Handling**:
  - thm_5: 5.79% missing (forward fill or median imputation)
  - tof_5: 5.24% missing (co-occurring with thm_5)
  - Create missing indicators: `thm_5_available`, `tof_5_available`
- Thermopile noise reduction and outlier detection
- Timestamp validation (50Hz consistency across 1.41s sequences)
- **Participant grouping for CV**: 81 participants → 5-fold = ~16 participants per fold

### Silver Layer - Feature Engineering (EDA-Optimized)
**Priority 1 Features (EDA-Verified High Impact):**
- **IMU Engineering**: `acc_magnitude = sqrt(acc_x² + acc_y² + acc_z²)`
- **Motion Derivatives**: velocity (diff), jerk (diff²) for gesture detection
- **Rolling Statistics**: mean/std/min/max over 5,10,20 timesteps (1.41s sequences)
- **Missing Value Indicators**: Binary flags for sensor availability

**Priority 2 Features (Frequency Domain):**
- **FFT Analysis**: Spectral energy in bands [0-2Hz, 2-5Hz, 5-10Hz, 10-25Hz]
- **ToF Dimensionality Reduction**: PCA to 8-16 components per 64-channel sensor
- **Cross-modal Correlations**: IMU-Thermopile, IMU-ToF fusion features

**Domain-Specific BFRB Features:**
- **Gesture-Specific Patterns**: Hand-to-face proximity (ToF), contact temperature (Thermopile)
- **Behavioral Signatures**: Repetitive motion detection, touch vs non-touch classification
- **18-Class Specialization**: Features targeting top gestures ("Text on phone", "Neck scratch")

### Gold Layer - Model Preparation
**Data Formats:**
- Tabular features for tree-based models (LightGBM/CatBoost)
- Sequence data for deep learning (CNN/RNN)
- GroupKFold cross-validation setup

## 【MODEL IMPLEMENTATION】

### Configuration-Driven Model Selection
- **Implementation**: `src/models.py` with `src/config/strategy.py` integration
- **Dynamic Parameters**: Model parameters loaded from `config/project_config.yaml`
- **Phase-Aware**: Algorithm selection based on `PROJECT_PHASE` environment variable
- **Baseline Focus**: Single algorithm when `PROJECT_PHASE=baseline`

**Model Configuration System**:
```python
from src.config import get_project_config, get_primary_algorithm
config = get_project_config()
model_params = config.get_model_params("lightgbm")
enabled_algorithms = config.get_enabled_algorithms()
```

### Training Configuration (Configuration-Driven)
- **Algorithm Selection**: Controlled by `config/project_config.yaml`
- **Phase Management**: `baseline` → `optimization` → `ensemble`
- **CV Strategy**: `${CV_STRATEGY}` by `${GROUP_COLUMN}` (from config)
- **Parameters**: Dynamic loading from config files
- **Validation**: Configuration validation with warning system

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

### Time-Series Specific Considerations (EDA-Calibrated)
- **Window Size**: EDA shows 1.41s average sequences → use 1-3 second windows (50-150 samples)
- **Overlap**: 50-80% overlap for training (accommodates short sequences)
- **Normalization**: Per-participant recommended (81 participants with individual differences)
- **Class Imbalance**: 5.9:1 ratio requires focal loss or weighted sampling
- **Augmentation**: Preserve sensor relationships across IMU+ToF+Thermopile modalities

### Common Pitfalls to Avoid (EDA-Informed)
- Using standard KFold instead of GroupKFold (81 participants must be split properly)
- **Critical**: Ignoring thm_5 and tof_5 missing patterns (5-6% missing, co-occurring)
- Over-focusing on IMU only (missing ToF proximity and Thermopile contact signals)
- **Class Imbalance**: Not addressing 5.9:1 ratio between most/least frequent gestures
- **Problematic Participants**: Not handling SUBJ_044680/SUBJ_016552 with 100% missing sensors

## 【BRONZE MEDAL ROADMAP】5-Week Detailed Plan

### Week 1: Foundation & Baseline (Target: LB 0.50+)

#### Day 1-2: Data Understanding & EDA ✅ COMPLETED
**EDA Results Summary** (Reference: [EDA結果.md](docs/project-info/EDA結果.md)):
- **Data Scale**: 574,945 train rows, 81 participants, 8,151 sequences
- **Sensor Analysis**: IMU (reliable), ToF/Thermopile (some missing patterns identified)
- **Critical Finding**: thm_5 (5.79%) and tof_5 (5.24%) missing with 94% co-occurrence
- **Target Distribution**: Binary 59.84% vs 40.16%, Multi-class 5.9:1 imbalance
- **Sequence Length**: Average 1.41s (70.5 samples at 50Hz)
- **Problematic Cases**: 2 participants with 100% missing sensor 5 data

#### Day 3-4: Preprocessing Pipeline (EDA-Optimized)
**Missing Value Strategy (Critical for sensors 5):**
- **thm_5**: Forward fill or median imputation from thm_1-4
- **tof_5**: PCA-based reconstruction or zero-fill with availability flag
- **Create Indicators**: `thm_5_available`, `tof_5_available` binary features
- **Participant Exclusion**: Consider excluding SUBJ_044680/SUBJ_016552 if problematic

**Normalization (Per-Participant Recommended):**
- Z-score normalization for IMU data (per participant to handle individual differences)
- Thermopile: Per-participant baseline temperature correction
- ToF: Per-participant distance calibration

**Segmentation (Sequence-Aware):**
- **Window Size**: 1.5-second windows (75 samples) to match 1.41s average
- 70% overlap for training (accommodate short sequences)
- Label assignment: Majority vote within window

#### Day 5-6: Feature Engineering (EDA-Priority Based)
**Priority 1 - Essential Features:**
- **IMU Magnitude**: `acc_magnitude = sqrt(acc_x² + acc_y² + acc_z²)`
- **Motion Derivatives**: velocity (diff), jerk (diff²) - critical for gesture detection
- **Rolling Statistics**: mean/std/min/max over 5,10,20 timesteps (optimized for 1.41s sequences)
- **Missing Indicators**: Binary flags for sensor 5 availability

**Priority 2 - Advanced Features:**
- **tsfresh Integration**: ComprehensiveFCParameters (monitor memory with 574K rows)
- **FFT Spectral**: Energy bands [0-2Hz, 2-5Hz, 5-10Hz, 10-25Hz] for gesture frequency
- **ToF PCA**: Reduce 64 channels to 8-16 components per sensor

**BFRB-Specific Features:**
- **Proximity**: min(tof_channels) for hand-to-face distance
- **Contact Detection**: Thermopile spike detection for touch events
- **Top Gesture Features**: Specialized for "Text on phone" (10.17%), "Neck scratch" (9.85%)

#### Day 7: Baseline Model Construction (EDA-Informed)
**GroupKFold Implementation (Critical for 81 participants):**
- **Group Splitting**: 81 participants → 5 folds = ~16 participants per fold
- **Validation**: Ensure no participant overlap between train/validation
- **Class Distribution**: Handle 5.9:1 imbalance across folds

**LightGBM Baseline (Class-Aware):**
- **Class Weights**: Implement for 5.9:1 imbalance ratio
- **Dual Model Strategy**: Binary classifier + 18-class multiclass
- **Evaluation**: 0.5 × (Binary F1 + Macro F1) composite metric
- **Missing Handling**: Utilize `thm_5_available`, `tof_5_available` features

**Expected Initial Results:**
- **Target CV**: 0.50-0.52 (baseline with essential features)
- **Target LB**: 0.50-0.51 (accounting for potential CV-LB gap)
- **Success Criteria**: Both Binary F1 and Macro F1 > 0.45 individually

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

### Week 4-5: Advanced Optimization & Robustness
**Feature Selection & Optimization:**
- Channel importance analysis for inference speed
- Binary vs Multi-class metric weighting optimization
- Threshold tuning for optimal F1 composite score

**Private LB Preparation:**
- CV-LB alignment validation
- Robust validation strategies
- Final ensemble weight calibration

## 【EXPECTED SCORE PROGRESSION】

| Week | Method | Expected CV Score | Expected LB Score |
|------|--------|------------------|------------------|
| 1 | tsfresh + LightGBM | 0.48-0.52 | 0.47-0.51 |
| 2 | + 1D CNN | 0.55-0.58 | 0.54-0.57 |
| 2 | + Data Augmentation | 0.57-0.60 | 0.56-0.59 |
| 3 | + Multimodal Fusion | 0.60-0.63 | 0.59-0.62 |
| 3 | + Ensemble | 0.62-0.65 | 0.61-0.64 |

## 【CRITICAL IMPLEMENTATION WARNINGS】

### Data Leakage Prevention (MANDATORY)
- **GroupKFold Only**: Must use participant_id for group splitting
- **Temporal Continuity**: Consider time-series continuity in splits
- **Test Participant Isolation**: Never use test participant IDs in training

### Computational Resource Management
- **Memory Monitoring**: tsfresh execution requires significant memory
- **GPU Batch Sizing**: Adjust batch size according to available GPU memory
- **Parallel Processing**: Utilize multi-core processing where possible

### Evaluation Metric Understanding
- **Dual Monitoring**: Track Binary F1 and Macro F1 separately
- **Bottleneck Identification**: Identify which metric limits overall performance
- **Class Imbalance**: Address through appropriate sampling or loss weighting

### Success Implementation Keys
1. **Incremental Progress**: Achieve each week's target consistently
2. **Error Analysis**: Deep-dive into misclassification patterns
3. **Domain Knowledge**: Understand BFRB behavioral characteristics
4. **Experiment Tracking**: Document and compare all experimental results

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