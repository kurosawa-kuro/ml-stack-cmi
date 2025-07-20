"""Bronze Level Data Management for CMI Sensor Data
Raw Sensor Data Standardization & Quality Assurance (Entry Point to Medallion Pipeline)
"""

from typing import Tuple, Dict, Any, Optional, List
import warnings

import duckdb
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest

DB_PATH = "/home/wsl/dev/my-study/ml/ml-stack-cmi/data/kaggle_datasets.duckdb"

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in multiply')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in reduce')

# CMI Sensor Data Configuration
SENSOR_COLUMNS = {
    'accelerometer': ['acc_x', 'acc_y', 'acc_z'],
    'gyroscope': ['rot_w', 'rot_x', 'rot_y', 'rot_z'],
    'thermopile': ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'],
    'tof_sensors': [f'tof_{sensor}_v{channel}' for sensor in range(1, 6) for channel in range(64)]
}

# Metadata columns
METADATA_COLUMNS = ['row_id', 'sequence_type', 'sequence_id', 'sequence_counter', 
                   'subject', 'orientation', 'behavior', 'phase', 'gesture']

# Target column for classification
TARGET_COLUMNS = ['behavior', 'gesture']


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Raw CMI sensor data access point - Single source entry to Medallion pipeline"""
    conn = duckdb.connect(DB_PATH)
    try:
        train = conn.execute("SELECT * FROM cmi_detect_behavior_with_sensor_data.train").df()
        test = conn.execute("SELECT * FROM cmi_detect_behavior_with_sensor_data.test").df()
    except Exception as e:
        print(f"Error loading CMI data: {e}")
        raise
    finally:
        conn.close()
    
    # Explicit dtype setting for sensor data optimization
    train = _set_optimal_dtypes_cmi(train)
    test = _set_optimal_dtypes_cmi(test)
    
    return train, test


def _set_optimal_dtypes_cmi(df: pd.DataFrame) -> pd.DataFrame:
    """Set optimal dtypes for CMI sensor data LightGBM compatibility and performance"""
    df = df.copy()
    
    # Sensor data - use float32 for memory efficiency
    all_sensor_cols = (
        SENSOR_COLUMNS['accelerometer'] + 
        SENSOR_COLUMNS['gyroscope'] + 
        SENSOR_COLUMNS['thermopile'] + 
        SENSOR_COLUMNS['tof_sensors']
    )
    
    for col in all_sensor_cols:
        if col in df.columns:
            # Convert sensor data to float32 for memory efficiency
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    
    # Sequence counter as int32
    if 'sequence_counter' in df.columns:
        df['sequence_counter'] = df['sequence_counter'].astype('int32')
    
    # Categorical features - ensure object type for processing
    categorical_cols = ['sequence_type', 'subject', 'orientation', 'behavior', 'phase', 'gesture']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')
    
    # String IDs
    id_cols = ['row_id', 'sequence_id']
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')
    
    return df


def validate_data_quality_cmi(df: pd.DataFrame) -> Dict[str, Any]:
    """Type validation and range guards for CMI sensor data quality assurance"""
    validation_results = {
        'type_validation': {},
        'range_validation': {},
        'schema_validation': {},
        'quality_metrics': {},
        'sensor_validation': {}
    }
    
    # Type validation for sensor data
    all_sensor_cols = (
        SENSOR_COLUMNS['accelerometer'] + 
        SENSOR_COLUMNS['gyroscope'] + 
        SENSOR_COLUMNS['thermopile']
    )
    
    for col in all_sensor_cols:
        if col in df.columns:
            validation_results['type_validation'][col] = pd.api.types.is_numeric_dtype(df[col])
    
    # Categorical type validation
    for col in ['sequence_type', 'subject', 'orientation', 'behavior', 'phase', 'gesture']:
        if col in df.columns:
            validation_results['type_validation'][col] = df[col].dtype == 'object'
    
    # Range validation for sensor data
    if 'acc_x' in df.columns:
        acc_cols = SENSOR_COLUMNS['accelerometer']
        for col in acc_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                validation_results['range_validation'][col] = {
                    'reasonable_range': (df[col].abs() <= 50).all(),  # Reasonable accelerometer range
                    'finite_values': np.isfinite(df[col]).all()
                }
    
    # Thermopile validation
    if 'thm_1' in df.columns:
        thm_cols = SENSOR_COLUMNS['thermopile']
        for col in thm_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                validation_results['range_validation'][col] = {
                    'reasonable_temp_range': ((df[col] >= 0) & (df[col] <= 100)).all(),  # Reasonable temp range
                    'finite_values': np.isfinite(df[col]).all()
                }
    
    # ToF sensor validation (many will have -1.0 for missing values)
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    if tof_cols:
        validation_results['sensor_validation']['tof_missing_rate'] = {
            col: (df[col] == -1.0).mean() for col in tof_cols[:10]  # Sample first 10
        }
    
    # Participant and sequence validation
    if 'subject' in df.columns:
        validation_results['schema_validation']['unique_subjects'] = df['subject'].nunique()
        validation_results['schema_validation']['subject_sample_sizes'] = df['subject'].value_counts().describe().to_dict()
    
    if 'sequence_id' in df.columns:
        validation_results['schema_validation']['unique_sequences'] = df['sequence_id'].nunique()
    
    # Quality metrics
    validation_results['quality_metrics'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values_excluding_tof': df.drop(columns=tof_cols, errors='ignore').isnull().sum().sum(),
        'tof_missing_values': df[tof_cols].isnull().sum().sum() if tof_cols else 0,
        'duplicate_rows': df.duplicated().sum(),
        'sensor_columns_count': len([col for col in df.columns if any(col.startswith(prefix) for prefix in ['acc_', 'rot_', 'thm_', 'tof_'])])
    }
    
    return validation_results


def normalize_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize sensor data for CMI dataset - Z-score normalization per sensor type"""
    df = df.copy()
    
    # IMU normalization (per-participant to handle individual differences)
    imu_cols = SENSOR_COLUMNS['accelerometer'] + SENSOR_COLUMNS['gyroscope']
    
    for col in imu_cols:
        if col in df.columns:
            # Group by participant for normalization to prevent leakage
            if 'subject' in df.columns:
                # Per-participant normalization
                df[col] = df.groupby('subject')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
            else:
                # Global normalization if no participant info
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    
    # Thermopile normalization (temperature sensors)
    thm_cols = SENSOR_COLUMNS['thermopile']
    for col in thm_cols:
        if col in df.columns:
            # Global normalization for temperature (less individual variation expected)
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    
    return df


def handle_tof_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle ToF sensor missing values (-1.0 indicates no detection)"""
    df = df.copy()
    
    # Get ToF columns
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    
    if tof_cols:
        # Create all missing flags at once to avoid DataFrame fragmentation
        missing_flags_data = {}
        
        for col in tof_cols:
            missing_flag_col = f"{col}_missing"
            missing_flags_data[missing_flag_col] = (df[col] == -1.0).astype('int8')
            
            # Replace -1.0 with NaN for proper handling
            df[col] = df[col].replace(-1.0, np.nan)
        
        # Add all missing flags at once using pd.concat for better performance
        missing_flags_df = pd.DataFrame(missing_flags_data, index=df.index)
        df = pd.concat([df, missing_flags_df], axis=1)
    
    return df


def create_participant_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create participant grouping information for GroupKFold CV"""
    df = df.copy()
    
    if 'subject' in df.columns:
        # Create numeric participant IDs for GroupKFold
        unique_subjects = df['subject'].unique()
        subject_to_id = {subject: idx for idx, subject in enumerate(unique_subjects)}
        df['participant_id'] = df['subject'].map(subject_to_id)
        
        # Add participant statistics
        participant_stats = df.groupby('subject').agg({
            'sequence_id': 'nunique',
            'sequence_counter': 'count'
        }).rename(columns={
            'sequence_id': 'sequences_per_participant',
            'sequence_counter': 'samples_per_participant'
        })
        
        df = df.merge(participant_stats, left_on='subject', right_index=True, how='left')
    
    return df


def create_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create sequence-level features for time-series analysis"""
    df = df.copy()
    
    if 'sequence_id' in df.columns and 'sequence_counter' in df.columns:
        # Sequence position features
        df['sequence_position'] = df['sequence_counter']
        
        # Sequence length (samples per sequence)
        seq_lengths = df.groupby('sequence_id')['sequence_counter'].max() + 1
        df = df.merge(seq_lengths.rename('sequence_length'), left_on='sequence_id', right_index=True, how='left')
        
        # Relative position in sequence
        df['sequence_progress'] = df['sequence_counter'] / df['sequence_length']
        
        # Sequence statistics for each participant
        if 'subject' in df.columns:
            participant_seq_stats = df.groupby('subject')['sequence_length'].agg(['mean', 'std', 'count']).add_prefix('participant_seq_')
            df = df.merge(participant_seq_stats, left_on='subject', right_index=True, how='left')
    
    return df


def advanced_missing_strategy_cmi(df: pd.DataFrame) -> pd.DataFrame:
    """Missing value intelligence for CMI sensor data with LightGBM native handling"""
    df = df.copy()
    
    # Create missing flags for critical sensors
    sensor_groups = ['accelerometer', 'gyroscope', 'thermopile']
    
    for group in sensor_groups:
        cols = SENSOR_COLUMNS[group]
        for col in cols:
            if col in df.columns:
                missing_flag_col = f"{col}_missing"
                df[missing_flag_col] = df[col].isna().astype('int8')
    
    # Create group-level missing patterns
    # IMU missing pattern (all IMU sensors missing together)
    imu_cols = SENSOR_COLUMNS['accelerometer'] + SENSOR_COLUMNS['gyroscope']
    if all(col in df.columns for col in imu_cols):
        imu_missing = df[imu_cols].isna().all(axis=1)
        df['imu_complete_missing'] = imu_missing.astype('int8')
    
    # Thermopile missing pattern
    thm_cols = SENSOR_COLUMNS['thermopile']
    if all(col in df.columns for col in thm_cols):
        thm_missing = df[thm_cols].isna().all(axis=1)
        df['thermopile_complete_missing'] = thm_missing.astype('int8')
    
    # Cross-sensor missing patterns
    if 'acc_x' in df.columns and 'thm_1' in df.columns:
        # Both IMU and thermal missing (complete sensor failure)
        df['sensor_failure_missing'] = (
            df['imu_complete_missing'] & df['thermopile_complete_missing']
        ).astype('int8')
    
    return df


def winsorize_outliers_cmi(df: pd.DataFrame, percentile: float = 0.01) -> pd.DataFrame:
    """Sensor-specific outlier clipping for numeric stability"""
    df = df.copy()
    
    # Different winsorization for different sensor types
    sensor_configs = [
        ('accelerometer', 0.001),  # More conservative for IMU
        ('gyroscope', 0.001),
        ('thermopile', 0.01)       # Less conservative for temperature
    ]
    
    for sensor_type, perc in sensor_configs:
        cols = SENSOR_COLUMNS[sensor_type]
        
        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isna().sum() > len(df) * 0.5:
                    continue
                    
                # Calculate bounds using quantiles
                lower_bound = df[col].quantile(perc)
                upper_bound = df[col].quantile(1 - perc)
                
                # Apply clipping
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


def create_bronze_tables() -> None:
    """Creates standardized bronze.train, bronze.test tables for CMI sensor data"""
    conn = duckdb.connect(DB_PATH)
    
    # Create bronze schema
    conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    
    # Load raw data
    train_raw, test_raw = load_data()
    
    print(f"Loaded raw data: train {train_raw.shape}, test {test_raw.shape}")
    
    # Apply bronze layer processing pipeline
    train_bronze = normalize_sensor_data(train_raw)
    test_bronze = normalize_sensor_data(test_raw)
    
    train_bronze = handle_tof_missing_values(train_bronze)
    test_bronze = handle_tof_missing_values(test_bronze)
    
    train_bronze = create_participant_groups(train_bronze)
    test_bronze = create_participant_groups(test_bronze)
    
    train_bronze = create_sequence_features(train_bronze)
    test_bronze = create_sequence_features(test_bronze)
    
    train_bronze = advanced_missing_strategy_cmi(train_bronze)
    test_bronze = advanced_missing_strategy_cmi(test_bronze)
    
    train_bronze = winsorize_outliers_cmi(train_bronze)
    test_bronze = winsorize_outliers_cmi(test_bronze)
    
    # Validate data quality
    train_validation = validate_data_quality_cmi(train_bronze)
    test_validation = validate_data_quality_cmi(test_bronze)
    
    # Create bronze tables
    conn.execute("DROP TABLE IF EXISTS bronze.train")
    conn.execute("DROP TABLE IF EXISTS bronze.test")
    
    conn.register("train_bronze_df", train_bronze)
    conn.register("test_bronze_df", test_bronze)
    
    conn.execute("CREATE TABLE bronze.train AS SELECT * FROM train_bronze_df")
    conn.execute("CREATE TABLE bronze.test AS SELECT * FROM test_bronze_df")
    
    print("Bronze tables created:")
    print(f"- bronze.train: {len(train_bronze)} rows, {len(train_bronze.columns)} columns")
    print(f"- bronze.test: {len(test_bronze)} rows, {len(test_bronze.columns)} columns")
    print(f"- Unique participants: {train_validation['schema_validation'].get('unique_subjects', 'N/A')}")
    print(f"- Sensor columns: {train_validation['quality_metrics']['sensor_columns_count']}")
    print(f"- Missing values (excl. ToF): {train_validation['quality_metrics']['missing_values_excluding_tof']}")
    
    conn.close()


def load_bronze_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load bronze layer CMI sensor data"""
    conn = duckdb.connect(DB_PATH)
    try:
        train = conn.execute("SELECT * FROM bronze.train").df()
        test = conn.execute("SELECT * FROM bronze.test").df()
    except Exception as e:
        print(f"Bronze tables not found. Creating them first...")
        create_bronze_tables()
        train = conn.execute("SELECT * FROM bronze.train").df()
        test = conn.execute("SELECT * FROM bronze.test").df()
    finally:
        conn.close()
    
    return train, test


# ===== Sklearn-Compatible Transformers for Pipeline Integration =====

class CMIBronzePreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for CMI Bronze layer processing"""
    
    def __init__(self, normalize_sensors: bool = True, handle_missing: bool = True):
        self.normalize_sensors = normalize_sensors
        self.handle_missing = handle_missing
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Fit the transformer (no fitting required for Bronze layer)"""
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply Bronze layer transformations for CMI sensor data"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        X_transformed = X.copy()
        
        if self.normalize_sensors:
            X_transformed = normalize_sensor_data(X_transformed)
        
        X_transformed = handle_tof_missing_values(X_transformed)
        X_transformed = create_participant_groups(X_transformed)
        X_transformed = create_sequence_features(X_transformed)
        
        if self.handle_missing:
            X_transformed = advanced_missing_strategy_cmi(X_transformed)
        
        X_transformed = winsorize_outliers_cmi(X_transformed)
        
        return X_transformed


class FoldSafeCMIPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-safe CMI Bronze preprocessor for CV integration"""
    
    def __init__(self):
        self.participant_stats = {}
        self.sensor_stats = {}
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Learn normalization parameters from training data only"""
        self.participant_stats = {}
        self.sensor_stats = {}
        
        # Learn participant-level statistics
        if 'subject' in X.columns:
            imu_cols = SENSOR_COLUMNS['accelerometer'] + SENSOR_COLUMNS['gyroscope']
            for col in imu_cols:
                if col in X.columns:
                    self.participant_stats[col] = X.groupby('subject')[col].agg(['mean', 'std']).to_dict()
        
        # Learn global sensor statistics for thermopile
        thm_cols = SENSOR_COLUMNS['thermopile']
        for col in thm_cols:
            if col in X.columns:
                self.sensor_stats[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std()
                }
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply fold-safe transformations"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        X_transformed = X.copy()
        
        # Apply learned normalization
        for col, stats in self.participant_stats.items():
            if col in X_transformed.columns and 'subject' in X_transformed.columns:
                for subject in X_transformed['subject'].unique():
                    if subject in stats['mean']:
                        mask = X_transformed['subject'] == subject
                        mean_val = stats['mean'][subject]
                        std_val = stats['std'][subject]
                        X_transformed.loc[mask, col] = (X_transformed.loc[mask, col] - mean_val) / (std_val + 1e-8)
        
        for col, stats in self.sensor_stats.items():
            if col in X_transformed.columns:
                X_transformed[col] = (X_transformed[col] - stats['mean']) / (stats['std'] + 1e-8)
        
        # Apply other transformations
        X_transformed = handle_tof_missing_values(X_transformed)
        X_transformed = create_participant_groups(X_transformed)
        X_transformed = create_sequence_features(X_transformed)
        X_transformed = advanced_missing_strategy_cmi(X_transformed)
        
        return X_transformed


# Legacy function for backward compatibility
def quick_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy preprocessing function - USE create_bronze_tables() for production pipeline"""
    # For CMI data, apply basic preprocessing
    df = df.copy()
    
    # Basic sensor normalization
    df = normalize_sensor_data(df)
    
    # Handle ToF missing values
    df = handle_tof_missing_values(df)
    
    # Create participant info
    df = create_participant_groups(df)
    
    return df


# Alias for backward compatibility
validate_data_quality = validate_data_quality_cmi