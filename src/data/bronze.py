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
    """Set optimal dtypes for sensor data LightGBM compatibility and performance"""
    df = df.copy()
    
    # Numeric features - use float32 for memory efficiency
    numeric_cols = [col for col in df.columns if col in SENSOR_COLUMNS['accelerometer'] + SENSOR_COLUMNS['gyroscope'] + SENSOR_COLUMNS['thermopile']]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    
    # ID column as int32
    if 'row_id' in df.columns:
        df['row_id'] = df['row_id'].astype('int32')
    
    # Categorical features - ensure object type for processing
    categorical_cols = [col for col in df.columns if col in METADATA_COLUMNS]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')
    
    return df


def validate_data_quality_cmi(df: pd.DataFrame) -> Dict[str, Any]:
    """Type validation and range guards for sensor data quality assurance"""
    validation_results = {
        'type_validation': {},
        'range_validation': {},
        'schema_validation': {},
        'quality_metrics': {},
        'feature_validation': {}
    }
    
    # Type validation for numeric features
    numeric_cols = [col for col in df.columns if col in SENSOR_COLUMNS['accelerometer'] + SENSOR_COLUMNS['gyroscope'] + SENSOR_COLUMNS['thermopile']]
    for col in numeric_cols:
        if col in df.columns:
            validation_results['type_validation'][col] = pd.api.types.is_numeric_dtype(df[col])
    
    # Categorical type validation
    categorical_cols = [col for col in df.columns if col in METADATA_COLUMNS]
    for col in categorical_cols:
        if col in df.columns:
            validation_results['type_validation'][col] = df[col].dtype == 'object'
    
    # Range validation for numeric features
    for col in numeric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if col in SENSOR_COLUMNS['accelerometer']:
                validation_results['range_validation'][col] = {
                    'reasonable_range': ((df[col] >= -10).all() & (df[col] <= 10).all()), # Accelerometer range
                    'finite_values': np.isfinite(df[col]).all()
                }
            elif col in SENSOR_COLUMNS['gyroscope']:
                validation_results['range_validation'][col] = {
                    'reasonable_range': ((df[col] >= -10).all() & (df[col] <= 10).all()), # Gyroscope range
                    'finite_values': np.isfinite(df[col]).all()
                }
            elif col in SENSOR_COLUMNS['thermopile']:
                validation_results['range_validation'][col] = {
                    'reasonable_range': ((df[col] >= 0).all() & (df[col] <= 100).all()), # Thermopile range
                    'finite_values': np.isfinite(df[col]).all()
                }
    
    # Categorical value validation
    if 'behavior' in df.columns:
        validation_results['feature_validation']['behavior_values'] = df['behavior'].value_counts().to_dict()
    
    if 'phase' in df.columns:
        validation_results['feature_validation']['phase_values'] = df['phase'].value_counts().to_dict()
    
    if 'gesture' in df.columns:
        validation_results['feature_validation']['gesture_values'] = df['gesture'].value_counts().to_dict()
    
    # Schema validation
    validation_results['schema_validation']['total_samples'] = len(df)
    validation_results['schema_validation']['unique_ids'] = df['row_id'].nunique() if 'row_id' in df.columns else 0
    
    # Quality metrics
    validation_results['quality_metrics'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_features_count': len([col for col in df.columns if col in numeric_cols]),
        'categorical_features_count': len([col for col in df.columns if col in categorical_cols])
    }
    
    return validation_results


def normalize_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize sensor data for CMI competition"""
    df = df.copy()
    
    # Sensor-specific normalization
    # IMU sensors (accelerometer, gyroscope)
    imu_cols = SENSOR_COLUMNS['accelerometer'] + SENSOR_COLUMNS['gyroscope']
    for col in imu_cols:
        if col in df.columns:
            # Z-score normalization per sensor channel
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val
    
    # Thermopile sensors - temperature normalization
    thermal_cols = SENSOR_COLUMNS['thermopile']
    for col in thermal_cols:
        if col in df.columns:
            # Normalize to 0-1 range based on typical temperature range
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
    
    return df


def handle_categorical_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in categorical features for CMI data"""
    df = df.copy()
    
    # Categorical features
    categorical_cols = [col for col in df.columns if col in METADATA_COLUMNS]
    
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            # Fill with most frequent value
            most_frequent = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(most_frequent)
    
    return df


def handle_tof_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in ToF sensor data"""
    df = df.copy()
    
    # ToF sensors often have missing values due to measurement issues
    tof_cols = SENSOR_COLUMNS['tof_sensors']
    
    for col in tof_cols:
        if col in df.columns and df[col].isnull().any():
            # Fill with median value (typical distance)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    return df


def create_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create sensor-specific features for CMI data"""
    df = df.copy()
    
    # IMU motion intensity
    acc_cols = SENSOR_COLUMNS['accelerometer']
    if all(col in df.columns for col in acc_cols):
        df['imu_total_motion'] = np.sqrt(df[acc_cols[0]]**2 + df[acc_cols[1]]**2 + df[acc_cols[2]]**2)
    
    # Thermal distance interaction
    thermal_cols = SENSOR_COLUMNS['thermopile']
    if len(thermal_cols) >= 2:
        df['thermal_distance_interaction'] = df[thermal_cols[0]] - df[thermal_cols[1]]
    
    # Movement intensity from gyroscope
    gyro_cols = SENSOR_COLUMNS['gyroscope']
    if all(col in df.columns for col in gyro_cols[:3]):  # rot_x, rot_y, rot_z
        df['movement_intensity'] = np.sqrt(df[gyro_cols[0]]**2 + df[gyro_cols[1]]**2 + df[gyro_cols[2]]**2)
    
    return df


def create_participant_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create participant groups for GroupKFold CV"""
    df = df.copy()
    
    # Use subject column as participant_id for GroupKFold
    if 'subject' in df.columns:
        df['participant_id'] = df['subject']
    else:
        # If no subject column, create groups based on sequence_id
        df['participant_id'] = df['sequence_id'] if 'sequence_id' in df.columns else df.index // 1000
    
    return df


def create_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create sequence-based features for time-series data"""
    df = df.copy()
    
    # Sequence-based features
    if 'sequence_id' in df.columns:
        # Sequence length
        sequence_lengths = df.groupby('sequence_id').size()
        df['sequence_length'] = df['sequence_id'].map(sequence_lengths)
        
        # Sequence position
        df['sequence_position'] = df.groupby('sequence_id').cumcount()
    
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
    
    train_bronze = create_sensor_features(train_bronze)
    test_bronze = create_sensor_features(test_bronze)
    
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
        X_transformed = create_sensor_features(X_transformed)
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
        X_transformed = create_sensor_features(X_transformed)
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