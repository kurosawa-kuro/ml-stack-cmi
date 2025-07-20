"""
Gold Layer Data Management for CMI Sensor Data
ML-Ready Data Preparation with GroupKFold Support
Configuration-Driven ML Pipeline
CLAUDE.md: ML-ready sequences for LightGBM/CNN training with participant-based CV
"""

from typing import Any, List, Optional, Tuple, Dict
import logging

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
import warnings

# Configuration-driven imports
try:
    from ..config import get_project_config
    CONFIG_AVAILABLE = True
except ImportError:
    try:
        from config import get_project_config
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration-driven database path
def get_db_path() -> str:
    """Get database path from configuration"""
    if CONFIG_AVAILABLE:
        config = get_project_config()
        return config.data.source_path
    else:
        return "/home/wsl/dev/my-study/ml/ml-stack-cmi/data/kaggle_datasets.duckdb"

DB_PATH = get_db_path()


def clean_and_validate_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """CMI sensor data cleaning and validation (CLAUDE.md specification)
    
    Handles multimodal sensor data with overflow protection
    """
    df = df.copy()

    # Sensor-specific feature cleaning
    sensor_cols = {
        'imu': [col for col in df.columns if any(prefix in col for prefix in ['acc_', 'rot_', 'imu_'])],
        'thermal': [col for col in df.columns if any(prefix in col for prefix in ['thm_', 'thermal'])],
        'tof': [col for col in df.columns if any(prefix in col for prefix in ['tof_', 'proximity'])],
        'behavior': [col for col in df.columns if any(prefix in col for prefix in ['movement_', 'close_proximity'])]
    }
    
    # Sensor-specific value ranges for overflow protection
    sensor_limits = {
        'imu': (-100, 100),      # IMU sensor values
        'thermal': (-50, 100),   # Temperature values
        'tof': (0, 1000),        # Distance values
        'behavior': (0, 10)      # Behavior indicators
    }
    
    # Apply sensor-specific cleaning
    for sensor_type, columns in sensor_cols.items():
        if columns:
            limit_range = sensor_limits.get(sensor_type, (-1e3, 1e3))
            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    # Replace infinite values
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Sensor-specific clipping
                    df[col] = df[col].clip(lower=limit_range[0], upper=limit_range[1])
                    
                    # Fill missing values with sensor-appropriate defaults
                    if df[col].isna().any():
                        if sensor_type == 'tof':
                            df[col] = df[col].fillna(500)  # Medium distance
                        elif sensor_type == 'thermal':
                            df[col] = df[col].fillna(25)   # Room temperature
                        else:
                            df[col] = df[col].fillna(df[col].median())
    
    # General numeric column processing for non-sensor features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    sensor_specific_cols = set()
    for cols in sensor_cols.values():
        sensor_specific_cols.update(cols)
    
    general_numeric_cols = [col for col in numeric_cols if col not in sensor_specific_cols and col not in ['id', 'participant_id', 'series_id']]
    
    for col in general_numeric_cols:
        # Conservative outlier handling for general features
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].notna().any():
            # More conservative clipping for general features
            df[col] = df[col].clip(-1e3, 1e3)
            # Fill missing values
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    return df


def select_sensor_features(df: pd.DataFrame, target_col: str, k: int = 100, method: str = "combined") -> List[str]:
    """Configuration-driven sensor-aware feature selection for CMI data (CLAUDE.md specification)
    
    Prioritizes multimodal sensor features for BFRB detection with configuration control
    """
    # Get feature selection configuration
    if CONFIG_AVAILABLE:
        config = get_project_config()
        configured_k = config.data.max_features
        if configured_k:
            k = configured_k
            logger.info(f"Using configured max_features: {k}")
        
        # Get selection method from configuration if available
        try:
            selection_config = getattr(config, 'features', {}).get('selection', {})
            configured_method = selection_config.get('method', method)
            if configured_method != method:
                method = configured_method
                logger.info(f"Using configured selection method: {method}")
        except AttributeError:
            # Fallback if features configuration is not available
            logger.info(f"Using default selection method: {method}")
        
        logger.info(f"Feature selection in {config.phase.value} phase: k={k}, method={method}")
    
    # Exclude ID and target columns
    exclude_cols = ['id', 'participant_id', 'series_id', target_col]
    if f"{target_col}_encoded" in df.columns:
        exclude_cols.append(f"{target_col}_encoded")
    
    # Get all numeric feature columns
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    if len(feature_cols) <= k:
        logger.info(f"Feature count ({len(feature_cols)}) <= max_features ({k}), using all features")
        return feature_cols
    
    # Prioritize sensor features for CMI competition
    priority_features = []
    
    # High priority: Core sensor fusion features
    sensor_patterns = [
        'imu_total_motion', 'thermal_distance_interaction', 'movement_intensity',
        'proximity_mean', 'thermal_contact_indicator', 'close_proximity_ratio'
    ]
    
    # Medium priority: Statistical sensor features  
    stat_patterns = [
        'imu_acc_mean', 'imu_gyro_mean', 'thermal_mean', 'tof_mean',
        'imu_acc_energy', 'thermal_gradient', 'movement_periodicity'
    ]
    
    # Low priority: Frequency domain features
    freq_patterns = [
        'spectral_centroid', 'dominant_freq', 'spectral_rolloff'
    ]
    
    # Add features by priority
    for pattern_group in [sensor_patterns, stat_patterns, freq_patterns]:
        for pattern in pattern_group:
            matching_features = [col for col in feature_cols if pattern in col]
            priority_features.extend(matching_features)
    
    # Remove duplicates while preserving order
    seen = set()
    priority_features = [x for x in priority_features if not (x in seen or seen.add(x))]
    
    # Add remaining features
    remaining_features = [col for col in feature_cols if col not in priority_features]
    all_features = priority_features + remaining_features
    
    # If we have too many features, use statistical selection
    if len(all_features) > k:
        try:
            # Prepare target variable
            if target_col in df.columns:
                if df[target_col].dtype == 'object':
                    # Binary classification for BFRB detection
                    target_values = pd.Categorical(df[target_col]).codes
                else:
                    target_values = df[target_col]
            else:
                print(f"Warning: Target column {target_col} not found, using first {k} features")
                return all_features[:k]
            
            # Prepare feature matrix
            X_df = df[all_features].copy()
            
            # Handle missing values and infinite values
            for col in X_df.columns:
                X_df[col] = X_df[col].replace([np.inf, -np.inf], np.nan)
                X_df[col] = X_df[col].fillna(X_df[col].median())
            
            X = X_df.values
            y = target_values.values
            
            # Combined feature selection
            if method == "combined":
                # F-test selection
                selector_f = SelectKBest(score_func=f_classif, k=min(k, len(all_features)))
                selector_f.fit(X, y)
                
                # Mutual information selection
                selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(k, len(all_features)))
                selector_mi.fit(X, y)
                
                # Combine scores
                f_scores = selector_f.scores_
                mi_scores = selector_mi.scores_
                
                # Normalize and combine
                f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
                mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
                
                combined_scores = f_scores_norm + mi_scores_norm
                
                # Get top features
                top_indices = np.argsort(combined_scores)[-k:][::-1]
                selected_features = [all_features[i] for i in top_indices]
            else:
                # Use specified method
                score_func = f_classif if method == "statistical" else mutual_info_classif
                selector = SelectKBest(score_func=score_func, k=min(k, len(all_features)))
                selector.fit(X, y)
                selected_indices = selector.get_support()
                selected_features = [all_features[i] for i, selected in enumerate(selected_indices) if selected]
            
            return selected_features
            
        except Exception as e:
            print(f"Warning: Statistical feature selection failed: {e}")
            return all_features[:k]
    
    return all_features[:k]


def prepare_ml_ready_data(
    df: pd.DataFrame, 
    target_col: str = "label", 
    feature_cols: List[str] = None, 
    auto_select: bool = True, 
    model_type: str = "lightgbm",
    max_features: int = 100
) -> pd.DataFrame:
    """Prepare ML-ready data for CMI sensor classification (CLAUDE.md specification)
    
    Converts Silver layer features to optimized format for BFRB detection
    """
    df = df.copy()

    # Step 1: Clean and validate sensor features
    df = clean_and_validate_sensor_features(df)

    # Step 2: Encode categorical columns (preserve original for GroupKFold)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in [target_col, 'participant_id', 'series_id']:
            # Create encoded version while keeping original
            try:
                df[f"{col}_encoded"] = pd.Categorical(df[col]).codes.astype('int32')
            except Exception as e:
                print(f"Warning: Failed to encode {col}: {e}")

    # Step 3: Feature selection optimized for sensor data
    if feature_cols is None:
        if auto_select and target_col and target_col in df.columns:
            # Sensor-aware feature selection
            selected_features = select_sensor_features(df, target_col, k=max_features)
        else:
            # Default sensor feature set for CMI data
            default_features = []
            
            # Core sensor fusion features
            sensor_features = [
                'imu_total_motion', 'thermal_distance_interaction', 'movement_intensity',
                'proximity_mean', 'thermal_contact_indicator', 'close_proximity_ratio',
                'imu_acc_mean', 'imu_gyro_mean', 'thermal_mean', 'tof_mean'
            ]
            
            # Statistical features from tsfresh
            tsfresh_features = [col for col in df.columns if col.startswith('tsfresh_')]
            
            # Frequency domain features
            freq_features = [col for col in df.columns if any(pattern in col for pattern in ['spectral_', 'dominant_freq'])]
            
            # Behavior-specific features
            behavior_features = [col for col in df.columns if any(pattern in col for pattern in ['movement_', 'proximity_', 'thermal_'])]
            
            # Combine feature sets
            default_features.extend([f for f in sensor_features if f in df.columns])
            default_features.extend(tsfresh_features[:20])  # Limit tsfresh features
            default_features.extend(freq_features[:15])     # Limit frequency features
            default_features.extend(behavior_features[:25]) # Limit behavior features
            
            # Remove duplicates
            default_features = list(dict.fromkeys(default_features))
            selected_features = default_features

        feature_cols = selected_features

    # Step 4: Ensure required columns are included
    required_cols = ['id', 'participant_id', 'series_id', target_col]
    
    # Available features only
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Build final column list
    final_cols = []
    for col in required_cols:
        if col in df.columns:
            final_cols.append(col)
    
    final_cols.extend(available_features)
    
    # Remove duplicates while preserving order
    final_cols = list(dict.fromkeys(final_cols))
    
    # Step 5: Target encoding for BFRB classification
    if target_col and target_col in df.columns:
        if df[target_col].dtype == 'object':
            # Create encoded target for classification
            label_encoder = LabelEncoder()
            df[f"{target_col}_encoded"] = label_encoder.fit_transform(df[target_col])
            if f"{target_col}_encoded" not in final_cols:
                final_cols.append(f"{target_col}_encoded")

    return df[final_cols]


def encode_bfrb_target(df: pd.DataFrame, target_col: str = "label") -> pd.DataFrame:
    """BFRB behavior target encoding (CLAUDE.md specification)
    
    Encodes multi-class BFRB behavior labels for classification
    """
    df = df.copy()
    
    if target_col in df.columns:
        if df[target_col].dtype == 'object':
            # Multi-class BFRB behavior encoding
            label_encoder = LabelEncoder()
            df[f"{target_col}_encoded"] = label_encoder.fit_transform(df[target_col])
            
            # Also create binary BFRB presence indicator
            # Assuming label 0 is "no behavior" and others are specific BFRB types
            df[f"{target_col}_binary"] = (df[f"{target_col}_encoded"] > 0).astype(int)
        else:
            # Already numeric, create binary version
            df[f"{target_col}_binary"] = (df[target_col] > 0).astype(int)
    
    return df


def create_gold_tables() -> None:
    """Creates ML-ready Gold tables for CMI sensor data (CLAUDE.md specification)
    
    Silver → Gold transformation with GroupKFold support for BFRB detection
    """
    conn = duckdb.connect(DB_PATH)

    # Create gold schema
    conn.execute("CREATE SCHEMA IF NOT EXISTS gold")

    # Load Silver layer data (dependency chain)
    try:
        train_silver = conn.execute("SELECT * FROM silver.train").df()
        test_silver = conn.execute("SELECT * FROM silver.test").df()
        print(f"Loaded Silver data - Train: {len(train_silver)} rows, Test: {len(test_silver)} rows")
    except Exception as e:
        print(f"Silver tables not found: {e}")
        print("Creating silver tables first...")
        from .silver import create_silver_tables

        create_silver_tables()
        train_silver = conn.execute("SELECT * FROM silver.train").df()
        test_silver = conn.execute("SELECT * FROM silver.test").df()
        print(f"Created Silver data - Train: {len(train_silver)} rows, Test: {len(test_silver)} rows")

    # Apply Gold layer processing pipeline (CLAUDE.md specification)
    
    # Step 1: BFRB target encoding for training data
    target_col = 'label' if 'label' in train_silver.columns else 'behavior'
    print(f"Target column detected: {target_col}")
    
    train_gold = encode_bfrb_target(train_silver, target_col=target_col)
    test_gold = test_silver.copy()  # Test data doesn't have targets

    # Step 2: ML-ready data preparation with sensor-aware cleaning
    train_gold = prepare_ml_ready_data(
        train_gold, 
        target_col=target_col, 
        auto_select=True, 
        model_type="lightgbm",
        max_features=150
    )
    
    test_gold = prepare_ml_ready_data(
        test_gold, 
        target_col=target_col, 
        auto_select=False,
        model_type="lightgbm",
        max_features=150
    )

    # Step 3: Ensure compatibility between train and test
    # Align columns (test should have same features as train, minus target)
    train_feature_cols = [col for col in train_gold.columns 
                         if col not in [target_col, f"{target_col}_encoded", f"{target_col}_binary"]]
    
    test_feature_cols = [col for col in train_feature_cols if col in test_gold.columns]
    
    # Add missing feature columns to test with default values
    for col in train_feature_cols:
        if col not in test_gold.columns:
            if pd.api.types.is_numeric_dtype(train_gold[col]):
                test_gold[col] = 0.0
            else:
                test_gold[col] = "unknown"
    
    # Reorder test columns to match train
    test_gold = test_gold[test_feature_cols]
    
    # Final validation: ensure participant_id exists for GroupKFold
    if 'participant_id' not in train_gold.columns:
        print("Warning: participant_id not found. Creating dummy participant IDs.")
        # Create dummy participant IDs based on row index
        train_gold['participant_id'] = train_gold.index // 100  # ~100 samples per participant
        test_gold['participant_id'] = test_gold.index // 100

    # Drop and create gold tables
    conn.execute("DROP TABLE IF EXISTS gold.train")
    conn.execute("DROP TABLE IF EXISTS gold.test")

    conn.register("train_gold_df", train_gold)
    conn.register("test_gold_df", test_gold)

    conn.execute("CREATE TABLE gold.train AS SELECT * FROM train_gold_df")
    conn.execute("CREATE TABLE gold.test AS SELECT * FROM test_gold_df")

    # Report creation results
    print("\n🥇 Gold layer tables created successfully:")
    print(f"  📊 gold.train: {len(train_gold):,} rows × {len(train_gold.columns)} columns")
    print(f"  📊 gold.test: {len(test_gold):,} rows × {len(test_gold.columns)} columns")
    
    # Feature analysis
    sensor_features = len([col for col in train_gold.columns if any(pattern in col for pattern in ['imu_', 'thermal_', 'tof_', 'movement_', 'proximity_'])])
    tsfresh_features = len([col for col in train_gold.columns if col.startswith('tsfresh_')])
    freq_features = len([col for col in train_gold.columns if 'spectral_' in col or 'dominant_freq' in col])
    
    print(f"\n📈 Feature composition:")
    print(f"  🤖 Sensor features: {sensor_features}")
    print(f"  📊 tsfresh features: {tsfresh_features}")
    print(f"  🌊 Frequency features: {freq_features}")
    print(f"  👥 Participants: {train_gold['participant_id'].nunique() if 'participant_id' in train_gold.columns else 'Unknown'}")
    
    if target_col in train_gold.columns:
        target_dist = train_gold[target_col].value_counts()
        print(f"\n🎯 Target distribution ({target_col}):")
        for label, count in target_dist.items():
            print(f"  {label}: {count:,} ({count/len(train_gold)*100:.1f}%)")
    
    print(f"\n✅ Gold layer ready for GroupKFold training with {target_col} classification")
    print(f"💡 Use load_gold_data() and get_ml_ready_sequences() for model training")

    conn.close()


def load_gold_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Gold layer ML-ready data (CLAUDE.md specification)"""
    conn = duckdb.connect(DB_PATH)
    
    try:
        train = conn.execute("SELECT * FROM gold.train").df()
        test = conn.execute("SELECT * FROM gold.test").df()
        
        print(f"Loaded Gold data - Train: {len(train):,} rows × {len(train.columns)} cols, Test: {len(test):,} rows × {len(test.columns)} cols")
        
        conn.close()
        return train, test
        
    except Exception as e:
        conn.close()
        raise ValueError(f"Failed to load Gold data: {e}. Run create_gold_tables() first.")


def get_ml_ready_sequences(df: pd.DataFrame, target_col: str = "label") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Get ML-ready sequences for CMI sensor data (CLAUDE.md specification)
    
    Returns:
        X: Feature matrix
        y: Target vector (encoded)
        groups: Participant IDs for GroupKFold
    """
    df = df.copy()
    
    # Determine target column
    available_targets = [target_col, f"{target_col}_encoded", "label", "label_encoded"]
    actual_target = None
    
    for target in available_targets:
        if target in df.columns:
            actual_target = target
            break
    
    if actual_target is None:
        raise ValueError(f"No target column found. Available columns: {list(df.columns)}")
    
    # Extract groups for GroupKFold (participant_id)
    if 'participant_id' in df.columns:
        groups = df['participant_id']
    else:
        print("Warning: participant_id not found. Creating dummy groups.")
        groups = pd.Series(df.index // 100, index=df.index)  # Dummy groups
    
    # Prepare target variable
    if actual_target.endswith('_encoded'):
        y = df[actual_target].astype('int32')
    else:
        if df[actual_target].dtype == 'object':
            y = pd.Categorical(df[actual_target]).codes.astype('int32')
        else:
            y = df[actual_target].astype('int32')
    
    # Prepare feature matrix
    exclude_cols = [
        'id', 'participant_id', 'series_id', 'timestamp',
        target_col, f"{target_col}_encoded", f"{target_col}_binary",
        'label', 'label_encoded', 'label_binary',
        'behavior', 'behavior_encoded', 'behavior_binary'
    ]
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            # Include only numeric features for model training
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
    
    if not feature_cols:
        raise ValueError("No numeric features found for model training")
    
    X = df[feature_cols].copy()
    
    # Final data validation
    # Replace any remaining infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill remaining NaN values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    print(f"ML-ready data prepared:")
    print(f"  Features: {X.shape[1]} columns")
    print(f"  Samples: {X.shape[0]} rows")
    print(f"  Target classes: {y.nunique()}")
    print(f"  Participants: {groups.nunique()}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, groups


def setup_groupkfold_cv(df: pd.DataFrame, n_splits: int = 5, target_col: str = "label") -> Tuple[GroupKFold, List[Tuple[np.ndarray, np.ndarray]]]:
    """Setup GroupKFold cross-validation for CMI sensor data (CLAUDE.md specification)
    
    Mandatory participant-based CV to prevent data leakage
    
    Returns:
        GroupKFold object and list of (train_idx, val_idx) tuples
    """
    if 'participant_id' not in df.columns:
        raise ValueError("participant_id column is required for GroupKFold CV")
    
    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    
    # Prepare data for splitting
    X, y, groups = get_ml_ready_sequences(df, target_col=target_col)
    
    # Generate CV splits
    cv_splits = list(gkf.split(X, y, groups))
    
    # Validate splits (ensure no participant leakage)
    for i, (train_idx, val_idx) in enumerate(cv_splits):
        train_participants = set(groups.iloc[train_idx])
        val_participants = set(groups.iloc[val_idx])
        
        if train_participants & val_participants:
            raise ValueError(f"Fold {i}: Participant leakage detected! {train_participants & val_participants}")
    
    print(f"\n🔄 GroupKFold CV setup complete:")
    print(f"  📊 {n_splits} folds")
    print(f"  👥 {groups.nunique()} participants")
    print(f"  📏 Average fold size: {len(X) // n_splits:,} samples")
    
    # Show participant distribution across folds
    for i, (train_idx, val_idx) in enumerate(cv_splits):
        train_participants = groups.iloc[train_idx].nunique()
        val_participants = groups.iloc[val_idx].nunique()
        print(f"  Fold {i+1}: {train_participants} train participants, {val_participants} val participants")
    
    return gkf, cv_splits


def create_sequence_windows(df: pd.DataFrame, window_size: int = 100, overlap: float = 0.5, target_col: str = "label") -> pd.DataFrame:
    """Create sequence windows for CNN/RNN training (CLAUDE.md specification)
    
    Converts tabular sensor data to sequence format for deep learning
    
    Args:
        df: Input dataframe with sensor features
        window_size: Number of timesteps per window (default: 100 = 2 seconds at 50Hz)
        overlap: Overlap ratio between windows (default: 0.5 = 50% overlap)
        target_col: Target column name
    
    Returns:
        DataFrame with sequence windows
    """
    windowed_data = []
    
    # Group by participant and series
    group_cols = ['participant_id'] if 'participant_id' in df.columns else []
    if 'series_id' in df.columns:
        group_cols.append('series_id')
    
    if not group_cols:
        print("Warning: No grouping columns found. Creating windows from entire dataset.")
        groups = [("all", df)]
    else:
        groups = df.groupby(group_cols)
    
    step_size = int(window_size * (1 - overlap))
    
    # Get feature columns (exclude metadata)
    feature_cols = get_sensor_feature_names(df, target_col)
    
    for group_name, group_df in groups:
        group_df = group_df.reset_index(drop=True)
        
        # Create windows
        for start_idx in range(0, len(group_df) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            
            # Extract window
            window = group_df.iloc[start_idx:end_idx]
            
            # Aggregate features for this window
            window_features = {}
            
            # Add metadata
            if isinstance(group_name, tuple):
                for i, col in enumerate(group_cols):
                    window_features[col] = group_name[i]
            else:
                window_features[group_cols[0]] = group_name
            
            window_features['window_start'] = start_idx
            window_features['window_end'] = end_idx
            
            # Aggregate sensor features
            for col in feature_cols:
                if col in window.columns:
                    # Statistical aggregations
                    window_features[f'{col}_mean'] = window[col].mean()
                    window_features[f'{col}_std'] = window[col].std()
                    window_features[f'{col}_min'] = window[col].min()
                    window_features[f'{col}_max'] = window[col].max()
            
            # Target assignment (majority vote or center value)
            if target_col in window.columns:
                if window[target_col].dtype == 'object':
                    # Most frequent label
                    window_features[target_col] = window[target_col].mode().iloc[0] if len(window[target_col].mode()) > 0 else window[target_col].iloc[0]
                else:
                    # Center value or mean
                    center_idx = len(window) // 2
                    window_features[target_col] = window[target_col].iloc[center_idx]
            
            windowed_data.append(window_features)
    
    windowed_df = pd.DataFrame(windowed_data)
    
    print(f"\n🔄 Sequence windows created:")
    print(f"  📏 Window size: {window_size} timesteps")
    print(f"  📊 Overlap: {overlap*100:.0f}%")
    print(f"  🪟 Total windows: {len(windowed_df):,}")
    print(f"  📈 Features per window: {len([col for col in windowed_df.columns if col not in group_cols + ['window_start', 'window_end', target_col]])}")
    
    return windowed_df


def create_submission_format(predictions: np.ndarray, filename: str = "outputs/submissions/baseline/submission.csv") -> None:
    """Create CMI competition submission file (CLAUDE.md specification)
    
    Creates submission in standardized output directory structure
    """
    import os
    from datetime import datetime
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Load test data
    _, test = load_gold_data()
    
    # Create submission DataFrame
    if 'id' in test.columns:
        test_ids = test['id']
    else:
        print("Warning: 'id' column not found, using index")
        test_ids = test.index
    
    # Handle multi-class predictions for BFRB detection
    if len(np.unique(predictions)) > 2:
        # Multi-class BFRB behavior prediction
        behavior_map = {0: 'no_behavior', 1: 'behavior_1', 2: 'behavior_2', 3: 'behavior_3'}
        pred_labels = [behavior_map.get(int(pred), f'behavior_{int(pred)}') for pred in predictions]
    else:
        # Binary BFRB presence prediction
        pred_labels = ['behavior' if pred > 0.5 else 'no_behavior' for pred in predictions]
    
    submission = pd.DataFrame({
        'id': test_ids,
        'label': pred_labels
    })
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = filename.replace('.csv', f'_{timestamp}.csv')
    
    submission.to_csv(base_name, index=False)
    
    print(f"\n📝 Submission file created: {base_name}")
    print(f"📊 Predictions distribution:")
    pred_counts = submission['label'].value_counts()
    for label, count in pred_counts.items():
        print(f"  {label}: {count:,} ({count/len(submission)*100:.1f}%)")
    
    return base_name


def create_submission_dataframe(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Create submission format DataFrame for CMI competition"""
    # Ensure predictions match DataFrame length
    if len(predictions) != len(df):
        raise ValueError(f"Predictions length {len(predictions)} does not match DataFrame length {len(df)}")
    
    # Get IDs
    test_ids = df['id'] if 'id' in df.columns else range(len(df))
    
    # Handle multi-class or binary predictions
    if len(np.unique(predictions)) > 2:
        # Multi-class BFRB behavior prediction
        behavior_map = {0: 'no_behavior', 1: 'behavior_1', 2: 'behavior_2', 3: 'behavior_3'}
        pred_labels = [behavior_map.get(int(pred), f'behavior_{int(pred)}') for pred in predictions]
    else:
        # Binary BFRB presence prediction
        pred_labels = ['behavior' if pred > 0.5 else 'no_behavior' for pred in predictions]
    
    submission = pd.DataFrame({
        'id': test_ids,
        'label': pred_labels
    })
    
    return submission


def get_sensor_feature_names(df: pd.DataFrame, target_col: str = "label") -> List[str]:
    """Get sensor feature names excluding ID and target columns (CLAUDE.md specification)"""
    exclude_cols = [
        'id', 'participant_id', 'series_id', 'timestamp',
        target_col, f"{target_col}_encoded", f"{target_col}_binary",
        'label', 'label_encoded', 'label_binary',
        'behavior', 'behavior_encoded', 'behavior_binary'
    ]
    
    feature_names = []
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            feature_names.append(col)
    
    return feature_names


def extract_model_arrays(df: pd.DataFrame, target_col: str = "label") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract X, y, feature_names from DataFrame (backward compatibility)
    
    Returns numpy arrays for traditional ML workflows
    """
    # Identify feature columns
    exclude_cols = ['id', 'participant_id', 'series_id', 'timestamp', target_col, f"{target_col}_encoded", f"{target_col}_binary"]
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # X (feature matrix)
    X = df[feature_cols].values
    
    # y (target)
    if f"{target_col}_encoded" in df.columns:
        y = df[f"{target_col}_encoded"].values
    elif target_col in df.columns:
        if df[target_col].dtype == 'object':
            # Encode string targets
            y = pd.Categorical(df[target_col]).codes
        else:
            y = df[target_col].values
    else:
        raise ValueError(f"Target column '{target_col}' or '{target_col}_encoded' not found")
    
    return X, y, feature_cols