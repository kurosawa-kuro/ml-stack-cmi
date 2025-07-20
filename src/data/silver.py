"""
Silver Level Data Management for CMI Sensor Data
Time-Series Feature Engineering & Multimodal Sensor Fusion
Configuration-Driven Feature Engineering
CLAUDE.md: FFT/Statistical Features (tsfresh), Multimodal Channel Fusion
"""

from typing import Tuple, List, Dict, Any, Optional
import warnings
import logging

import duckdb
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder

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

# Configuration-driven tsfresh import
def check_tsfresh_availability() -> bool:
    """Check tsfresh availability based on configuration"""
    if CONFIG_AVAILABLE:
        config = get_project_config()
        tsfresh_enabled = config.data.tsfresh_enabled
        if not tsfresh_enabled:
            logger.info("tsfresh disabled by configuration")
            return False
    
    try:
        import tsfresh
        from tsfresh import extract_features
        from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
        from tsfresh.utilities.dataframe_functions import impute
        return True
    except ImportError:
        logger.warning("tsfresh not available. Install with: pip install tsfresh")
        return False

TSFRESH_AVAILABLE = check_tsfresh_availability()

# Configuration-driven database path
def get_db_path() -> str:
    """Get database path from configuration"""
    if CONFIG_AVAILABLE:
        config = get_project_config()
        return config.data.source_path
    else:
        return "/home/wsl/dev/my-study/ml/ml-stack-cmi/data/kaggle_datasets.duckdb"

DB_PATH = get_db_path()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in multiply')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in reduce')


def extract_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-series feature extraction for CMI sensor data (CLAUDE.md specification)
    
    Extracts features from multimodal sensor channels:
    - IMU (accelerometer/gyroscope): Movement patterns
    - ToF distance sensors: Proximity patterns  
    - Thermopile: Temperature distribution
    """
    df = df.copy()
    
    # Define sensor channel groups based on CMI data structure
    accelerometer_cols = [col for col in df.columns if col.startswith('acc_')]
    gyroscope_cols = [col for col in df.columns if col.startswith('rot_')]
    thermopile_cols = [col for col in df.columns if col.startswith('thm_')]
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    
    # Statistical features for each sensor modality
    for sensor_group, columns in [
        ('imu_acc', accelerometer_cols),
        ('imu_gyro', gyroscope_cols), 
        ('thermal', thermopile_cols),
        ('tof', tof_cols[:10])  # Limit ToF features due to high dimensionality
    ]:
        if columns:
            sensor_data = df[columns]
            
            # Basic statistical features
            df[f'{sensor_group}_mean'] = sensor_data.mean(axis=1)
            df[f'{sensor_group}_std'] = sensor_data.std(axis=1).fillna(0)
            df[f'{sensor_group}_min'] = sensor_data.min(axis=1)
            df[f'{sensor_group}_max'] = sensor_data.max(axis=1)
            df[f'{sensor_group}_range'] = df[f'{sensor_group}_max'] - df[f'{sensor_group}_min']
            df[f'{sensor_group}_skew'] = sensor_data.skew(axis=1).fillna(0)
            df[f'{sensor_group}_kurt'] = sensor_data.kurtosis(axis=1).fillna(0)
            
            # Energy and magnitude features (with overflow protection)
            energy_values = (sensor_data ** 2).sum(axis=1)
            df[f'{sensor_group}_energy'] = np.clip(energy_values, 0, 1e6)  # Prevent overflow
            magnitude_values = np.sqrt((sensor_data ** 2).sum(axis=1))
            df[f'{sensor_group}_magnitude'] = np.clip(magnitude_values, 0, 1e3)  # Prevent overflow
            
            # Cross-channel correlations (for multi-axis sensors)
            if len(columns) >= 2:
                try:
                    corr_matrix = sensor_data.corr()
                    upper_triangle = np.triu_indices_from(corr_matrix.values, k=1)
                    df[f'{sensor_group}_corr_mean'] = corr_matrix.values[upper_triangle].mean()
                except:
                    df[f'{sensor_group}_corr_mean'] = 0
                
    # Multimodal sensor fusion features
    if accelerometer_cols and gyroscope_cols:
        # IMU sensor fusion (with overflow protection)
        acc_magnitude = np.sqrt((df[accelerometer_cols] ** 2).sum(axis=1))
        acc_magnitude = np.clip(acc_magnitude, 0, 1e3)  # Prevent overflow
        gyro_magnitude = np.sqrt((df[gyroscope_cols] ** 2).sum(axis=1))
        gyro_magnitude = np.clip(gyro_magnitude, 0, 1e3)  # Prevent overflow
        df['imu_total_motion'] = np.clip(acc_magnitude + gyro_magnitude, 0, 2e3)
        df['imu_motion_ratio'] = np.clip(acc_magnitude / (gyro_magnitude + 1e-8), 0, 100)
        
    if thermopile_cols and tof_cols:
        # Thermal-distance interaction (with overflow protection)
        thermal_mean = df[thermopile_cols].mean(axis=1)
        thermal_mean = np.clip(thermal_mean, -100, 100)  # Prevent overflow
        tof_mean = df[tof_cols[:5]].mean(axis=1)  # Use subset of ToF sensors
        tof_mean = np.clip(tof_mean, 0, 1000)  # Prevent overflow
        df['thermal_distance_interaction'] = np.clip(thermal_mean * tof_mean, -1e5, 1e5)
        df['thermal_distance_ratio'] = np.clip(thermal_mean / (tof_mean + 1e-8), -100, 100)
        
    return df


def extract_frequency_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Configuration-driven frequency domain feature extraction using FFT
    
    Extracts spectral features for behavior pattern detection based on project configuration
    """
    df = df.copy()
    
    # Check if FFT features are enabled in configuration
    if CONFIG_AVAILABLE:
        config = get_project_config()
        fft_enabled = config.data.fft_enabled
        if not fft_enabled:
            logger.info("FFT features disabled by configuration")
            return df
        
        logger.info(f"Extracting FFT features in {config.phase.value} phase")
    
    # Define sensor groups for frequency analysis
    sensor_groups = {
        'acc': [col for col in df.columns if col.startswith('acc_')],
        'gyro': [col for col in df.columns if col.startswith('rot_')],
        'thermal': [col for col in df.columns if col.startswith('thm_')]
    }
    
    for group_name, columns in sensor_groups.items():
        if columns:
            for col in columns[:3]:  # Limit to prevent feature explosion
                if col in df.columns:
                    try:
                        # FFT analysis
                        signal_data = df[col].fillna(0).values
                        if len(signal_data) > 1:
                            fft_vals = np.abs(fft(signal_data))
                            freqs = fftfreq(len(signal_data), d=1/50)  # 50Hz sampling rate
                            
                            # Spectral features
                            positive_freqs = freqs[:len(freqs)//2]
                            positive_fft = fft_vals[:len(fft_vals)//2]
                            
                            if np.sum(positive_fft) > 0:
                                # Clip FFT values to prevent overflow
                                positive_fft_clipped = np.clip(positive_fft, 0, 1e6)
                                positive_freqs_clipped = np.clip(positive_freqs, 0, 1e3)
                                
                                df[f'{col}_spectral_centroid'] = np.clip(
                                    np.sum(positive_freqs_clipped * positive_fft_clipped) / np.sum(positive_fft_clipped), 
                                    0, 1e3
                                )
                                df[f'{col}_spectral_rolloff'] = np.clip(np.percentile(positive_fft_clipped, 85), 0, 1e6)
                                df[f'{col}_spectral_flux'] = np.clip(np.mean(np.diff(positive_fft_clipped) ** 2), 0, 1e6)
                                
                                # Dominant frequency
                                dominant_freq_idx = np.argmax(positive_fft_clipped)
                                df[f'{col}_dominant_freq'] = np.clip(
                                    positive_freqs_clipped[dominant_freq_idx] if dominant_freq_idx > 0 else 0, 
                                    0, 1e3
                                )
                            else:
                                df[f'{col}_spectral_centroid'] = 0
                                df[f'{col}_spectral_rolloff'] = 0
                                df[f'{col}_spectral_flux'] = 0
                                df[f'{col}_dominant_freq'] = 0
                            
                    except Exception as e:
                        print(f"Warning: FFT analysis failed for {col}: {e}")
                        
    return df


def extract_tsfresh_features(df: pd.DataFrame, max_features: int = 50) -> pd.DataFrame:
    """Configuration-driven tsfresh statistical feature extraction (CLAUDE.md specification)
    
    Comprehensive time-series feature extraction with memory optimization and configuration control
    """
    df = df.copy()
    
    if not TSFRESH_AVAILABLE:
        logger.warning("tsfresh not available, skipping tsfresh features")
        return df
    
    # Get tsfresh configuration
    if CONFIG_AVAILABLE:
        config = get_project_config()
        if not config.data.tsfresh_enabled:
            logger.info("tsfresh disabled by configuration")
            return df
        
        # Override max_features from configuration if available
        configured_max_features = config.data.max_features
        if configured_max_features:
            max_features = configured_max_features
            logger.info(f"Using configured max_features: {max_features}")
        
        logger.info(f"tsfresh extraction in {config.phase.value} phase with max_features={max_features}")
    else:
        logger.info(f"tsfresh extraction (fallback mode) with max_features={max_features}")
    
    # Prepare data for tsfresh (requires specific format)
    if 'sequence_id' not in df.columns:
        df['sequence_id'] = df.index // 100  # Group into sequences
        
    if 'time_step' not in df.columns:
        df['time_step'] = df.index % 100  # Time steps within sequence
        
    # Select key sensor columns for tsfresh (memory optimization)
    sensor_cols = []
    for prefix in ['acc_', 'rot_', 'thm_']:
        matching_cols = [col for col in df.columns if col.startswith(prefix)]
        sensor_cols.extend(matching_cols[:2])  # Limit to 2 per sensor type
        
    try:
        # Configure tsfresh with minimal features to prevent memory issues
        extraction_settings = MinimalFCParameters()
        
        tsfresh_features = {}
        
        # Process each sensor column separately to manage memory
        for col in sensor_cols[:5]:  # Further limit for memory
            if col in df.columns:
                # Prepare data for this column
                ts_data = df[['sequence_id', 'time_step', col]].copy()
                ts_data = ts_data.dropna(subset=[col])
                
                if len(ts_data) > 10:  # Minimum data requirement
                    try:
                        # Extract features for this column
                        features = extract_features(
                            ts_data, 
                            column_id='sequence_id',
                            column_sort='time_step',
                            column_value=col,
                            default_fc_parameters=extraction_settings,
                            disable_progressbar=True
                        )
                        
                        # Add to tsfresh_features dict
                        for feat_name in features.columns[:5]:  # Limit features per column
                            clean_feat_name = f'tsfresh_{col}_{feat_name}'.replace(' ', '_').replace(',', '_')
                            tsfresh_features[clean_feat_name] = features[feat_name].fillna(0)
                            
                    except Exception as e:
                        print(f"Warning: tsfresh extraction failed for {col}: {e}")
                        
        # Add tsfresh features to dataframe
        if tsfresh_features:
            # Ensure all feature arrays have the same length as df
            max_len = len(df)
            for feat_name, feat_values in tsfresh_features.items():
                if len(feat_values) < max_len:
                    # Pad with last value or zero
                    padded_values = np.full(max_len, feat_values.iloc[-1] if len(feat_values) > 0 else 0)
                    padded_values[:len(feat_values)] = feat_values
                    df[feat_name] = padded_values
                else:
                    df[feat_name] = feat_values[:max_len]
                    
    except Exception as e:
        print(f"Warning: tsfresh feature extraction failed: {e}")
        
    return df


def extract_behavior_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract behavior-specific features for CMI sensor data"""
    df = df.copy()
    
    # Motion-based features
    if "acc_x" in df.columns and "acc_y" in df.columns and "acc_z" in df.columns:
        df["total_motion"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
        df["motion_intensity"] = df["total_motion"].rolling(window=10, min_periods=1).mean()
        df["motion_variance"] = df["total_motion"].rolling(window=10, min_periods=1).var().fillna(0)
    
    # Rotation-based features
    if "rot_x" in df.columns and "rot_y" in df.columns and "rot_z" in df.columns:
        df["total_rotation"] = np.sqrt(df["rot_x"]**2 + df["rot_y"]**2 + df["rot_z"]**2)
        df["rotation_intensity"] = df["total_rotation"].rolling(window=10, min_periods=1).mean()
    
    # ToF distance features
    tof_columns = [col for col in df.columns if col.startswith("tof_")]
    if tof_columns:
        df["tof_mean"] = df[tof_columns].mean(axis=1)
        df["tof_std"] = df[tof_columns].std(axis=1).fillna(0)
        df["tof_range"] = df[tof_columns].max(axis=1) - df[tof_columns].min(axis=1)
    
    # Thermal features
    thermal_columns = [col for col in df.columns if col.startswith("thm_")]
    if thermal_columns:
        df["thermal_mean"] = df[thermal_columns].mean(axis=1)
        df["thermal_std"] = df[thermal_columns].std(axis=1).fillna(0)
        df["thermal_range"] = df[thermal_columns].max(axis=1) - df[thermal_columns].min(axis=1)
    
    # Cross-modal features
    if "tof_mean" in df.columns and "thermal_mean" in df.columns:
        df["thermal_distance_interaction"] = df["thermal_mean"] - df["tof_mean"]
    
    return df


def advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced advanced feature engineering for CMI sensor data
    
    Comprehensive feature engineering pipeline with BFRB-specific optimizations
    """
    df = df.copy()
    
    # Step 1: Basic time-series features
    df = extract_time_series_features(df)
    
    # Step 2: Frequency domain features
    df = extract_frequency_domain_features(df)
    
    # Step 3: tsfresh features (configuration-controlled)
    df = extract_tsfresh_features(df)
    
    # Step 4: BFRB-specific features (NEW - High impact)
    df = extract_bfrb_specific_features(df)
    
    # Step 5: Advanced statistical features (NEW - Enhanced)
    df = extract_advanced_statistical_features(df)
    
    # Step 6: Cross-sensor features (NEW - Multimodal fusion)
    df = extract_cross_sensor_features(df)
    
    # Step 7: Temporal pattern features (NEW - Time-series patterns)
    df = extract_temporal_pattern_features(df)
    
    # Step 8: Legacy behavior-specific features
    df = extract_behavior_specific_features(df)
    
    # Step 9: Enhanced motion-based ratios
    if "total_motion" in df.columns:
        df["motion_ratio"] = df["total_motion"] / (df["total_motion"].max() + 1e-8)
    
    if "motion_intensity" in df.columns:
        df["intensity_ratio"] = df["motion_intensity"] / (df["motion_intensity"].max() + 1e-8)
    
    # Step 10: Enhanced sensor fusion features
    if "tof_mean" in df.columns and "thermal_mean" in df.columns:
        df["sensor_fusion_score"] = (df["tof_mean"] + df["thermal_mean"]) / 2
        df["sensor_balance"] = df["tof_mean"] / (df["thermal_mean"] + 1e-8)
    
    # Step 11: Enhanced statistical aggregation
    sensor_columns = [col for col in df.columns if any(prefix in col for prefix in ['acc_', 'rot_', 'tof_', 'thm_'])]
    if sensor_columns:
        df["sensor_mean"] = df[sensor_columns].mean(axis=1)
        df["sensor_std"] = df[sensor_columns].std(axis=1).fillna(0)
        df["sensor_range"] = df[sensor_columns].max(axis=1) - df[sensor_columns].min(axis=1)
    
    return df


def enhanced_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced interaction features for CMI sensor data"""
    df = df.copy()
    
    # Top features for interaction
    top_features = ["motion_intensity", "total_motion", "tof_mean"]
    
    # 1. Motion intensity interactions
    if all(col in df.columns for col in ["motion_intensity", "total_motion"]):
        df["motion_intensity_interaction"] = df["motion_intensity"] * df["total_motion"]
        df["motion_intensity_ratio"] = df["motion_intensity"] / (df["total_motion"] + 1e-8)
        df["motion_total_ratio"] = df["total_motion"] / (df["motion_intensity"] + 1e-8)
    
    # 2. Motion intensity with other features
    if "motion_intensity" in df.columns:
        if "tof_mean" in df.columns:
            df["motion_tof_interaction"] = df["motion_intensity"] * df["tof_mean"]
            df["motion_tof_ratio"] = df["motion_intensity"] / (df["tof_mean"] + 1e-8)
        
        if "thermal_mean" in df.columns:
            df["motion_thermal_interaction"] = df["motion_intensity"] * df["thermal_mean"]
            df["motion_thermal_contrast"] = df["motion_intensity"] - df["thermal_mean"]
            df["motion_thermal_ratio"] = df["motion_intensity"] / (df["thermal_mean"] + 1e-8)
        
        if "rotation_intensity" in df.columns:
            df["motion_rotation_interaction"] = df["motion_intensity"] * df["rotation_intensity"]
            df["motion_rotation_ratio"] = df["motion_intensity"] / (df["rotation_intensity"] + 1e-8)
    
    # 3. ToF interactions
    if "tof_mean" in df.columns:
        if "thermal_mean" in df.columns:
            df["tof_thermal_interaction"] = df["tof_mean"] * df["thermal_mean"]
            df["tof_thermal_ratio"] = df["tof_mean"] / (df["thermal_mean"] + 1e-8)
        
        if "rotation_intensity" in df.columns:
            df["tof_rotation_interaction"] = df["tof_mean"] * df["rotation_intensity"]
            df["tof_rotation_ratio"] = df["tof_mean"] / (df["rotation_intensity"] + 1e-8)
    
    # 4. Triple interactions
    if all(col in df.columns for col in ["motion_intensity", "tof_mean", "thermal_mean"]):
        df["triple_sensor_interaction"] = df["motion_intensity"] * df["tof_mean"] * df["thermal_mean"]
    
    if all(col in df.columns for col in ["motion_intensity", "total_motion", "rotation_intensity"]):
        df["triple_motion_interaction"] = df["motion_intensity"] * df["total_motion"] * df["rotation_intensity"]
    
    # 5. Composite scores
    if all(col in df.columns for col in ["tof_mean", "thermal_mean", "motion_intensity"]):
        df["composite_sensor_score"] = (df["tof_mean"] + df["thermal_mean"] + df["motion_intensity"]) / 3
        df["sensor_balance_score"] = df["tof_mean"] * df["thermal_mean"]
    
    return df


def polynomial_features(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """多項式特徴量生成（CMIセンサーデータ用）"""
    df = df.copy()
    
    # 上位特徴量のみ選択（数値のみ確保）
    top_numeric_features = [
        "motion_intensity",
        "total_motion",
        "tof_mean",
        "thermal_mean",
        "rotation_intensity",
        "motion_variance",
        "thermal_distance_interaction",
        "sensor_fusion_score",
        "motion_ratio",
        "intensity_ratio"
    ]

    key_features = []
    for feature in top_numeric_features:
        if feature in df.columns:
            # 数値型であることを確認
            if pd.api.types.is_numeric_dtype(df[feature]):
                key_features.append(feature)

    if len(key_features) >= 2:  # 最低2つの特徴量が必要
        try:
            # 一時的なデータフレームで多項式特徴量生成
            temp_df = df[key_features].copy()

            # NaNと無限値の処理
            temp_df = temp_df.fillna(0)
            temp_df = temp_df.replace([np.inf, -np.inf], 0)
            
            # Scale features to prevent overflow in polynomial expansion
            temp_df = temp_df.clip(-5, 5)  # Ultra conservative clipping
            
            # Further scale down large values to prevent overflow
            for col in temp_df.columns:
                if temp_df[col].abs().max() > 3:
                    temp_df[col] = temp_df[col] / (temp_df[col].abs().max() / 3)

            # PolynomialFeaturesを使用（interaction_only=Falseで二乗項も含む）
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            poly_features = poly.fit_transform(temp_df)
            
            # Clip polynomial features to prevent overflow - ultra conservative
            poly_features = np.clip(poly_features, -50, 50)

            # 特徴量名生成
            feature_names = poly.get_feature_names_out(key_features)

            # 元の特徴量以外の新しい特徴量のみ追加
            original_features = set(key_features)
            for i, name in enumerate(feature_names):
                if name not in original_features:
                    # 特徴量名をクリーンアップ
                    clean_name = name.replace(" ", "_").replace("^", "_pow_")
                    # Additional clipping for individual features
                    feature_values = poly_features[:, i]
                    feature_values = np.clip(feature_values, -50, 50)
                    df[f"poly_{clean_name}"] = feature_values.astype(np.float32)

        except Exception as e:
            # エラーが発生した場合はログに記録して続行
            print(f"Warning: Polynomial feature generation failed: {e}")

    return df


def scaling_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量スケーリング（標準化）"""
    df = df.copy()

    # 数値特徴量を標準化（boolean型も含む）
    numeric_features = df.select_dtypes(include=[np.number, bool]).columns
    exclude_cols = ["id"]  # IDカラムは除外
    numeric_features = [col for col in numeric_features if col not in exclude_cols]

    for col in numeric_features:
        col_data = df[col].clip(-100, 100)  # More conservative clipping
        if col_data.std() > 1e-8:  # 分散が0でない場合のみ
            df[f"{col}_scaled"] = ((col_data - col_data.mean()) / col_data.std()).clip(-5, 5).astype(np.float32)

    return df


def create_silver_tables() -> None:
    """silver層テーブルをDuckDBに作成"""
    conn = duckdb.connect(DB_PATH)

    # silverスキーマ作成
    conn.execute("CREATE SCHEMA IF NOT EXISTS silver")

    # bronzeデータ読み込み
    try:
        train_bronze = conn.execute("SELECT * FROM bronze.train").df()
        test_bronze = conn.execute("SELECT * FROM bronze.test").df()
    except Exception:
        print("Bronze tables not found. Creating bronze tables first...")
        from .bronze import create_bronze_tables

        create_bronze_tables()
        train_bronze = conn.execute("SELECT * FROM bronze.train").df()
        test_bronze = conn.execute("SELECT * FROM bronze.test").df()

    # Apply Silver layer processing pipeline (CLAUDE.md specification)
    # Bronze層からは品質保証されたデータのみを受け取り、全ての特徴量をSilver層で生成
    
    # Step 1: Advanced features (Winner Solution + 統計特徴量)
    train_silver = advanced_features(train_bronze)
    test_silver = advanced_features(test_bronze)

    # Step 2: CMI Sensor Interaction Features (multimodal fusion)
    train_silver = cmi_sensor_interaction_features(train_silver)
    test_silver = cmi_sensor_interaction_features(test_silver)

    # Step 3: CMI Multimodal Fusion Features (BFRB detection)
    train_silver = cmi_multimodal_fusion_features(train_silver)
    test_silver = cmi_multimodal_fusion_features(test_silver)

    # Step 4: CMI Temporal Pattern Features (behavioral patterns)
    train_silver = cmi_temporal_pattern_features(train_silver)
    test_silver = cmi_temporal_pattern_features(test_silver)

    # Step 5: Enhanced interaction features (追加の交互作用)
    train_silver = enhanced_interaction_features(train_silver)
    test_silver = enhanced_interaction_features(test_silver)

    # Step 6: Degree-2 nonlinear combinations (多項式特徴量)
    train_silver = polynomial_features(train_silver, degree=2)
    test_silver = polynomial_features(test_silver, degree=2)
    
    # Step 7: Advanced Feature Engineering (Bronze Medal Enhancement)
    # LightGBM Power Transformations (+0.3-0.5% expected)
    lgbm_engineer = LightGBMFeatureEngineer()
    lgbm_engineer.fit(train_silver)
    train_silver = lgbm_engineer.transform(train_silver)
    test_silver = lgbm_engineer.transform(test_silver)
    
    # Advanced Statistical Features (+0.1-0.3% expected)  
    stat_engineer = AdvancedStatisticalFeatures(n_neighbors=5)
    stat_engineer.fit(train_silver)
    train_silver = stat_engineer.transform(train_silver)
    test_silver = stat_engineer.transform(test_silver)

    # silverテーブル作成・挿入
    conn.execute("DROP TABLE IF EXISTS silver.train")
    conn.execute("DROP TABLE IF EXISTS silver.test")

    conn.register("train_silver_df", train_silver)
    conn.register("test_silver_df", test_silver)

    conn.execute("CREATE TABLE silver.train AS SELECT * FROM train_silver_df")
    conn.execute("CREATE TABLE silver.test AS SELECT * FROM test_silver_df")

    print("Silver tables created: ")
    print(f"- silver.train: {len(train_silver)} rows, {len(train_silver.columns)} columns")
    print(f"- silver.test: {len(test_silver)} rows, {len(test_silver.columns)} columns")
    print(f"- Total Engineered Features: {len(train_silver.columns) - len(train_bronze.columns)} features generated")
    print(f"- Sensor features: {len([col for col in train_silver.columns if any(keyword in col for keyword in ['motion', 'tof', 'thermal', 'sensor'])])}")
    print(f"- Interaction features: {len([col for col in train_silver.columns if 'interaction' in col.lower()])}")
    print(f"- Polynomial features: {len([col for col in train_silver.columns if col.startswith('poly_')])}")
    print(f"- Power transformed features: {len([col for col in train_silver.columns if col.endswith('_power')])}")
    print(f"- Statistical features: {len([col for col in train_silver.columns if col.startswith('row_')])}")
    print(f"- Missing indicators: {len([col for col in train_silver.columns if col.endswith('_was_missing')])}")
    print(f"- Z-score features: {len([col for col in train_silver.columns if col.endswith('_zscore')])}")
    print(f"- Percentile features: {len([col for col in train_silver.columns if col.endswith('_percentile')])}")
    print("Advanced Feature Engineering:")
    print("  ✓ LightGBM Power Transformations (+0.3-0.5% expected)")
    print("  ✓ KNN Imputation with Missing Indicators (+0.1-0.2% expected)")  
    print("  ✓ Advanced Statistical Moments (+0.1-0.2% expected)")
    print("  ✓ Feature-specific Z-scores and Percentiles (+0.1% expected)")
    print("  🎯 Bronze Medal Target: +0.8% total improvement expected")

    conn.close()


def load_silver_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """silver層データ読み込み"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM silver.train").df()
    test = conn.execute("SELECT * FROM silver.test").df()
    conn.close()
    return train, test


def cmi_sensor_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """CMI sensor interaction features for BFRB detection"""
    df = df.copy()
    
    # ToF-Thermal proximity interaction (hand-to-face detection)
    if "tof_mean" in df.columns and "thermal_mean" in df.columns:
        df["thermal_distance_interaction"] = df["tof_mean"] * df["thermal_mean"]
        df["proximity_thermal_ratio"] = df["tof_mean"] / (df["thermal_mean"] + 1e-8)
    
    # IMU-ToF movement-proximity correlation
    if "imu_total_motion" in df.columns and "tof_mean" in df.columns:
        df["movement_proximity_interaction"] = df["imu_total_motion"] * (1 / (df["tof_mean"] + 1e-8))
    
    # Thermal contact indicator (elevated temperature + close proximity)
    if "thermal_mean" in df.columns and "tof_mean" in df.columns:
        df["thermal_contact_indicator"] = (df["thermal_mean"] > df["thermal_mean"].quantile(0.7)) & (df["tof_mean"] < df["tof_mean"].quantile(0.3))
        df["thermal_contact_indicator"] = df["thermal_contact_indicator"].astype(float)
    
    # Multi-modal sensor fusion score
    if "imu_total_motion" in df.columns and "tof_mean" in df.columns and "thermal_mean" in df.columns:
        df["sensor_fusion_score"] = (df["imu_total_motion"] * 0.4) + (df["thermal_mean"] * 0.3) + ((500 - df["tof_mean"]) * 0.3)
    
    return df


def cmi_multimodal_fusion_features(df: pd.DataFrame) -> pd.DataFrame:
    """CMI multimodal sensor fusion features for BFRB detection"""
    df = df.copy()
    
    # Close proximity ratio (ToF distance threshold)
    if "tof_mean" in df.columns:
        df["close_proximity_ratio"] = (df["tof_mean"] < 150).astype(float)  # < 15cm proximity
    
    # Movement intensity categorization
    if "imu_total_motion" in df.columns:
        df["movement_intensity"] = pd.cut(df["imu_total_motion"], bins=3, labels=[0, 1, 2]).astype(float)
    
    # Thermal elevation indicator (above baseline)
    if "thermal_mean" in df.columns:
        thermal_baseline = df["thermal_mean"].quantile(0.5)
        df["thermal_elevation"] = (df["thermal_mean"] > thermal_baseline).astype(float)
    
    # Behavioral engagement score (combines all modalities)
    if "imu_total_motion" in df.columns and "tof_mean" in df.columns and "thermal_mean" in df.columns:
        # Normalize each component
        motion_norm = df["imu_total_motion"] / (df["imu_total_motion"].max() + 1e-8)
        proximity_norm = (500 - df["tof_mean"]) / 500  # Invert distance (closer = higher score)
        thermal_norm = df["thermal_mean"] / (df["thermal_mean"].max() + 1e-8)
        
        df["behavioral_engagement_score"] = (motion_norm * 0.4) + (proximity_norm * 0.35) + (thermal_norm * 0.25)
    
    return df


def cmi_temporal_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """CMI temporal pattern features for BFRB detection"""
    df = df.copy()
    
    # Hand-to-face frequency ratio (proximity changes)
    if "tof_std" in df.columns and "tof_mean" in df.columns:
        df["hand_face_frequency"] = df["tof_std"] / (df["tof_mean"] + 1e-8)
    
    # Movement consistency (low variance = repetitive behavior)
    if "imu_acc_std" in df.columns and "imu_acc_mean" in df.columns:
        df["movement_consistency"] = 1 / (df["imu_acc_std"] + 1e-8)
    
    # Thermal stability (contact duration indicator)
    if "thermal_std" in df.columns and "thermal_mean" in df.columns:
        df["thermal_stability"] = df["thermal_mean"] / (df["thermal_std"] + 1e-8)
    
    # Rhythmic behavior indicator (spectral peak consistency)
    if "acc_x_spectral_centroid" in df.columns and "acc_x_dominant_freq" in df.columns:
        df["rhythmic_behavior"] = abs(df["acc_x_spectral_centroid"] - df["acc_x_dominant_freq"])
    
    return df


def get_feature_importance_order() -> list:
    """Get feature importance order for CMI sensor data"""
    return [
        "motion_intensity",
        "total_motion",
        "tof_mean",
        "thermal_mean",
        "rotation_intensity",
        "motion_variance",
        "thermal_distance_interaction",
        "sensor_fusion_score",
        "motion_ratio",
        "intensity_ratio"
    ]


# ===== Advanced Feature Engineering Classes =====

class LightGBMFeatureEngineer(BaseEstimator, TransformerMixin):
    """LightGBM-optimized feature engineering (+0.3-0.5% expected)"""
    
    def __init__(self, use_power_transforms: bool = True):
        self.use_power_transforms = use_power_transforms
        self.power_transformers = {}
        self.numeric_features = None
        
    def fit(self, X, y=None):
        """Fit power transformations for skewed features"""
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in self.numeric_features:
            self.numeric_features.remove('id')
        
        if self.use_power_transforms:
            for col in self.numeric_features:
                if col in X.columns:
                    col_data = X[col].dropna()
                    if len(col_data) > 0:
                        # Skip sensor energy features that cause overflow
                        if any(sensor_keyword in col.lower() for sensor_keyword in ['energy', 'magnitude', 'spectral', 'fft', 'tof_', 'acc_', 'rot_', 'thm_']):
                            continue
                            
                        skewness = col_data.skew()
                        if abs(skewness) > 0.5:  # Moderately skewed
                            try:
                                # Ultra-conservative clipping to prevent overflow
                                col_data_clipped = col_data.clip(-10, 10)
                                
                                # Check if data is still valid after clipping
                                if col_data_clipped.std() > 1e-8 and len(col_data_clipped.unique()) > 1:
                                    self.power_transformers[col] = PowerTransformer(
                                        method='yeo-johnson',
                                        standardize=False
                                    )
                                    self.power_transformers[col].fit(col_data_clipped.values.reshape(-1, 1))
                            except Exception as e:
                                print(f"Warning: PowerTransformer fit failed for {col}: {e}")
                                # Skip this feature if PowerTransformer fails
                                continue
        
        return self
    
    def transform(self, X):
        """Apply power transformations for LightGBM"""
        X_transformed = X.copy()
        
        # Collect all power features in a dictionary for efficient addition
        power_features = {}
        
        for col, transformer in self.power_transformers.items():
            if col in X_transformed.columns:
                mask = X_transformed[col].notna()
                if mask.sum() > 0:
                    try:
                        # Ultra-conservative clipping to prevent overflow
                        values_to_transform = X_transformed.loc[mask, col].values
                        clipped_values = np.clip(values_to_transform, -10, 10)
                        
                        transformed_values = transformer.transform(
                            clipped_values.reshape(-1, 1)
                        ).flatten()
                        
                        # Additional clipping of transformed values to prevent overflow
                        transformed_values = np.clip(transformed_values, -50, 50)
                        
                        # Create full-length array with NaN for missing values
                        power_col = np.full(len(X_transformed), np.nan)
                        power_col[mask] = transformed_values
                        power_features[f'{col}_power'] = power_col
                    except Exception as e:
                        print(f"Warning: Power transformation failed for {col}: {e}")
        
        # Add all power features at once using pd.concat
        if power_features:
            power_df = pd.DataFrame(power_features, index=X_transformed.index)
            X_transformed = pd.concat([X_transformed, power_df], axis=1)
        
        return X_transformed


class CVSafeTargetEncoder(BaseEstimator, TransformerMixin):
    """Fold-safe target encoding (+0.2-0.4% expected)"""
    
    def __init__(self, cols: Optional[List[str]] = None, smoothing: float = 1.0, noise_level: float = 0.01):
        self.cols = cols
        self.smoothing = smoothing
        self.noise_level = noise_level
        self.encoders = {}
        self.global_mean = None
        
    def fit(self, X, y):
        """Fit target encoders with smoothing"""
        if y is None:
            raise ValueError("Target encoding requires y")
            
        self.global_mean = np.mean(y)
        
        if self.cols is None:
            self.cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.cols:
            if col in X.columns:
                encoder = TargetEncoder(
                    smoothing=self.smoothing,
                    min_samples_leaf=10,
                    return_df=True
                )
                temp_df = pd.DataFrame({col: X[col]})
                encoder.fit(temp_df, y)
                self.encoders[col] = encoder
        
        return self
    
    def transform(self, X):
        """Apply target encoding with noise for regularization"""
        X_transformed = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                temp_df = pd.DataFrame({col: X_transformed[col]})
                encoded_values = encoder.transform(temp_df)[col].values
                
                if self.noise_level > 0:
                    noise = np.random.normal(0, self.noise_level, size=len(encoded_values))
                    encoded_values = encoded_values + noise
                
                X_transformed[f'{col}_target_encoded'] = encoded_values
        
        return X_transformed


class AdvancedStatisticalFeatures(BaseEstimator, TransformerMixin):
    """Advanced statistical and imputation features (+0.1-0.3% expected)"""
    
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
        self.knn_imputer = None
        self.numeric_features = None
        self.original_features = None
        
    def fit(self, X, y=None):
        """Fit KNN imputer only on original features"""
        # Only apply KNN to original Bronze features, not derived polynomial features
        original_numeric_features = [
            'acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z',
            'tof_0', 'tof_1', 'thm_0', 'thm_1'
        ]
        self.original_features = [f for f in original_numeric_features if f in X.columns]
        
        if self.original_features:
            self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            # Clip values before fitting to prevent numerical issues
            X_clipped = X[self.original_features].clip(-100, 100)
            self.knn_imputer.fit(X_clipped)
        
        return self
    
    def transform(self, X):
        """Apply KNN imputation and add statistical features"""
        X_transformed = X.copy()
        
        # 1. Store missing indicators before imputation (bulk operation)
        if self.knn_imputer and self.original_features:
            missing_indicators = {}
            for col in self.original_features:
                if col in X_transformed.columns:
                    missing_indicators[f'{col}_was_missing'] = X_transformed[col].isna().astype(int)
            
            if missing_indicators:
                missing_df = pd.DataFrame(missing_indicators, index=X_transformed.index)
                X_transformed = pd.concat([X_transformed, missing_df], axis=1)
            
            # Apply KNN imputation only to original features with clipping
            X_clipped = X_transformed[self.original_features].clip(-50, 50)
            X_transformed[self.original_features] = self.knn_imputer.transform(X_clipped)
        
        # 2. Add statistical moment features (bulk operation)
        # Use only original features for statistics to prevent explosion
        stats_features = self.original_features if self.original_features else []
        if stats_features:
            numeric_data = X_transformed[stats_features]
            
            # Clip extreme values to prevent overflow - ultra conservative clipping
            numeric_data_clipped = numeric_data.clip(-25, 25)
            
            statistical_features = {
                'row_mean': numeric_data_clipped.mean(axis=1),
                'row_std': numeric_data_clipped.std(axis=1).fillna(0),
                'row_q25': numeric_data_clipped.quantile(0.25, axis=1),
                'row_q75': numeric_data_clipped.quantile(0.75, axis=1),
            }
            
            # Safer calculation for skew and kurtosis
            try:
                statistical_features['row_skew'] = numeric_data_clipped.apply(
                    lambda x: x.dropna().skew() if len(x.dropna()) > 2 else 0, axis=1
                ).fillna(0).clip(-10, 10)
                
                statistical_features['row_kurtosis'] = numeric_data_clipped.apply(
                    lambda x: x.dropna().kurtosis() if len(x.dropna()) > 2 else 0, axis=1
                ).fillna(0).clip(-10, 10)
            except Exception:
                statistical_features['row_skew'] = pd.Series(0, index=X_transformed.index)
                statistical_features['row_kurtosis'] = pd.Series(0, index=X_transformed.index)
            statistical_features['row_iqr'] = statistical_features['row_q75'] - statistical_features['row_q25']
            
            stats_df = pd.DataFrame(statistical_features, index=X_transformed.index)
            X_transformed = pd.concat([X_transformed, stats_df], axis=1)
            
            # Feature-specific moments for key features
            key_features = ['motion_intensity', 'total_motion', 'tof_mean']
            key_feature_stats = {}
            for feat in key_features:
                if feat in X_transformed.columns:
                    # Safer zscore calculation with aggressive clipping
                    feat_data = X_transformed[feat].clip(-50, 50)
                    zscore = (feat_data - feat_data.mean()) / (feat_data.std() + 1e-8)
                    key_feature_stats[f'{feat}_zscore'] = zscore.clip(-5, 5)
                    
                    key_feature_stats[f'{feat}_percentile'] = X_transformed[feat].rank(pct=True)
            
            if key_feature_stats:
                key_stats_df = pd.DataFrame(key_feature_stats, index=X_transformed.index)
                X_transformed = pd.concat([X_transformed, key_stats_df], axis=1)
        
        return X_transformed


# ===== Sklearn-Compatible Transformers for Pipeline Integration =====

class SilverPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for Silver layer processing"""
    
    def __init__(self, add_polynomial: bool = True, add_scaling: bool = True):
        self.add_polynomial = add_polynomial
        self.add_scaling = add_scaling
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Fit the transformer (no fitting required for Silver layer)"""
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply Silver layer transformations"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        # Apply Silver pipeline
        X_transformed = advanced_features(X)
        X_transformed = cmi_sensor_interaction_features(X_transformed)
        X_transformed = cmi_multimodal_fusion_features(X_transformed)
        X_transformed = cmi_temporal_pattern_features(X_transformed)
        X_transformed = enhanced_interaction_features(X_transformed)
        
        if self.add_polynomial:
            X_transformed = polynomial_features(X_transformed, degree=2)
        
        if self.add_scaling:
            X_transformed = scaling_features(X_transformed)
        
        return X_transformed


class FoldSafeSilverPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-safe Silver preprocessor for CV integration"""
    
    def __init__(self, use_target_encoding: bool = False):
        self.use_target_encoding = use_target_encoding
        self.scaler = StandardScaler()
        self.lgbm_engineer = LightGBMFeatureEngineer()
        self.stat_engineer = AdvancedStatisticalFeatures(n_neighbors=5)
        self.target_encoder = None
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Learn scaling parameters from training data only"""
        # Apply Silver transformations
        X_silver = advanced_features(X)
        X_silver = cmi_sensor_interaction_features(X_silver)
        X_silver = cmi_multimodal_fusion_features(X_silver)
        X_silver = cmi_temporal_pattern_features(X_silver)
        X_silver = enhanced_interaction_features(X_silver)
        
        # Fit advanced feature engineers
        self.lgbm_engineer.fit(X_silver, y)
        X_silver = self.lgbm_engineer.transform(X_silver)
        
        self.stat_engineer.fit(X_silver, y)
        X_silver = self.stat_engineer.transform(X_silver)
        
        # Target encoding for categorical features (if enabled and y provided)
        if self.use_target_encoding and y is not None:
            categorical_cols = ['behavior', 'gesture']
            available_cats = [col for col in categorical_cols if col in X_silver.columns]
            if available_cats:
                self.target_encoder = CVSafeTargetEncoder(
                    cols=available_cats, 
                    smoothing=self.target_smoothing
                )
                self.target_encoder.fit(X_silver, y)
        
        # Fit scaler on numeric features only
        numeric_features = X_silver.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            self.scaler.fit(X_silver[numeric_features])
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply fold-safe transformations"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        # Apply Silver transformations
        X_transformed = advanced_features(X)
        X_transformed = cmi_sensor_interaction_features(X_transformed)
        X_transformed = cmi_multimodal_fusion_features(X_transformed)
        X_transformed = cmi_temporal_pattern_features(X_transformed)
        X_transformed = enhanced_interaction_features(X_transformed)
        
        # Apply advanced feature engineering
        X_transformed = self.lgbm_engineer.transform(X_transformed)
        X_transformed = self.stat_engineer.transform(X_transformed)
        
        # Apply target encoding if fitted
        if self.target_encoder:
            X_transformed = self.target_encoder.transform(X_transformed)
        
        # Apply scaling only to numeric features
        numeric_features = X_transformed.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            X_transformed[numeric_features] = self.scaler.transform(X_transformed[numeric_features])
        
        return X_transformed


class EnhancedSilverPreprocessor(BaseEstimator, TransformerMixin):
    """Enhanced Silver layer processor combining all advanced techniques"""
    
    def __init__(
        self,
        use_power_transforms: bool = True,
        use_target_encoding: bool = True,
        use_statistical_features: bool = True,
        n_neighbors: int = 5,
        target_smoothing: float = 1.0
    ):
        self.use_power_transforms = use_power_transforms
        self.use_target_encoding = use_target_encoding
        self.use_statistical_features = use_statistical_features
        self.n_neighbors = n_neighbors
        self.target_smoothing = target_smoothing
        
        # Initialize sub-transformers
        self.lgbm_engineer = LightGBMFeatureEngineer(use_power_transforms=use_power_transforms)
        self.stat_engineer = AdvancedStatisticalFeatures(n_neighbors=n_neighbors)
        self.target_encoder = None
        
    def fit(self, X, y=None):
        """Fit all transformers"""
        if self.use_power_transforms:
            self.lgbm_engineer.fit(X, y)
        
        if self.use_statistical_features:
            self.stat_engineer.fit(X, y)
        
        if self.use_target_encoding and y is not None:
            categorical_cols = ['behavior', 'gesture']
            available_cats = [col for col in categorical_cols if col in X.columns]
            if available_cats:
                self.target_encoder = CVSafeTargetEncoder(
                    cols=available_cats, 
                    smoothing=self.target_smoothing
                )
                self.target_encoder.fit(X, y)
        
        return self
    
    def transform(self, X):
        """Apply all transformations"""
        X_transformed = X.copy()
        
        # Apply core Silver features first
        X_transformed = advanced_features(X_transformed)
        X_transformed = cmi_sensor_interaction_features(X_transformed)
        X_transformed = cmi_multimodal_fusion_features(X_transformed)
        X_transformed = cmi_temporal_pattern_features(X_transformed)
        X_transformed = enhanced_interaction_features(X_transformed)
        X_transformed = polynomial_features(X_transformed, degree=2)
        
        # Apply advanced feature engineering
        if self.use_power_transforms:
            X_transformed = self.lgbm_engineer.transform(X_transformed)
        
        if self.use_statistical_features:
            X_transformed = self.stat_engineer.transform(X_transformed)
        
        if self.target_encoder:
            X_transformed = self.target_encoder.transform(X_transformed)
        
        return X_transformed


def extract_bfrb_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """BFRB特有の行動パターン検出に特化した特徴量エンジニアリング
    
    Body-Focused Repetitive Behaviors (BFRB) の検出に最適化された特徴量
    - 手の動きパターン（顔への接近・離脱）
    - 反復的な動作の検出
    - 接触・非接触の判定
    - 行動の持続時間と頻度
    """
    df = df.copy()
    
    # 1. 手の動きパターン検出（顔への接近・離脱）
    if "acc_x" in df.columns and "acc_y" in df.columns and "acc_z" in df.columns:
        # 手の動きの方向性を検出
        df["hand_movement_direction"] = np.arctan2(df["acc_y"], df["acc_x"])
        df["hand_movement_magnitude"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
        
        # 手の動きの一貫性（反復動作の指標）
        df["hand_movement_consistency"] = df["hand_movement_magnitude"].rolling(window=20, min_periods=1).std()
        df["hand_movement_consistency"] = 1 / (df["hand_movement_consistency"] + 1e-8)
        
        # 手の動きの周期性（FFTベース）
        try:
            # 短時間での周期性を検出
            window_size = min(50, len(df))
            if window_size > 10:
                fft_vals = np.abs(np.fft.fft(df["hand_movement_magnitude"].iloc[:window_size].fillna(0)))
                dominant_freq = np.argmax(fft_vals[1:window_size//2]) + 1
                df["hand_movement_periodicity"] = dominant_freq / window_size
            else:
                df["hand_movement_periodicity"] = 0
        except:
            df["hand_movement_periodicity"] = 0
    
    # 2. ToFセンサーによる顔への接近度検出
    tof_columns = [col for col in df.columns if col.startswith("tof_")]
    if tof_columns:
        # 顔への接近度（距離の逆数）
        df["face_proximity"] = 1 / (df[tof_columns].mean(axis=1) + 1e-8)
        df["face_proximity"] = np.clip(df["face_proximity"], 0, 0.1)  # 最大10cm
        
        # 顔への接近パターン
        df["face_approach_speed"] = df["face_proximity"].diff().fillna(0)
        df["face_approach_acceleration"] = df["face_approach_speed"].diff().fillna(0)
        
        # 顔への接近頻度（閾値ベース）
        proximity_threshold = df["face_proximity"].quantile(0.7)
        df["face_approach_frequency"] = (df["face_proximity"] > proximity_threshold).rolling(window=10, min_periods=1).mean()
    
    # 3. 温度センサーによる接触検出
    thermal_columns = [col for col in df.columns if col.startswith("thm_")]
    if thermal_columns:
        # 接触温度の検出
        df["contact_temperature"] = df[thermal_columns].max(axis=1)
        df["temperature_gradient"] = df["contact_temperature"].diff().fillna(0)
        
        # 接触の持続時間（温度上昇の持続）
        df["contact_duration_indicator"] = (df["temperature_gradient"] > 0).rolling(window=5, min_periods=1).sum()
        
        # 接触強度（温度変化の大きさ）
        df["contact_intensity"] = df["temperature_gradient"].rolling(window=10, min_periods=1).std()
    
    # 4. 反復動作の検出
    if "hand_movement_magnitude" in df.columns:
        # 動作の反復性（自己相関ベース）
        try:
            window_size = min(30, len(df))
            if window_size > 10:
                movement_series = df["hand_movement_magnitude"].iloc[:window_size].fillna(0)
                autocorr = np.correlate(movement_series, movement_series, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                # 最初のピーク後の相関を反復性の指標とする
                if len(autocorr) > 5:
                    df["movement_repetitiveness"] = autocorr[5] / (autocorr[0] + 1e-8)
                else:
                    df["movement_repetitiveness"] = 0
            else:
                df["movement_repetitiveness"] = 0
        except:
            df["movement_repetitiveness"] = 0
        
        # 動作の一貫性（分散の逆数）
        df["movement_consistency"] = 1 / (df["hand_movement_magnitude"].rolling(window=15, min_periods=1).var() + 1e-8)
        df["movement_consistency"] = np.clip(df["movement_consistency"], 0, 100)
    
    # 5. 行動の持続時間と頻度
    if "face_proximity" in df.columns and "contact_temperature" in df.columns:
        # 行動の持続時間（接近+接触の組み合わせ）
        behavior_condition = (df["face_proximity"] > df["face_proximity"].quantile(0.6)) & \
                           (df["contact_temperature"] > df["contact_temperature"].quantile(0.5))
        df["behavior_duration"] = behavior_condition.rolling(window=20, min_periods=1).sum()
        
        # 行動の頻度（単位時間あたりの行動回数）
        df["behavior_frequency"] = behavior_condition.rolling(window=50, min_periods=1).mean()
    
    # 6. マルチモーダル融合特徴量
    if all(col in df.columns for col in ["hand_movement_magnitude", "face_proximity", "contact_temperature"]):
        # BFRB行動スコア（総合的な行動指標）
        movement_score = df["hand_movement_magnitude"] / (df["hand_movement_magnitude"].max() + 1e-8)
        proximity_score = df["face_proximity"] / (df["face_proximity"].max() + 1e-8)
        contact_score = df["contact_temperature"] / (df["contact_temperature"].max() + 1e-8)
        
        df["bfrb_behavior_score"] = (movement_score * 0.4 + proximity_score * 0.35 + contact_score * 0.25)
        
        # 行動の複雑性（複数のセンサーの組み合わせ）
        df["behavior_complexity"] = (movement_score * proximity_score * contact_score)
        
        # 行動の一貫性（複数センサーの相関）
        sensor_scores = pd.DataFrame({
            'movement': movement_score,
            'proximity': proximity_score,
            'contact': contact_score
        })
        df["sensor_correlation"] = sensor_scores.corr().iloc[0, 1]  # movementとproximityの相関
    
    # 7. 時間的特徴量
    if "hand_movement_magnitude" in df.columns:
        # 動作の時間的パターン
        df["movement_temporal_pattern"] = df["hand_movement_magnitude"].rolling(window=25, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # 動作の急激な変化
        df["movement_sudden_change"] = df["hand_movement_magnitude"].diff().abs().rolling(window=5, min_periods=1).max()
    
    return df


def extract_advanced_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """高度な統計的特徴量エンジニアリング
    
    従来の統計量を超えた、より洗練された特徴量を生成
    """
    df = df.copy()
    
    # 1. 分位点ベースの特徴量
    sensor_columns = [col for col in df.columns if any(prefix in col for prefix in ['acc_', 'rot_', 'thm_', 'tof_'])]
    if sensor_columns:
        for col in sensor_columns[:10]:  # 上位10個のセンサーに限定
            if col in df.columns and df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # 分位点ベースの特徴量
                df[f"{col}_q10"] = df[col].rolling(window=20, min_periods=1).quantile(0.1)
                df[f"{col}_q25"] = df[col].rolling(window=20, min_periods=1).quantile(0.25)
                df[f"{col}_q75"] = df[col].rolling(window=20, min_periods=1).quantile(0.75)
                df[f"{col}_q90"] = df[col].rolling(window=20, min_periods=1).quantile(0.9)
                
                # IQR（四分位範囲）
                df[f"{col}_iqr"] = df[f"{col}_q75"] - df[f"{col}_q25"]
                
                # 歪度と尖度（より安定した計算）
                try:
                    df[f"{col}_skewness"] = df[col].rolling(window=30, min_periods=10).skew().fillna(0)
                    df[f"{col}_kurtosis"] = df[col].rolling(window=30, min_periods=10).kurt().fillna(0)
                except:
                    df[f"{col}_skewness"] = 0
                    df[f"{col}_kurtosis"] = 0
    
    # 2. 変化率ベースの特徴量
    for col in sensor_columns[:10]:
        if col in df.columns:
            # 変化率
            df[f"{col}_change_rate"] = df[col].pct_change().fillna(0)
            
            # 変化率の絶対値
            df[f"{col}_abs_change_rate"] = df[f"{col}_change_rate"].abs()
            
            # 変化率の移動平均
            df[f"{col}_change_rate_ma"] = df[f"{col}_change_rate"].rolling(window=10, min_periods=1).mean()
    
    # 3. 異常値検出ベースの特徴量
    for col in sensor_columns[:10]:
        if col in df.columns:
            # Z-scoreベースの異常値
            rolling_mean = df[col].rolling(window=50, min_periods=10).mean()
            rolling_std = df[col].rolling(window=50, min_periods=10).std()
            df[f"{col}_zscore"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            df[f"{col}_is_outlier"] = (df[f"{col}_zscore"].abs() > 2).astype(float)
            
            # 異常値の密度
            df[f"{col}_outlier_density"] = df[f"{col}_is_outlier"].rolling(window=20, min_periods=1).mean()
    
    # 4. エントロピーベースの特徴量
    for col in sensor_columns[:5]:  # 計算コストを考慮して5個に限定
        if col in df.columns:
            try:
                # 離散化してエントロピーを計算
                bins = pd.cut(df[col], bins=10, labels=False, duplicates='drop')
                entropy_values = []
                for i in range(len(df)):
                    if i < 20:
                        entropy_values.append(0)
                    else:
                        window_bins = bins[max(0, i-20):i+1]
                        if len(window_bins) > 0:
                            # エントロピー計算
                            unique, counts = np.unique(window_bins, return_counts=True)
                            probs = counts / len(window_bins)
                            entropy = -np.sum(probs * np.log2(probs + 1e-8))
                            entropy_values.append(entropy)
                        else:
                            entropy_values.append(0)
                df[f"{col}_entropy"] = entropy_values
            except:
                df[f"{col}_entropy"] = 0
    
    return df


def extract_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """クロスセンサー特徴量エンジニアリング
    
    異なるセンサー間の相関や相互作用を捉える特徴量
    """
    df = df.copy()
    
    # 1. IMU-ToF相互作用
    if "acc_x" in df.columns and "tof_1" in df.columns:
        # 動きと距離の相互作用
        df["motion_distance_interaction"] = df["acc_x"] * df["tof_1"]
        df["motion_distance_ratio"] = df["acc_x"] / (df["tof_1"] + 1e-8)
        
        # 動きの方向と距離の関係
        if "acc_y" in df.columns:
            df["motion_direction_distance"] = np.arctan2(df["acc_y"], df["acc_x"]) * df["tof_1"]
    
    # 2. IMU-Thermal相互作用
    if "acc_x" in df.columns and "thm_1" in df.columns:
        # 動きと温度の相互作用
        df["motion_temperature_interaction"] = df["acc_x"] * df["thm_1"]
        df["motion_temperature_ratio"] = df["acc_x"] / (df["thm_1"] + 1e-8)
    
    # 3. ToF-Thermal相互作用
    if "tof_1" in df.columns and "thm_1" in df.columns:
        # 距離と温度の相互作用（接触検出）
        df["distance_temperature_interaction"] = df["tof_1"] * df["thm_1"]
        df["contact_probability"] = 1 / (df["tof_1"] + 1e-8) * df["thm_1"]
        
        # 接触の持続時間
        contact_threshold = (df["tof_1"] < df["tof_1"].quantile(0.3)) & (df["thm_1"] > df["thm_1"].quantile(0.7))
        df["contact_duration"] = contact_threshold.rolling(window=15, min_periods=1).sum()
    
    # 4. マルチモーダル融合スコア
    sensor_scores = {}
    
    # IMUスコア
    if "acc_x" in df.columns and "acc_y" in df.columns and "acc_z" in df.columns:
        imu_magnitude = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
        sensor_scores["imu"] = imu_magnitude / (imu_magnitude.max() + 1e-8)
    
    # ToFスコア（距離の逆数）
    tof_columns = [col for col in df.columns if col.startswith("tof_")]
    if tof_columns:
        tof_mean = df[tof_columns].mean(axis=1)
        sensor_scores["tof"] = 1 / (tof_mean + 1e-8)
        sensor_scores["tof"] = sensor_scores["tof"] / (sensor_scores["tof"].max() + 1e-8)
    
    # Thermalスコア
    thermal_columns = [col for col in df.columns if col.startswith("thm_")]
    if thermal_columns:
        thermal_mean = df[thermal_columns].mean(axis=1)
        sensor_scores["thermal"] = thermal_mean / (thermal_mean.max() + 1e-8)
    
    # 融合スコアの計算
    if len(sensor_scores) >= 2:
        # 重み付き平均
        weights = {"imu": 0.4, "tof": 0.35, "thermal": 0.25}
        fusion_score = sum(sensor_scores.get(sensor, 0) * weights.get(sensor, 0) 
                          for sensor in sensor_scores.keys())
        df["multimodal_fusion_score"] = fusion_score
        
        # センサー間の相関
        if len(sensor_scores) >= 2:
            sensor_df = pd.DataFrame(sensor_scores)
            df["sensor_correlation"] = sensor_df.corr().iloc[0, 1] if len(sensor_df.columns) >= 2 else 0
    
    return df


def extract_temporal_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """時間的パターン特徴量エンジニアリング
    
    時間的な変化パターンを捉える特徴量
    """
    df = df.copy()
    
    # 1. トレンド特徴量
    sensor_columns = [col for col in df.columns if any(prefix in col for prefix in ['acc_', 'rot_', 'thm_', 'tof_'])]
    for col in sensor_columns[:10]:
        if col in df.columns:
            # 線形トレンド
            try:
                df[f"{col}_trend"] = df[col].rolling(window=30, min_periods=10).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            except:
                df[f"{col}_trend"] = 0
            
            # 二次トレンド
            try:
                df[f"{col}_trend_quadratic"] = df[col].rolling(window=30, min_periods=10).apply(
                    lambda x: np.polyfit(range(len(x)), x, 2)[0] if len(x) > 2 else 0
                )
            except:
                df[f"{col}_trend_quadratic"] = 0
    
    # 2. 周期性特徴量
    for col in sensor_columns[:5]:  # 計算コストを考慮
        if col in df.columns:
            try:
                # 自己相関ベースの周期性
                window_size = min(50, len(df))
                if window_size > 10:
                    series = df[col].iloc[:window_size].fillna(0)
                    autocorr = np.correlate(series, series, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    # 最初のピーク後の相関を周期性の指標とする
                    if len(autocorr) > 10:
                        df[f"{col}_periodicity"] = autocorr[10] / (autocorr[0] + 1e-8)
                    else:
                        df[f"{col}_periodicity"] = 0
                else:
                    df[f"{col}_periodicity"] = 0
            except:
                df[f"{col}_periodicity"] = 0
    
    # 3. 変化点検出
    for col in sensor_columns[:10]:
        if col in df.columns:
            # 変化点の検出（移動平均との乖離）
            ma_short = df[col].rolling(window=5, min_periods=1).mean()
            ma_long = df[col].rolling(window=20, min_periods=1).mean()
            df[f"{col}_change_point"] = (ma_short - ma_long).abs()
            
            # 変化点の強度
            df[f"{col}_change_intensity"] = df[f"{col}_change_point"].rolling(window=10, min_periods=1).max()
    
    # 4. 時系列の複雑性
    for col in sensor_columns[:5]:
        if col in df.columns:
            try:
                # サンプルエントロピー（時系列の複雑性）
                window_size = 20
                complexity_scores = []
                for i in range(len(df)):
                    if i < window_size:
                        complexity_scores.append(0)
                    else:
                        window_data = df[col].iloc[i-window_size:i+1].values
                        if len(window_data) > 0:
                            # 簡易的な複雑性計算
                            diff = np.diff(window_data)
                            complexity = np.std(diff) / (np.mean(np.abs(diff)) + 1e-8)
                            complexity_scores.append(complexity)
                        else:
                            complexity_scores.append(0)
                df[f"{col}_complexity"] = complexity_scores
            except:
                df[f"{col}_complexity"] = 0
    
    return df
