"""
Test for Gold Layer Data Functions - CMI Sensor Data
Tests ML-ready data preparation with GroupKFold support
CLAUDE.md: Gold layer tests for BFRB detection with participant-based CV
"""

import tempfile
from unittest.mock import Mock, patch, MagicMock
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PolynomialFeatures

from src.data.gold import (
    encode_bfrb_target, 
    prepare_ml_ready_data,
    clean_and_validate_sensor_features,
    select_sensor_features,
    get_ml_ready_sequences,
    create_submission_dataframe,
    load_gold_data,
    create_gold_tables,
    get_sensor_feature_names,
    setup_groupkfold_cv,
    create_sequence_windows,
    create_submission_format,
    extract_model_arrays
)

# Import common fixtures and utilities
from tests.conftest import (
    sample_bronze_data, sample_silver_data, sample_gold_data, edge_case_data, 
    missing_data, large_test_data, mock_db_connection, assert_sub_second_performance, 
    assert_lightgbm_compatibility, assert_no_data_loss, assert_data_quality, 
    performance_test, lightgbm_compatibility_test, assert_database_operations,
    assert_feature_engineering_quality, create_correlated_test_data, 
    create_missing_pattern_data, create_outlier_data
)


@pytest.fixture
def sample_cmi_data():
    """Create sample CMI sensor data for testing"""
    return pd.DataFrame({
        'id': range(1, 101),
        'participant_id': np.repeat(range(1, 21), 5),  # 20 participants, 5 samples each
        'series_id': range(1, 101),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='20ms'),  # 50Hz
        # IMU sensor data
        'acc_x': np.random.normal(0, 1, 100),
        'acc_y': np.random.normal(0, 1, 100),
        'acc_z': np.random.normal(9.8, 1, 100),
        'rot_x': np.random.normal(0, 0.1, 100),
        'rot_y': np.random.normal(0, 0.1, 100),
        'rot_z': np.random.normal(0, 0.1, 100),
        # ToF distance sensors
        'tof_0': np.random.uniform(50, 500, 100),
        'tof_1': np.random.uniform(50, 500, 100),
        'tof_2': np.random.uniform(50, 500, 100),
        'tof_3': np.random.uniform(50, 500, 100),
        # Thermopile temperature sensors
        'thm_0': np.random.normal(25, 2, 100),
        'thm_1': np.random.normal(25, 2, 100),
        'thm_2': np.random.normal(25, 2, 100),
        'thm_3': np.random.normal(25, 2, 100),
        'thm_4': np.random.normal(25, 2, 100),
        # Target labels
        'label': np.random.choice(['no_behavior', 'behavior_1', 'behavior_2'], 100)
    })


@pytest.fixture
def sample_processed_cmi_data():
    """Create sample processed CMI data with engineered features"""
    return pd.DataFrame({
        'id': range(1, 51),
        'participant_id': np.repeat(range(1, 11), 5),  # 10 participants, 5 samples each
        'series_id': range(1, 51),
        # Sensor fusion features
        'imu_total_motion': np.random.uniform(0, 10, 50),
        'thermal_distance_interaction': np.random.uniform(-100, 100, 50),
        'movement_intensity': np.random.uniform(0, 5, 50),
        'proximity_mean': np.random.uniform(100, 400, 50),
        'thermal_contact_indicator': np.random.uniform(0, 1, 50),
        'close_proximity_ratio': np.random.uniform(0, 0.5, 50),
        # Statistical features
        'imu_acc_mean': np.random.normal(0, 1, 50),
        'imu_gyro_mean': np.random.normal(0, 0.1, 50),
        'thermal_mean': np.random.normal(25, 2, 50),
        'tof_mean': np.random.uniform(100, 400, 50),
        # tsfresh features
        'tsfresh_acc_x_mean': np.random.normal(0, 1, 50),
        'tsfresh_acc_x_std': np.random.uniform(0, 2, 50),
        # Frequency features
        'acc_x_spectral_centroid': np.random.uniform(0, 25, 50),
        'acc_x_dominant_freq': np.random.uniform(0, 25, 50),
        # Target
        'label': np.random.choice(['no_behavior', 'behavior_1', 'behavior_2'], 50)
    })


class TestGoldCMISensorFunctions:
    """Gold layer function tests for CMI sensor data"""

    def test_prepare_ml_ready_data_basic(self, sample_processed_cmi_data):
        """Test basic ML-ready data preparation"""
        result = prepare_ml_ready_data(sample_processed_cmi_data)

        # Use common assertions
        assert_no_data_loss(sample_processed_cmi_data, result)
        assert_data_quality(result)

        # CMI-specific checks
        assert 'participant_id' in result.columns
        assert 'label_encoded' in result.columns

    def test_prepare_ml_ready_data_with_target(self, sample_processed_cmi_data):
        """Test ML-ready data preparation with BFRB target"""
        result = prepare_ml_ready_data(sample_processed_cmi_data, target_col="label")

        # Use common assertions
        assert_no_data_loss(sample_processed_cmi_data, result)
        assert_data_quality(result)

        # Target encoding should be present
        assert 'label_encoded' in result.columns

    def test_encode_bfrb_target_basic(self, sample_processed_cmi_data):
        """Test BFRB target encoding"""
        result = encode_bfrb_target(sample_processed_cmi_data)

        # Use common assertions
        assert_no_data_loss(sample_processed_cmi_data, result)
        assert_data_quality(result)
        
        # Should have encoded labels
        assert "label_encoded" in result.columns
        assert "label_binary" in result.columns
        
        # Encoded values should be integers
        assert result["label_encoded"].dtype in ['int32', 'int64']
        assert result["label_binary"].dtype in ['int32', 'int64']

    def test_encode_bfrb_target_multiclass(self):
        """Test BFRB target encoding with multi-class labels"""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'label': ['no_behavior', 'behavior_1', 'behavior_2', 'behavior_1']
        })
        result = encode_bfrb_target(df, target_col="label")

        # Should create encoded and binary versions
        assert "label_encoded" in result.columns
        assert "label_binary" in result.columns
        
        # Binary should be 0 for no_behavior, 1 for any behavior
        assert result.loc[0, 'label_binary'] == 0  # no_behavior
        assert result.loc[1, 'label_binary'] == 1  # behavior_1
        assert result.loc[2, 'label_binary'] == 1  # behavior_2

    @patch("src.data.gold.duckdb.connect")
    def test_create_gold_tables_success(self, mock_connect, mock_db_connection):
        """Test gold table creation for CMI data"""
        mock_connect.return_value = mock_db_connection.get_mock_conn()

        create_gold_tables()

        # Use common database assertions
        assert_database_operations(mock_connect)

    @patch("src.data.gold.duckdb.connect")
    def test_load_gold_data_success(self, mock_connect, mock_db_connection):
        """Test gold data loading for CMI sensor data"""
        mock_connect.return_value = mock_db_connection.get_mock_conn()

        train, test = load_gold_data()

        # Use common assertions
        assert len(train) == 5  # sample_bronze_data length
        assert len(test) == 5   # sample_gold_data length
        assert "id" in train.columns
        assert "id" in test.columns

    def test_get_ml_ready_sequences(self, sample_processed_cmi_data):
        """Test ML-ready sequence preparation"""
        X, y, groups = get_ml_ready_sequences(sample_processed_cmi_data)
        
        # Basic structure checks
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(groups, pd.Series)
        
        # Length consistency
        assert len(X) == len(y) == len(groups)
        assert len(X) == len(sample_processed_cmi_data)
        
        # Target encoding
        assert y.dtype in ['int32', 'int64']
        
        # Groups for GroupKFold
        assert groups.name == 'participant_id' or groups.dtype in ['int32', 'int64']

    def test_setup_groupkfold_cv(self, sample_processed_cmi_data):
        """Test GroupKFold CV setup for participant-based splitting"""
        gkf, cv_splits = setup_groupkfold_cv(sample_processed_cmi_data, n_splits=3)
        
        # Basic structure
        assert len(cv_splits) == 3
        assert all(len(split) == 2 for split in cv_splits)
        
        # No participant leakage validation
        X, y, groups = get_ml_ready_sequences(sample_processed_cmi_data)
        for train_idx, val_idx in cv_splits:
            train_participants = set(groups.iloc[train_idx])
            val_participants = set(groups.iloc[val_idx])
            assert not (train_participants & val_participants), "Participant leakage detected!"

    def test_create_sequence_windows(self, sample_cmi_data):
        """Test sequence window creation for CNN/RNN training"""
        windowed_df = create_sequence_windows(
            sample_cmi_data, 
            window_size=10, 
            overlap=0.5
        )
        
        # Basic structure
        assert isinstance(windowed_df, pd.DataFrame)
        assert len(windowed_df) > 0
        
        # Should have window metadata
        assert 'window_start' in windowed_df.columns
        assert 'window_end' in windowed_df.columns
        
        # Should have aggregated features
        aggregated_features = [col for col in windowed_df.columns if col.endswith('_mean')]
        assert len(aggregated_features) > 0

    def test_empty_dataframe_handling(self):
        """Test functions handle empty DataFrames gracefully"""
        empty_df = pd.DataFrame()

        # Should not crash
        result1 = prepare_ml_ready_data(empty_df)
        result2 = encode_bfrb_target(empty_df)

        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)


class TestGoldCMISensorCleaning:
    """Test gold.py personality data cleaning functionality"""

    def test_clean_and_validate_sensor_features_basic(self, sample_cmi_data):
        """Test basic personality feature cleaning"""
        result = clean_and_validate_sensor_features(sample_cmi_data)
        
        # Use common assertions
        assert_no_data_loss(sample_cmi_data, result)
        assert_data_quality(result)
        
        # Personality-specific validations
        # Time features should be within reasonable range
        if 'timestamp' in result.columns:
            assert result['timestamp'].min() >= pd.Timestamp('2024-01-01')
            assert result['timestamp'].max() <= pd.Timestamp('2024-01-01') + pd.Timedelta(100, '20ms')
        
        # Sensor features should be within reasonable range
        for col in ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'tof_0', 'tof_1', 'tof_2', 'tof_3', 'thm_0', 'thm_1', 'thm_2', 'thm_3', 'thm_4']:
            if col in result.columns:
                assert result[col].min() >= -10 and result[col].max() <= 10 # Placeholder for actual min/max

    def test_clean_and_validate_sensor_features_infinite_values(self):
        """Test cleaning of infinite values in personality data"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'acc_x': [1.0, np.inf, -np.inf],
            'acc_y': [2, 3, np.inf],
            'acc_z': [5, -np.inf, 10]
        })
        
        result = clean_and_validate_sensor_features(df)
        
        # No infinite values should remain
        for col in result.columns:
            if result[col].dtype in ['float64', 'float32']:
                assert not np.isinf(result[col]).any()

    def test_clean_and_validate_sensor_features_missing_values(self):
        """Test handling of missing values in personality data"""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'acc_x': [1.0, np.nan, 3.0, 4.0],
            'acc_y': [2, 3, np.nan, 5],
            'label': ['no_behavior', 'behavior_1', np.nan, 'behavior_1']
        })
        
        result = clean_and_validate_sensor_features(df)
        
        # Missing values should be handled appropriately
        assert len(result) == len(df)
        
        # Numeric missing values should be filled
        if 'acc_x' in result.columns:
            assert not result['acc_x'].isnull().any()
        
        # Categorical missing values should be handled
        if 'label' in result.columns:
            assert not result['label'].isnull().any()


class TestGoldCMISensorFeatureSelection:
    """Test sensor-aware feature selection functionality"""

    def test_select_sensor_features_basic(self, sample_processed_cmi_data):
        """Test basic sensor feature selection"""
        selected_features = select_sensor_features(
            sample_processed_cmi_data, 
            'label', 
            k=5
        )
        
        # Basic validation
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 5
        assert 'id' not in selected_features
        assert 'label' not in selected_features
        assert 'participant_id' not in selected_features

    def test_select_sensor_features_prioritization(self, sample_processed_cmi_data):
        """Test sensor feature prioritization"""
        selected_features = select_sensor_features(
            sample_processed_cmi_data, 
            'label', 
            k=3
        )
        
        # Should prioritize sensor fusion features
        priority_patterns = [
            'imu_total_motion', 'thermal_distance_interaction', 'movement_intensity'
        ]
        
        # At least one priority feature should be selected
        priority_selected = any(
            any(pattern in feature for pattern in priority_patterns)
            for feature in selected_features
        )
        assert priority_selected or len(selected_features) == 0

    def test_select_sensor_features_statistical_method(self, sample_processed_cmi_data):
        """Test statistical feature selection method"""
        selected_features = select_sensor_features(
            sample_processed_cmi_data, 
            'label', 
            k=3, 
            method='statistical'
        )
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 3


class TestGoldCMISensorGroupKFoldSupport:
    """Test GroupKFold support for participant-based CV"""

    def test_groupkfold_cv_no_leakage(self, sample_processed_cmi_data):
        """Test GroupKFold ensures no participant leakage"""
        gkf, cv_splits = setup_groupkfold_cv(sample_processed_cmi_data, n_splits=3)
        
        X, y, groups = get_ml_ready_sequences(sample_processed_cmi_data)
        
        # Check each fold for participant leakage
        for i, (train_idx, val_idx) in enumerate(cv_splits):
            train_participants = set(groups.iloc[train_idx])
            val_participants = set(groups.iloc[val_idx])
            
            # No overlap between train and validation participants
            assert not (train_participants & val_participants), f"Fold {i}: Participant leakage detected!"
            
            # Both sets should be non-empty
            assert len(train_participants) > 0, f"Fold {i}: No training participants"
            assert len(val_participants) > 0, f"Fold {i}: No validation participants"

    def test_groupkfold_cv_missing_participant_id(self):
        """Test GroupKFold with missing participant_id"""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'feature1': [1, 2, 3, 4],
            'label': ['no_behavior', 'behavior_1', 'behavior_2', 'behavior_1']
        })
        
        # Should create dummy groups when participant_id is missing
        X, y, groups = get_ml_ready_sequences(df)
        
        assert isinstance(groups, pd.Series)
        assert len(groups) == len(df)

    def test_groupkfold_participant_distribution(self, sample_processed_cmi_data):
        """Test participant distribution across folds"""
        gkf, cv_splits = setup_groupkfold_cv(sample_processed_cmi_data, n_splits=3)
        
        X, y, groups = get_ml_ready_sequences(sample_processed_cmi_data)
        total_participants = groups.nunique()
        
        # Each fold should have reasonable participant distribution
        for train_idx, val_idx in cv_splits:
            train_participants = groups.iloc[train_idx].nunique()
            val_participants = groups.iloc[val_idx].nunique()
            
            # Validation should have at least 1 participant
            assert val_participants >= 1
            # Training should have majority of participants
            assert train_participants >= val_participants


class TestGoldCMISensorSubmissionFormat:
    """Test CMI competition submission format"""

    def test_create_submission_format_basic(self, sample_processed_cmi_data):
        """Test basic submission format creation"""
        # Mock predictions
        predictions = np.array([0, 1, 2, 1, 0] * 10)  # Multi-class predictions
        
        filename = create_submission_format(predictions, filename="test_submission.csv")
        
        # Should create a file
        assert filename.endswith('.csv')
        
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)

    def test_create_submission_dataframe_multiclass(self, sample_processed_cmi_data):
        """Test submission DataFrame creation for multi-class"""
        predictions = np.array([0, 1, 2, 1, 0] * 10)  # Multi-class predictions
        
        submission = create_submission_dataframe(sample_processed_cmi_data, predictions)
        
        # Basic structure
        assert isinstance(submission, pd.DataFrame)
        assert len(submission) == len(sample_processed_cmi_data)
        assert 'id' in submission.columns
        assert 'label' in submission.columns
        
        # Multi-class labels
        unique_labels = set(submission['label'])
        expected_labels = {'no_behavior', 'behavior_1', 'behavior_2', 'behavior_3'}
        assert unique_labels.issubset(expected_labels)

    def test_create_submission_dataframe_binary(self, sample_processed_cmi_data):
        """Test submission DataFrame creation for binary"""
        predictions = np.array([0.1, 0.9, 0.3, 0.7, 0.5] * 10)  # Binary predictions
        
        submission = create_submission_dataframe(sample_processed_cmi_data, predictions)
        
        # Basic structure
        assert isinstance(submission, pd.DataFrame)
        assert len(submission) == len(sample_processed_cmi_data)
        
        # Binary labels
        unique_labels = set(submission['label'])
        expected_labels = {'no_behavior', 'behavior'}
        assert unique_labels.issubset(expected_labels)


class TestGoldCMISensorLightGBMCompatibility:
    """Test LightGBM compatibility for CMI sensor data"""

    def test_get_ml_ready_sequences_lightgbm_interface(self, sample_processed_cmi_data):
        """Test LightGBM interface"""
        X, y, groups = get_ml_ready_sequences(sample_processed_cmi_data)
        
        # Use common assertions
        assert_lightgbm_compatibility(X)
        
        # LightGBM-specific checks
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert y.dtype in ['int32', 'int64']
        
        # No missing values
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_extract_model_arrays(self, sample_processed_cmi_data):
        """Test model array extraction for traditional ML workflows"""
        X, y, feature_names = extract_model_arrays(sample_processed_cmi_data)
        
        # Basic structure
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)
        
        # Dimensions
        assert X.shape[0] == len(y)
        assert X.shape[1] == len(feature_names)
        
        # Feature names should exclude metadata
        metadata_cols = ['id', 'participant_id', 'series_id', 'label']
        for col in metadata_cols:
            assert col not in feature_names

    def test_get_sensor_feature_names(self, sample_processed_cmi_data):
        """Test sensor feature name extraction"""
        feature_names = get_sensor_feature_names(sample_processed_cmi_data)
        
        # Basic validation
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        # Should exclude metadata
        metadata_cols = ['id', 'participant_id', 'series_id', 'timestamp', 'label']
        for col in metadata_cols:
            assert col not in feature_names
        
        # Should include sensor features
        sensor_features = [col for col in feature_names if any(pattern in col for pattern in ['imu_', 'thermal_', 'tof_'])]
        assert len(sensor_features) > 0


class TestGoldCMISensorDataIntegration:
    """Test Gold layer data integration and pipeline"""

    def test_full_pipeline_integration(self, sample_cmi_data):
        """Test full Gold layer pipeline integration"""
        # Step 1: Clean sensor features
        cleaned_data = clean_and_validate_sensor_features(sample_cmi_data)
        
        # Step 2: Encode targets
        encoded_data = encode_bfrb_target(cleaned_data)
        
        # Step 3: Prepare ML-ready data
        ml_ready_data = prepare_ml_ready_data(encoded_data)
        
        # Step 4: Get sequences for training
        X, y, groups = get_ml_ready_sequences(ml_ready_data)
        
        # Integration checks
        assert len(X) == len(sample_cmi_data)
        assert len(y) == len(sample_cmi_data)
        assert len(groups) == len(sample_cmi_data)
        
        # Data quality throughout pipeline
        assert_data_quality(ml_ready_data)
        assert_lightgbm_compatibility(X)

    def test_memory_optimization(self, large_test_data):
        """Test memory optimization for large datasets"""
        # Convert to sensor data format
        sensor_data = large_test_data.copy()
        sensor_data['participant_id'] = sensor_data.index // 100
        sensor_data['label'] = 'no_behavior'
        
        X, y, groups = get_ml_ready_sequences(sensor_data)
        
        # Memory efficiency checks
        assert X.memory_usage(deep=True).sum() < sensor_data.memory_usage(deep=True).sum()
        assert y.dtype in ['int32', 'int64']  # Efficient integer encoding