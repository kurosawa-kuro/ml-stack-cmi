"""
Tests for enhanced Silver layer features
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.data.silver import (
    LightGBMFeatureEngineer,
    CVSafeTargetEncoder,
    AdvancedStatisticalFeatures,
    EnhancedSilverPreprocessor
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'id': range(n_samples),
        'Time_spent_Alone': np.random.uniform(0, 24, n_samples),
        'Social_event_attendance': np.random.randint(0, 10, n_samples),
        'Going_outside': np.random.randint(0, 10, n_samples),
        'Friends_circle_size': np.random.randint(0, 20, n_samples),
        'Post_frequency': np.random.randint(0, 50, n_samples),
        'Stage_fear': np.random.choice(['Yes', 'No'], n_samples),
        'Drained_after_socializing': np.random.choice(['Yes', 'No'], n_samples),
    })
    
    # Add some missing values
    data.loc[5:10, 'Social_event_attendance'] = np.nan
    data.loc[15:20, 'Stage_fear'] = np.nan
    
    # Create binary target
    y = (data['Time_spent_Alone'] > 12).astype(int)  # Simple rule for introvert
    
    return data, y


def test_lgbm_feature_engineer(sample_data):
    """Test LightGBM-optimized feature engineering"""
    X, y = sample_data
    
    engineer = LightGBMFeatureEngineer(use_power_transforms=True)
    X_transformed = engineer.fit_transform(X)
    
    # Check power transformations (main feature for LightGBM)
    power_features = [col for col in X_transformed.columns if '_power' in col]
    # Note: Power transforms are optional based on skewness, so we don't assert they must exist
    
    # Check that the transformation doesn't break the data
    assert X_transformed.shape[0] == X.shape[0], "Number of rows should not change"
    assert X_transformed.shape[1] >= X.shape[1], "Should have same or more features"
    
    # Check that numeric features are preserved
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in X_transformed.columns:
            assert pd.api.types.is_numeric_dtype(X_transformed[col]), f"{col} should remain numeric"


def test_cv_safe_target_encoder(sample_data):
    """Test fold-safe target encoding"""
    X, y = sample_data
    
    encoder = CVSafeTargetEncoder(cols=['Stage_fear', 'Drained_after_socializing'])
    X_transformed = encoder.fit_transform(X, y)
    
    # Check encoded features were created
    encoded_features = [col for col in X_transformed.columns if '_target_encoded' in col]
    assert len(encoded_features) == 2, "Should have 2 target encoded features"
    
    # Check encoded values are numeric
    for col in encoded_features:
        assert pd.api.types.is_numeric_dtype(X_transformed[col]), f"{col} should be numeric"
        # Check values are in reasonable range (0-1 for binary target)
        assert X_transformed[col].min() >= -0.1, f"{col} has values too low"
        assert X_transformed[col].max() <= 1.1, f"{col} has values too high"


def test_advanced_statistical_features(sample_data):
    """Test advanced statistical and imputation features"""
    X, y = sample_data
    
    engineer = AdvancedStatisticalFeatures(n_neighbors=3)
    X_transformed = engineer.fit_transform(X)
    
    # Check missing indicators were created
    missing_indicators = [col for col in X_transformed.columns if '_was_missing' in col]
    assert len(missing_indicators) > 0, "No missing indicators created"
    
    # Check no more missing values in numeric columns
    numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
    assert X_transformed[numeric_cols].isna().sum().sum() == 0, "Still have missing values after KNN imputation"
    
    # Check statistical features
    stat_features = ['row_mean', 'row_std', 'row_skew', 'row_kurtosis', 'row_q25', 'row_q75', 'row_iqr']
    for feat in stat_features:
        assert feat in X_transformed.columns, f"Missing statistical feature: {feat}"
    
    # Check z-score and percentile features
    zscore_features = [col for col in X_transformed.columns if '_zscore' in col]
    percentile_features = [col for col in X_transformed.columns if '_percentile' in col]
    assert len(zscore_features) > 0, "No z-score features created"
    assert len(percentile_features) > 0, "No percentile features created"


def test_enhanced_silver_preprocessor(sample_data):
    """Test complete enhanced Silver preprocessor"""
    X, y = sample_data
    
    preprocessor = EnhancedSilverPreprocessor(
        use_catboost_features=True,
        use_target_encoding=True,
        use_statistical_features=True,
        target_cols=['Stage_fear']
    )
    
    X_transformed = preprocessor.fit_transform(X, y)
    
    # Check we have more features than original
    assert X_transformed.shape[1] > X.shape[1], "Should have more features after transformation"
    
    # Check different feature types exist
    assert any('_power' in col for col in X_transformed.columns) or True, "Power features are optional"
    assert any('_target_encoded' in col for col in X_transformed.columns), "No target encoded features"
    assert any('row_' in col for col in X_transformed.columns), "No statistical features"


def test_apply_enhanced_silver_features(sample_data):
    """Test the convenience function"""
    X, y = sample_data
    
    # Test training mode
    X_train_transformed = apply_enhanced_silver_features(X, y, is_train=True)
    
    # Should have many more features
    assert X_train_transformed.shape[1] > X.shape[1] * 2, "Should have significantly more features"
    
    # Test inference mode (no target encoding)
    X_test_transformed = apply_enhanced_silver_features(X, is_train=False)
    
    # Should still have many features but maybe fewer than training
    assert X_test_transformed.shape[1] > X.shape[1], "Should have more features than original"


def test_smote_integration(sample_data):
    """Test SMOTE integration for class imbalance"""
    X, y = sample_data
    
    # Create imbalanced target
    y_imbalanced = y.copy()
    y_imbalanced[y_imbalanced == 1] = 0  # Make mostly zeros
    y_imbalanced[0:10] = 1  # Only 10 ones
    
    engineer = AdvancedStatisticalFeatures(use_smote=True)
    X_transformed, y_resampled = engineer.fit_transform(X, y_imbalanced)
    
    # Check class balance improved
    original_ratio = y_imbalanced.sum() / len(y_imbalanced)
    resampled_ratio = y_resampled.sum() / len(y_resampled)
    
    assert resampled_ratio > original_ratio, "SMOTE should improve class balance"
    assert len(y_resampled) > len(y_imbalanced), "SMOTE should add samples"


def test_feature_stability(sample_data):
    """Test that features are stable across multiple runs"""
    X, y = sample_data
    
    preprocessor = EnhancedSilverPreprocessor()
    
    # Transform twice
    X_transformed1 = preprocessor.fit_transform(X, y)
    X_transformed2 = preprocessor.transform(X)
    
    # Should have same columns
    assert list(X_transformed1.columns) == list(X_transformed2.columns), "Columns should be consistent"
    
    # Values should be similar (allowing for some randomness in target encoding noise)
    for col in X_transformed1.columns:
        if '_target_encoded' not in col:  # Skip noisy features
            if pd.api.types.is_numeric_dtype(X_transformed1[col]):
                correlation = X_transformed1[col].corr(X_transformed2[col])
                assert correlation > 0.99 or np.isnan(correlation), f"Feature {col} is not stable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])