#!/usr/bin/env python3
"""Enhanced Silver Layer with BFRB-specific features."""

import pandas as pd
import numpy as np
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')

from src.data.bronze import load_bronze_data
import duckdb
import time

def timer(func):
    """Simple timer decorator."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} completed in {end-start:.2f} seconds")
        return result
    return wrapper

def create_bfrb_features(df):
    """Create Body-Focused Repetitive Behavior specific features."""
    print("Creating BFRB-specific features...")
    
    # 1. Repetitive Motion Detection
    # Calculate autocorrelation for detecting repetitive patterns
    for axis in ['x', 'y', 'z']:
        acc_col = f'acc_{axis}'
        if acc_col in df.columns:
            # Autocorrelation at different lags
            df[f'{acc_col}_autocorr_lag5'] = df.groupby('sequence_id')[acc_col].transform(
                lambda x: x.rolling(10, min_periods=5).apply(
                    lambda y: y.autocorr(lag=5) if len(y) >= 10 else 0
                )
            ).fillna(0)
    
    # 2. Touch Detection Features
    # Proximity to body (low ToF values indicate closeness)
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    if tof_cols:
        # Minimum distance across all ToF sensors
        df['min_tof_distance'] = df[tof_cols].min(axis=1)
        df['touch_detected'] = (df['min_tof_distance'] < df['min_tof_distance'].quantile(0.1)).astype(int)
        
        # Sustained touch (rolling mean of touch detection)
        df['sustained_touch'] = df.groupby('sequence_id')['touch_detected'].transform(
            lambda x: x.rolling(25, min_periods=10).mean()
        ).fillna(0)
    
    # 3. Temperature-based Contact Features
    thm_cols = [col for col in df.columns if col.startswith('thm_')]
    if thm_cols:
        # Temperature spike detection (contact with warm body)
        df['thm_mean'] = df[thm_cols].mean(axis=1)
        df['thm_spike'] = df.groupby('sequence_id')['thm_mean'].transform(
            lambda x: (x - x.rolling(50, min_periods=20).mean()) / (x.rolling(50, min_periods=20).std() + 1e-6)
        ).fillna(0)
        df['thermal_contact'] = (df['thm_spike'] > 2).astype(int)
    
    # 4. Gesture-Specific Patterns
    # Hand movement velocity and acceleration
    df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    
    # Jerk (rate of change of acceleration) - important for gesture detection
    df['jerk'] = df.groupby('sequence_id')['acc_magnitude'].transform(
        lambda x: x.diff().fillna(0)
    )
    
    # 5. Behavioral State Features
    # Movement intensity categories
    df['movement_state'] = pd.cut(
        df['acc_magnitude'], 
        bins=[0, 0.5, 1.0, 2.0, np.inf],
        labels=['still', 'low_movement', 'moderate_movement', 'high_movement']
    )
    
    # 6. Sequence Position Features (important based on analysis)
    df['sequence_progress'] = df.groupby('sequence_id').cumcount() / df.groupby('sequence_id')['sequence_id'].transform('count')
    df['sequence_start'] = (df['sequence_progress'] < 0.1).astype(int)
    df['sequence_end'] = (df['sequence_progress'] > 0.9).astype(int)
    
    # 7. Cross-Modal Correlations
    # IMU-Thermal correlation (movement with temperature change)
    df['imu_thermal_corr'] = df['acc_magnitude'] * df.get('thm_mean', 0)
    
    # IMU-ToF correlation (movement with proximity)
    if 'min_tof_distance' in df.columns:
        df['imu_tof_corr'] = df['acc_magnitude'] * (1 / (df['min_tof_distance'] + 1))
    
    return df

def create_advanced_frequency_features(df):
    """Create advanced frequency domain features."""
    print("Creating advanced frequency features...")
    
    # Group by series for window-based processing
    freq_features = []
    
    for sequence_id in df['sequence_id'].unique():
        series_data = df[df['sequence_id'] == sequence_id].copy()
        
        if len(series_data) < 50:  # Skip short sequences
            continue
            
        # FFT on acceleration magnitude
        acc_mag = series_data['acc_magnitude'].values
        fft_vals = np.fft.fft(acc_mag)
        fft_freq = np.fft.fftfreq(len(acc_mag), d=1/50)  # 50Hz sampling
        
        # Spectral features
        power_spectrum = np.abs(fft_vals)**2
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_freq = np.abs(fft_freq[dominant_freq_idx])
        
        # Spectral entropy (measure of signal complexity)
        norm_power = power_spectrum / np.sum(power_spectrum)
        spectral_entropy = -np.sum(norm_power * np.log2(norm_power + 1e-10))
        
        # Add features to series
        series_data['dominant_freq'] = dominant_freq
        series_data['spectral_entropy'] = spectral_entropy
        
        # Band power features
        freq_bands = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 25)]
        for low, high in freq_bands:
            band_mask = (np.abs(fft_freq) >= low) & (np.abs(fft_freq) < high)
            band_power = np.sum(power_spectrum[band_mask])
            series_data[f'band_power_{low}_{high}hz'] = band_power
        
        freq_features.append(series_data)
    
    if freq_features:
        df_freq = pd.concat(freq_features, ignore_index=True)
        # Merge back with original df
        freq_cols = [col for col in df_freq.columns if col not in df.columns]
        for col in freq_cols:
            df[col] = df_freq[col]
    
    return df

@timer
def enhance_silver_layer():
    """Create enhanced Silver layer with BFRB-specific features."""
    print("🥈 Enhanced Silver Layer Processing")
    print("=" * 50)
    
    # Load Bronze data
    train_df, test_df = load_bronze_data()
    print(f"Loaded Bronze data - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Apply BFRB-specific features
    train_df = create_bfrb_features(train_df)
    test_df = create_bfrb_features(test_df)
    
    # Apply advanced frequency features
    train_df = create_advanced_frequency_features(train_df)
    test_df = create_advanced_frequency_features(test_df)
    
    # One-hot encode movement states
    if 'movement_state' in train_df.columns:
        movement_dummies_train = pd.get_dummies(train_df['movement_state'], prefix='movement_state')
        movement_dummies_test = pd.get_dummies(test_df['movement_state'], prefix='movement_state')
        
        # Ensure same columns
        for col in movement_dummies_train.columns:
            if col not in movement_dummies_test.columns:
                movement_dummies_test[col] = 0
        
        train_df = pd.concat([train_df, movement_dummies_train], axis=1)
        test_df = pd.concat([test_df, movement_dummies_test], axis=1)
        
        # Drop original categorical column
        train_df = train_df.drop('movement_state', axis=1)
        test_df = test_df.drop('movement_state', axis=1)
    
    print(f"\nEnhanced features - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Save to database
    conn = duckdb.connect('data/kaggle_datasets.duckdb')
    train_df.to_sql('silver_enhanced_train', conn, if_exists='replace', index=False)
    test_df.to_sql('silver_enhanced_test', conn, if_exists='replace', index=False)
    conn.close()
    
    print("\n✅ Enhanced Silver layer created successfully!")
    print(f"New features added: {len([col for col in train_df.columns if col not in ['participant_id', 'sequence_id', 'timestamp']])}")
    
    # Show sample of new features
    new_features = [col for col in train_df.columns if any(
        keyword in col for keyword in ['autocorr', 'touch', 'spike', 'jerk', 'corr', 'band_power', 'entropy', 'sustained']
    )]
    print(f"\nSample new features: {new_features[:10]}")
    
    return train_df, test_df

if __name__ == "__main__":
    enhance_silver_layer()