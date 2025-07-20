#!/usr/bin/env python3
"""
Simple Prediction Script
Generates a basic submission file with data cleaning to handle nan/inf issues
"""

import sys
import warnings
from pathlib import Path
from typing import Tuple

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")

DB_PATH = "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by handling nan/inf values"""
    df = df.copy()
    
    # Convert inf to nan first
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Get numeric columns (excluding id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'id']
    
    # Fill nan values with median for numeric columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):  # If all values are nan
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(median_val)
    
    # Ensure no remaining nan/inf values
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        # Clip extreme values for numerical stability
        df[col] = df[col].clip(-1e6, 1e6)
    
    return df


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data from DuckDB"""
    conn = duckdb.connect(DB_PATH)
    
    # Try to load from bronze first, then raw
    try:
        train_df = conn.execute("SELECT * FROM bronze.train").df()
        test_df = conn.execute("SELECT * FROM bronze.test").df()
        print("Loaded data from bronze tables")
    except:
        try:
            train_df = conn.execute("SELECT * FROM playground_series_s5e7.train").df()
            test_df = conn.execute("SELECT * FROM playground_series_s5e7.test").df()
            print("Loaded data from raw tables")
        except Exception as e:
            print(f"Error loading data: {e}")
            conn.close()
            raise
    
    conn.close()
    return train_df, test_df


def prepare_simple_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare simple features with basic preprocessing"""
    
    # Clean both datasets
    train_clean = clean_data(train_df)
    test_clean = clean_data(test_df)
    
    # Encode categorical variables (Yes/No -> 1/0)
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    
    for col in categorical_cols:
        if col in train_clean.columns:
            # Handle various formats
            train_clean[col] = train_clean[col].astype(str).str.lower()
            test_clean[col] = test_clean[col].astype(str).str.lower()
            
            train_clean[col] = train_clean[col].map({'yes': 1, 'no': 0}).fillna(0)
            test_clean[col] = test_clean[col].map({'yes': 1, 'no': 0}).fillna(0)
    
    # Encode target variable
    if 'Personality' in train_clean.columns:
        train_clean['Personality_encoded'] = (train_clean['Personality'] == 'Extrovert').astype(int)
    
    # Create some basic engineered features
    numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                       'Friends_circle_size', 'Post_frequency']
    
    # Add simple ratios and interactions
    if all(col in train_clean.columns for col in numeric_features):
        # Social ratio
        train_clean['social_ratio'] = train_clean['Social_event_attendance'] / (train_clean['Going_outside'] + 1)
        test_clean['social_ratio'] = test_clean['Social_event_attendance'] / (test_clean['Going_outside'] + 1)
        
        # Activity sum
        train_clean['activity_sum'] = train_clean[numeric_features].sum(axis=1)
        test_clean['activity_sum'] = test_clean[numeric_features].sum(axis=1)
        
        # Post per friend
        train_clean['post_per_friend'] = train_clean['Post_frequency'] / (train_clean['Friends_circle_size'] + 1)
        test_clean['post_per_friend'] = test_clean['Post_frequency'] / (test_clean['Friends_circle_size'] + 1)
    
    return train_clean, test_clean


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns excluding id and target"""
    exclude_cols = ['id', 'Personality', 'Personality_encoded']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def train_simple_model(X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
    """Train a simple LightGBM model with basic parameters"""
    
    # Simple, stable parameters
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        random_state=42,
        n_estimators=100,
        verbosity=-1
    )
    
    model.fit(X_train, y_train)
    return model


def cross_validate_simple(X: np.ndarray, y: np.ndarray) -> float:
    """Simple cross-validation to check model performance"""
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = train_simple_model(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)
        scores.append(score)
    
    return np.mean(scores)


def create_submission(test_df: pd.DataFrame, predictions: np.ndarray, filename: str = "submission.csv") -> None:
    """Create submission file"""
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Extrovert' if pred == 1 else 'Introvert' for pred in predictions]
    })
    
    # Save to file
    submission_df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    print(f"Prediction distribution:")
    print(submission_df['Personality'].value_counts())


def main():
    """Main prediction workflow"""
    print("=" * 50)
    print("Simple Prediction Script")
    print("=" * 50)
    
    try:
        # Load data
        print("1. Loading data...")
        train_df, test_df = load_raw_data()
        print(f"   Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        # Prepare features
        print("2. Preparing features...")
        train_prepared, test_prepared = prepare_simple_features(train_df, test_df)
        
        # Get feature columns
        feature_cols = get_feature_columns(train_prepared)
        print(f"   Using {len(feature_cols)} features: {feature_cols}")
        
        # Extract arrays
        X_train = train_prepared[feature_cols].values
        y_train = train_prepared['Personality_encoded'].values
        X_test = test_prepared[feature_cols].values
        
        # Clean arrays one more time
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"   Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"   Target distribution - Extrovert: {np.sum(y_train)}, Introvert: {len(y_train) - np.sum(y_train)}")
        
        # Cross-validation check
        print("3. Running cross-validation...")
        cv_score = cross_validate_simple(X_train, y_train)
        print(f"   CV Accuracy: {cv_score:.4f}")
        
        # Train final model
        print("4. Training final model...")
        final_model = train_simple_model(X_train, y_train)
        
        # Make predictions
        print("5. Making predictions...")
        predictions = final_model.predict(X_test)
        
        # Create submission
        print("6. Creating submission...")
        create_submission(test_prepared, predictions, "submissions/simple_submission.csv")
        
        print("=" * 50)
        print("SUCCESS: Submission file created successfully!")
        print(f"CV Score: {cv_score:.4f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()