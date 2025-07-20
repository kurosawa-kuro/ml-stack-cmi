#!/usr/bin/env python3
"""
Enhanced Prediction Script Using Gold Layer
Uses the medallion architecture (Bronze -> Silver -> Gold) for better features
Handles nan/inf issues and generates high-quality submission
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


def ensure_medallion_data():
    """Ensure Bronze, Silver, and Gold tables exist"""
    try:
        # Try importing and creating medallion layers
        from src.data.bronze import create_bronze_tables
        from src.data.silver import create_silver_tables
        from src.data.gold import create_gold_tables
        
        print("Creating medallion data layers...")
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        print("✓ Medallion data layers created successfully")
        
    except ImportError as e:
        print(f"Warning: Could not import medallion modules: {e}")
        print("Fallback: Using Bronze data only")
        return False
    except Exception as e:
        print(f"Warning: Could not create medallion data: {e}")
        print("Fallback: Using existing data")
        return False
    
    return True


def load_gold_data_safe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Gold layer data with fallback to Bronze"""
    conn = duckdb.connect(DB_PATH)
    
    try:
        # Try Gold layer first
        train_df = conn.execute("SELECT * FROM gold.train").df()
        test_df = conn.execute("SELECT * FROM gold.test").df()
        print("✓ Loaded data from Gold layer")
        data_source = "gold"
        
    except:
        try:
            # Fallback to Silver layer
            train_df = conn.execute("SELECT * FROM silver.train").df()
            test_df = conn.execute("SELECT * FROM silver.test").df()
            print("✓ Loaded data from Silver layer")
            data_source = "silver"
        except:
            try:
                # Fallback to Bronze layer
                train_df = conn.execute("SELECT * FROM bronze.train").df()
                test_df = conn.execute("SELECT * FROM bronze.test").df()
                print("✓ Loaded data from Bronze layer")
                data_source = "bronze"
            except:
                # Final fallback to raw data
                train_df = conn.execute("SELECT * FROM playground_series_s5e7.train").df()
                test_df = conn.execute("SELECT * FROM playground_series_s5e7.test").df()
                print("✓ Loaded data from raw tables")
                data_source = "raw"
    
    conn.close()
    return train_df, test_df, data_source


def clean_advanced_data(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced data cleaning for engineered features"""
    df = df.copy()
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Get numeric columns (excluding id and target)
    exclude_cols = ['id', 'Personality', 'Personality_encoded']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Advanced missing value handling
    for col in numeric_cols:
        if df[col].isnull().any():
            # Use different strategies based on column type
            if 'ratio' in col.lower() or 'per' in col.lower():
                # For ratio features, use 0 as default
                df[col] = df[col].fillna(0.0)
            elif 'score' in col.lower():
                # For score features, use median
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            else:
                # For other features, use median
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
    
    # Ensure all numeric values are clean
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        # Clip extreme values for numerical stability
        df[col] = df[col].clip(-1e8, 1e8)
    
    return df


def get_best_features(df: pd.DataFrame) -> list:
    """Get the best features based on data source"""
    exclude_cols = ['id', 'Personality', 'Personality_encoded']
    
    # All available numeric features
    all_features = [col for col in df.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Priority features based on known importance
    priority_features = [
        # Original features
        'Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
        'Friends_circle_size', 'Post_frequency',
        
        # Encoded categorical
        'Stage_fear_encoded', 'Drained_after_socializing_encoded',
        
        # Missing indicators
        'Stage_fear_missing', 'Going_outside_missing', 'Time_spent_Alone_missing',
        
        # Basic engineered features
        'social_ratio', 'activity_sum', 'post_per_friend',
        'extrovert_score', 'introvert_score',
        
        # Advanced engineered features (if available)
        'Social_event_participation_rate', 'Non_social_outings',
        'Communication_ratio', 'Drain_adjusted_activity',
        'Friend_social_efficiency', 'Activity_ratio',
        'Introvert_extrovert_spectrum', 'Communication_balance',
        
        # Polynomial features (if available)
        'poly_extrovert_score_Post_frequency', 'poly_extrovert_score_Social_event_attendance',
        'poly_social_ratio_Post_frequency', 'poly_activity_sum_extrovert_score'
    ]
    
    # Select features that actually exist in the data
    selected_features = [f for f in priority_features if f in all_features]
    
    # Add any remaining features up to a reasonable limit
    remaining_features = [f for f in all_features if f not in selected_features]
    max_additional = max(0, min(50, 50 - len(selected_features)))
    selected_features.extend(remaining_features[:max_additional])
    
    return selected_features


def train_enhanced_model(X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
    """Train an enhanced LightGBM model with optimized parameters"""
    
    # Enhanced parameters for better performance
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        num_leaves=50,
        learning_rate=0.08,
        max_depth=8,
        feature_fraction=0.8,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_estimators=200,
        verbosity=-1,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def cross_validate_enhanced(X: np.ndarray, y: np.ndarray) -> dict:
    """Enhanced cross-validation with detailed metrics"""
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    feature_importances = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = train_enhanced_model(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)
        scores.append(score)
        
        # Collect feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importances.append(model.feature_importances_)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores,
        'feature_importances': np.mean(feature_importances, axis=0) if feature_importances else None
    }


def create_enhanced_submission(test_df: pd.DataFrame, predictions: np.ndarray, 
                             cv_results: dict, filename: str = "enhanced_submission.csv") -> None:
    """Create enhanced submission file with metadata"""
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Extrovert' if pred == 1 else 'Introvert' for pred in predictions]
    })
    
    # Save to file
    submission_df.to_csv(filename, index=False)
    
    # Print detailed results
    print(f"✓ Enhanced submission saved to {filename}")
    print(f"✓ CV Score: {cv_results['mean_score']:.6f} ± {cv_results['std_score']:.6f}")
    print(f"✓ Prediction distribution:")
    print(submission_df['Personality'].value_counts())
    
    # Bronze medal status
    bronze_target = 0.976518
    gap_to_bronze = bronze_target - cv_results['mean_score']
    print(f"✓ Gap to Bronze Medal: {gap_to_bronze:+.6f}")
    
    if gap_to_bronze <= 0:
        print("🏆 BRONZE MEDAL ACHIEVED!")
    elif gap_to_bronze <= 0.005:
        print("🎯 Very close to Bronze Medal!")
    else:
        print("📈 Good progress towards Bronze Medal")


def main():
    """Main enhanced prediction workflow"""
    print("=" * 60)
    print("Enhanced Prediction Script - Gold Layer Features")
    print("=" * 60)
    
    try:
        # Ensure medallion data exists
        print("1. Setting up medallion data architecture...")
        medallion_success = ensure_medallion_data()
        
        # Load data
        print("2. Loading data...")
        train_df, test_df, data_source = load_gold_data_safe()
        print(f"   Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        print(f"   Data source: {data_source} layer")
        
        # Clean data
        print("3. Cleaning and preparing data...")
        train_clean = clean_advanced_data(train_df)
        test_clean = clean_advanced_data(test_df)
        
        # Get features
        feature_cols = get_best_features(train_clean)
        print(f"   Using {len(feature_cols)} features")
        print(f"   Top 10 features: {feature_cols[:10]}")
        
        # Prepare target
        if 'Personality_encoded' in train_clean.columns:
            y_train = train_clean['Personality_encoded'].values
        elif 'Personality' in train_clean.columns:
            y_train = (train_clean['Personality'] == 'Extrovert').astype(int).values
        else:
            raise ValueError("No target column found")
        
        # Extract feature arrays
        X_train = train_clean[feature_cols].values
        X_test = test_clean[feature_cols].values
        
        # Final data cleaning
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"   Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"   Target distribution - Extrovert: {np.sum(y_train)}, Introvert: {len(y_train) - np.sum(y_train)}")
        
        # Cross-validation
        print("4. Running enhanced cross-validation...")
        cv_results = cross_validate_enhanced(X_train, y_train)
        
        # Train final model
        print("5. Training final enhanced model...")
        final_model = train_enhanced_model(X_train, y_train)
        
        # Make predictions
        print("6. Making predictions...")
        predictions = final_model.predict(X_test)
        
        # Create submission
        print("7. Creating enhanced submission...")
        submission_filename = f"submissions/enhanced_{data_source}_submission.csv"
        create_enhanced_submission(test_clean, predictions, cv_results, submission_filename)
        
        # Feature importance analysis
        if cv_results['feature_importances'] is not None:
            print("\n📊 Top 10 Feature Importances:")
            importance_pairs = list(zip(feature_cols, cv_results['feature_importances']))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            for i, (feat, imp) in enumerate(importance_pairs[:10]):
                print(f"   {i+1}. {feat}: {imp:.1f}")
        
        print("\n" + "=" * 60)
        print("SUCCESS: Enhanced submission created successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()