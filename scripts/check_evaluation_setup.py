#!/usr/bin/env python3
"""Check local evaluation setup and data format."""

import sys
import pandas as pd
sys.path.append('/home/wsl/dev/my-study/ml/ml-stack-cmi')

print("🔍 Checking Evaluation Setup")
print("=" * 30)

# Check kaggle_evaluation
try:
    from kaggle_evaluation.cmi_gateway import CMIGateway
    print("✅ Kaggle evaluation module available")
    
    # Check what the gateway expects
    gateway = CMIGateway()
    print(f"Target column: {gateway.target_column_name}")
    print(f"Target gestures: {gateway.target_gestures[:3]}...")
    print(f"Non-target gestures: {gateway.non_target_gestures[:3]}...")
    
except ImportError as e:
    print(f"❌ Kaggle evaluation not available: {e}")
except Exception as e:
    print(f"⚠️ Gateway error: {e}")

# Check our data structure
print("\n📊 Our Data Structure:")
from src.data.gold import load_gold_data
df_train, df_test = load_gold_data()

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

# Check column names for grouping
id_cols = [col for col in df_train.columns if 'id' in col.lower()]
print(f"ID columns in train: {id_cols}")

id_cols_test = [col for col in df_test.columns if 'id' in col.lower()]
print(f"ID columns in test: {id_cols_test}")

# Check behavior values
if 'behavior' in df_train.columns:
    print(f"\nBehavior values in train:")
    print(df_train['behavior'].value_counts())

# Check if we have gesture column (what gateway expects)
if 'gesture' in df_train.columns:
    print(f"\nGesture values in train:")
    print(df_train['gesture'].value_counts())

# Sample train data
print(f"\nSample train columns:")
print(list(df_train.columns)[:15])

print(f"\nSample test columns:")
print(list(df_test.columns)[:15])

# Try simple submission format
print("\n📤 Testing Simple Submission Format:")
# Create a simple submission with test IDs
if 'id' in df_test.columns:
    test_ids = df_test['id'].values
elif 'row_id' in df_test.columns:
    test_ids = df_test['row_id'].values
else:
    test_ids = range(len(df_test))

# Use most common behavior from train
if 'behavior' in df_train.columns:
    most_common_behavior = df_train['behavior'].mode()[0]
    print(f"Most common behavior: {most_common_behavior}")
    
    submission = pd.DataFrame({
        'id': test_ids,
        'behavior': [most_common_behavior] * len(test_ids)
    })
    
    print(f"Simple submission shape: {submission.shape}")
    print(submission.head())
    
    # Save for potential evaluation
    submission.to_parquet('/tmp/simple_submission.parquet', index=False)
    print("✅ Simple submission saved to /tmp/simple_submission.parquet")

print("\n🎯 Next Steps:")
print("1. Fix groupby column name (use participant_id or available ID)")
print("2. Test with simple baseline submission")
print("3. Run evaluation if system is properly configured")