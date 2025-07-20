#!/usr/bin/env python3
"""Check Bronze layer structure."""

from src.data.bronze import load_bronze_data

train_df, test_df = load_bronze_data()

print("Bronze layer columns:")
print("Train columns:", list(train_df.columns))
print("Train shape:", train_df.shape)
print("\nSample data:")
print(train_df.head()[['participant_id']])

# Check for series identifier
id_cols = [col for col in train_df.columns if 'id' in col.lower()]
print(f"\nID columns: {id_cols}")

# Check unique values
for col in id_cols:
    print(f"{col}: {train_df[col].nunique()} unique values")