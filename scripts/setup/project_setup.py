#!/usr/bin/env python3
"""
CMI Competition - Project Setup and Initialization
==================================================
This script initializes the project environment and validates prerequisites.

Features:
- Create necessary directories
- Validate data availability  
- Install required packages
- Environment configuration check
- Initial data connection test
"""

import os
import sys
import subprocess
from pathlib import Path
import importlib.util

def create_directories():
    """Create necessary project directories"""
    directories = [
        "data/processed",
        "data/features", 
        "data/models",
        "data/submissions",
        "logs",
        "outputs/figures",
        "outputs/reports",
        "cache"
    ]
    
    project_root = Path(__file__).parent.parent
    
    print("📁 Creating project directories...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")

def check_data_availability():
    """Check if competition data is available"""
    project_root = Path(__file__).parent.parent
    
    print("\n📊 Checking data availability...")
    
    # Check for DuckDB database
    db_path = project_root / "data" / "kaggle_datasets.duckdb"
    if db_path.exists():
        print(f"  ✓ DuckDB database found: {db_path}")
        return True
    
    # Check for CSV files
    csv_files = ["train.csv", "test.csv", "sample_submission.csv"]
    data_dir = project_root / "data"
    
    missing_files = []
    for csv_file in csv_files:
        csv_path = data_dir / csv_file
        if csv_path.exists():
            print(f"  ✓ Found: {csv_file}")
        else:
            missing_files.append(csv_file)
            print(f"  ✗ Missing: {csv_file}")
    
    if missing_files:
        print(f"\n⚠️  Missing data files: {missing_files}")
        print("Please download competition data from:")
        print("https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data")
        return False
    
    return True

def check_python_packages():
    """Check and install required Python packages"""
    required_packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "lightgbm",
        "tsfresh",
        "duckdb",
        "matplotlib",
        "seaborn",
        "plotly"
    ]
    
    print("\n📦 Checking Python packages...")
    
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
            print(f"  ✗ Missing: {package}")
        else:
            print(f"  ✓ Found: {package}")
    
    if missing_packages:
        print(f"\n⚠️  Installing missing packages: {missing_packages}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("  ✓ Packages installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Failed to install packages: {e}")
            print("  ℹ️  This is expected in WSL environments. Please install manually if needed:")
            print("     pip install --user " + " ".join(missing_packages))
            print("  ℹ️  Continuing with available packages...")
            # Don't return False - continue with available packages
            return True
    
    return True

def test_data_connection():
    """Test data loading functionality"""
    print("\n🔗 Testing data connection...")
    
    try:
        # Try importing our modules
        sys.path.append(str(Path(__file__).parent.parent))
        from src.data.bronze import load_data
        
        print("  ✓ Bronze layer import successful")
        
        # Try basic data loading (just headers)
        try:
            train_df, test_df = load_data()
            print(f"  ✓ Data loading test successful")
            print(f"    - Train shape: {train_df.shape}")
            print(f"    - Test shape: {test_df.shape}")
            
            # Check expected columns for CMI sensor data
            expected_cols = ['acc_x', 'acc_y', 'acc_z', 'subject', 'behavior']
            missing_cols = [col for col in expected_cols if col not in train_df.columns]
            if missing_cols:
                print(f"  ⚠️  Missing expected columns: {missing_cols}")
            else:
                print("  ✓ All expected sensor columns found")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Data loading failed: {e}")
            return False
            
    except ImportError as e:
        print(f"  ✗ Module import failed: {e}")
        return False

def display_next_steps():
    """Display recommended next steps"""
    print("\n🚀 Setup Complete! Recommended next steps:")
    print("")
    print("Week 1 - Foundation & Baseline:")
    print("  1. python scripts/workflow-to-eda.py          # Data exploration")
    print("  2. python scripts/workflow-to-data-check.py   # Data quality check")
    print("  3. python scripts/workflow-to-bronze.py       # Bronze layer processing")
    print("  4. python scripts/workflow-to-silver.py       # Feature engineering")
    print("  5. python scripts/workflow-to-training-lgb.py # Baseline model")
    print("")
    print("Documentation:")
    print("  - Project overview: README.md")
    print("  - Development guide: CLAUDE.md")
    print("  - Competition: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data")

def main():
    """Main setup workflow"""
    print("🎯 CMI Competition - Project Setup")
    print("=" * 50)
    
    # Create directories (always successful)
    create_directories()
    
    # Check data availability
    data_available = check_data_availability()
    
    # Check Python packages (continue even if installation fails)
    check_python_packages()
    
    # Test data connection (continue even if data is missing)
    if data_available:
        test_data_connection()
    else:
        print("\n⚠️  Skipping data connection test due to missing data files")
    
    print("\n" + "=" * 50)
    print("✅ Basic setup completed!")
    if not data_available:
        print("⚠️  Data files are missing - please download competition data")
        print("   Download from: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data")
    display_next_steps()

if __name__ == "__main__":
    main()