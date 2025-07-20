#!/usr/bin/env python3
"""
Silver Layer Processing Script for CMI Sensor Data
Time-Series Feature Engineering & Multimodal Sensor Fusion

CLAUDE.md: Silver Layer Processing
- FFT/Statistical Features (tsfresh) 
- Multimodal Channel Fusion
- Time-series Feature Engineering
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data.silver import create_silver_tables

def main():
    """Execute Silver layer processing for CMI sensor data"""
    print(">H Silver Layer Processing for CMI Sensor Data")
    print("=" * 60)
    print("CLAUDE.md: Time-Series Feature Engineering & Multimodal Sensor Fusion")
    print()
    
    try:
        # Create Silver tables with CMI-specific feature engineering
        print("Processing Bronze -> Silver transformation...")
        print("Features:")
        print("   Time-series statistical features (IMU, ToF, Thermopile)")
        print("   Frequency domain features (FFT analysis)")  
        print("   Behavior-specific domain features (BFRB detection)")
        print("   tsfresh comprehensive features (memory-optimized)")
        print("   Multimodal sensor fusion")
        print()
        
        create_silver_tables()
        
        print()
        print(" Silver layer processing completed successfully!")
        print("=== Features created:")
        print("   - IMU sensor features (accelerometer + gyroscope)")
        print("   - ToF proximity patterns")
        print("   - Thermopile temperature distribution")
        print("   - Frequency domain spectral features")
        print("   - BFRB-specific behavioral patterns")
        print("   - tsfresh statistical features")
        print()
        print("<� Next step: Gold layer preparation (make gold)")
        
    except Exception as e:
        print(f"L Error in Silver layer processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()