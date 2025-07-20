#!/usr/bin/env python3
"""
Silver Layer Processing Script for Personality Data
Feature Engineering & Domain Knowledge Integration

CLAUDE.md: Silver Layer Processing
- Statistical Features & Interactions
- Domain-Specific Personality Features
- Feature Engineering & Scaling
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data.silver import create_silver_tables

def main():
    """Execute Silver layer processing for personality data"""
    print(">H Silver Layer Processing for Personality Data")
    print("=" * 60)
    print("CLAUDE.md: Feature Engineering & Domain Knowledge Integration")
    print()
    
    try:
        # Create Silver tables with personality-specific feature engineering
        print("Processing Bronze -> Silver transformation...")
        print("Features:")
        print("   Statistical features (mean, std, ratios)")
        print("   Interaction features (social × time patterns)")  
        print("   Domain-specific personality features")
        print("   Polynomial features (quadratic interactions)")
        print("   Scaling and normalization")
        print()
        
        create_silver_tables()
        
        print()
        print(" Silver layer processing completed successfully!")
        print("=== Features created:")
        print("   - Social activity patterns")
        print("   - Time allocation ratios")
        print("   - Communication efficiency metrics")
        print("   - Personality interaction features")
        print("   - Statistical transformations")
        print("   - Polynomial combinations")
        print()
        print("< Next step: Gold layer preparation (make gold)")
        
    except Exception as e:
        print(f"L Error in Silver layer processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()