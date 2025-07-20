"""
Smoke tests for enhanced Silver features CV performance
Bronze medal target validation
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.quick_cv_enhanced import run_quick_cv_with_enhanced_features


class TestEnhancedSilverSmokeTests:
    """Smoke tests for enhanced Silver layer performance"""
    
    def test_enhanced_silver_cv_performance_small(self):
        """Bronze medal target向けスモークテスト - 小サンプル"""
        # Expected: 0.9684 → 0.976518 (+0.8%)
        # Use small sample for speed in CI
        score = run_quick_cv_with_enhanced_features(sample_ratio=0.05, folds=2)
        
        # Should be significantly better than random (0.5) and baseline expectation
        assert score > 0.90, f"Score {score:.6f} too low - possible implementation issue"
        assert score < 1.0, f"Score {score:.6f} suspiciously high - possible data leakage"
        
        print(f"✅ Small sample CV: {score:.6f}")
    
    def test_enhanced_silver_cv_performance_medium(self):
        """Bronze medal target向けスモークテスト - 中サンプル"""
        # Use medium sample for more reliable estimate
        score = run_quick_cv_with_enhanced_features(sample_ratio=0.1, folds=3)
        
        # Should approach Bronze medal performance
        assert score > 0.95, f"Score {score:.6f} below expected enhanced performance"
        
        # Bronze medal stretch goal
        bronze_target = 0.976518
        gap_to_bronze = bronze_target - score
        
        print(f"✅ Medium sample CV: {score:.6f}")
        print(f"🎯 Gap to Bronze: {gap_to_bronze:+.6f}")
        
        # Warn if far from Bronze target
        if gap_to_bronze > 0.01:
            pytest.warn(f"Gap to Bronze ({gap_to_bronze:+.6f}) larger than expected +0.008")
    
    @pytest.mark.slow
    def test_enhanced_silver_cv_performance_full(self):
        """Bronze medal target向けスモークテスト - フルパフォーマンス"""
        # Full performance test - marked as slow
        score = run_quick_cv_with_enhanced_features(sample_ratio=0.2, folds=5)
        
        # Should be close to Bronze medal target
        bronze_target = 0.976518
        assert score > 0.96, f"Score {score:.6f} significantly below Bronze target"
        
        gap_to_bronze = bronze_target - score
        print(f"✅ Large sample CV: {score:.6f}")
        print(f"🎯 Gap to Bronze: {gap_to_bronze:+.6f}")
        
        # Success criteria for Bronze medal approach
        if score >= 0.975:
            print("🏆 VERY CLOSE TO BRONZE MEDAL!")
        elif score >= 0.970:
            print("🎯 Good progress toward Bronze medal")
        else:
            print("⚠️  Need further optimization for Bronze medal")
    
    def test_quick_cv_subprocess_integration(self):
        """Test subprocess integration like the original memo"""
        try:
            # Run as subprocess to test CLI integration
            result = subprocess.run(
                [sys.executable, "scripts/quick_cv_enhanced.py", "--sample", "0.03", "--folds", "2"],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
            
            # Extract score from output
            lines = result.stdout.strip().split('\n')
            last_line = lines[-1].strip()
            
            try:
                score = float(last_line)
            except ValueError:
                # If last line isn't a float, look for CV Result line
                cv_lines = [line for line in lines if "CV Result:" in line]
                if cv_lines:
                    score_str = cv_lines[-1].split("CV Result:")[1].split("±")[0].strip()
                    score = float(score_str)
                else:
                    pytest.fail(f"Could not parse score from output: {result.stdout}")
            
            assert score > 0.90, f"Subprocess CV score {score:.6f} too low"
            print(f"✅ Subprocess CV: {score:.6f}")
            
        except subprocess.TimeoutExpired:
            pytest.fail("Quick CV test timed out - performance issue")
        except FileNotFoundError:
            pytest.skip("quick_cv_enhanced.py script not found")
    
    def test_feature_expansion_ratio(self):
        """Test that enhanced features significantly expand feature space"""
        # Load a small sample to test feature expansion
        from src.data.bronze import load_data
        from src.data.silver import EnhancedSilverPreprocessor
        
        train_data, _ = load_data()
        sample_data = train_data.sample(n=100, random_state=42)
        
        X_base = sample_data.drop(['id', 'Personality'], axis=1, errors='ignore')
        y = (sample_data['Personality'] == 'Extrovert').astype(int)
        
        X_enhanced = apply_enhanced_silver_features(X_base, y, is_train=True)
        
        expansion_ratio = X_enhanced.shape[1] / X_base.shape[1]
        
        print(f"Feature expansion: {X_base.shape[1]} → {X_enhanced.shape[1]} ({expansion_ratio:.1f}x)")
        
        # Should significantly expand features for enhanced performance
        assert expansion_ratio >= 10, f"Feature expansion {expansion_ratio:.1f}x too low for enhanced performance"
        assert expansion_ratio <= 100, f"Feature expansion {expansion_ratio:.1f}x suspiciously high - possible feature explosion"
    
    def test_performance_timing(self):
        """Test that enhanced features maintain reasonable performance"""
        start_time = time.time()
        
        # Small but meaningful test
        score = run_quick_cv_with_enhanced_features(sample_ratio=0.02, folds=2)
        
        elapsed = time.time() - start_time
        
        print(f"Timing: {elapsed:.2f}s for 2% sample, 2-fold CV")
        
        # Should complete quickly for small samples
        assert elapsed < 60, f"Performance too slow: {elapsed:.2f}s for small sample"
        assert score > 0.85, f"Score {score:.6f} unexpectedly low for timing test"


if __name__ == "__main__":
    # Run a quick test when executed directly
    print("Running enhanced Silver smoke test...")
    score = run_quick_cv_with_enhanced_features(sample_ratio=0.05, folds=2)
    print(f"Quick smoke test result: {score:.6f}")
    
    bronze_target = 0.976518
    if score >= 0.975:
        print("🏆 EXCELLENT - Very close to Bronze medal!")
    elif score >= 0.970:
        print("🎯 GOOD - Strong progress toward Bronze medal")  
    elif score >= 0.960:
        print("✅ DECENT - On track for Bronze medal")
    else:
        print("⚠️  NEEDS WORK - Requires optimization for Bronze medal")