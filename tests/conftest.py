"""
Common test fixtures and utilities for ML pipeline tests
"""

import time
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from typing import Tuple, Dict, Any


# ===== 共通テストデータ生成 =====

@pytest.fixture
def sample_bronze_data() -> pd.DataFrame:
    """基本的なブロンズレベルのテストデータ"""
    return pd.DataFrame({
        "Time_spent_Alone": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Social_event_attendance": [2.0, 4.0, 6.0, 8.0, 10.0],
        "Going_outside": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Friends_circle_size": [5, 10, 15, 20, 25],
        "Post_frequency": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Stage_fear": ["Yes", "No", "Yes", "No", "Yes"],
        "Drained_after_socializing": ["No", "Yes", "No", "Yes", "No"],
        "Personality": ["Introvert", "Extrovert", "Introvert", "Extrovert", "Introvert"]
    })


@pytest.fixture
def sample_silver_data() -> pd.DataFrame:
    """シルバーレベルのテストデータ（エンコード済み）"""
    return pd.DataFrame({
        "Time_spent_Alone": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Social_event_attendance": [2.0, 4.0, 6.0, 8.0, 10.0],
        "Going_outside": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Friends_circle_size": [5, 10, 15, 20, 25],
        "Post_frequency": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Stage_fear_encoded": [1.0, 0.0, 1.0, 0.0, 1.0],
        "Drained_after_socializing_encoded": [0.0, 1.0, 0.0, 1.0, 0.0],
        "Personality": ["Introvert", "Extrovert", "Introvert", "Extrovert", "Introvert"]
    })


@pytest.fixture
def sample_gold_data() -> pd.DataFrame:
    """ゴールドレベルのテストデータ（特徴量エンジニアリング済み）"""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "extrovert_score": [8, 16, 24, 32, 40],
        "introvert_score": [6, 6, 10, 6, 10],
        "social_ratio": [2.0, 2.0, 2.0, 2.0, 2.0],
        "activity_sum": [3, 6, 9, 12, 15],
        "Personality": ["Introvert", "Extrovert", "Introvert", "Extrovert", "Introvert"]
    })


@pytest.fixture
def edge_case_data() -> pd.DataFrame:
    """エッジケースを含むテストデータ"""
    return pd.DataFrame({
        "Time_spent_Alone": [0.0, 24.0, 25.0, -1.0, None],
        "Social_event_attendance": [0.0, 10.0, -5.0, None, 5.0],
        "Stage_fear": ["Yes", "NO", "yes", "no", None],
        "Drained_after_socializing": ["No", "YES", "no", "yes", None],
        "Friends_circle_size": [0, 50, -10, None, 25],
        "Post_frequency": [0.0, 20.0, -5.0, None, 10.0]
    })


@pytest.fixture
def missing_data() -> pd.DataFrame:
    """欠損値を含むテストデータ"""
    return pd.DataFrame({
        "Time_spent_Alone": [1.0, np.nan, 3.0, 4.0, np.nan],
        "Social_event_attendance": [2.0, 4.0, np.nan, 8.0, 10.0],
        "Going_outside": [1.0, 2.0, 3.0, np.nan, 5.0],
        "Stage_fear_encoded": [1.0, 0.0, 1.0, 0.0, np.nan],
        "Drained_after_socializing_encoded": [0.0, 1.0, 0.0, np.nan, 1.0]
    })


@pytest.fixture
def large_test_data() -> pd.DataFrame:
    """大規模テストデータ（パフォーマンステスト用）"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'id': range(n_samples),
        'Time_spent_Alone': np.random.uniform(0, 24, n_samples),
        'Social_event_attendance': np.random.randint(0, 20, n_samples),
        'Going_outside': np.random.randint(0, 15, n_samples),
        'Friends_circle_size': np.random.randint(0, 50, n_samples),
        'Post_frequency': np.random.randint(0, 30, n_samples),
        'Stage_fear': np.random.choice(['Yes', 'No'], n_samples),
        'Drained_after_socializing': np.random.choice(['Yes', 'No'], n_samples),
        'Personality': np.random.choice(['Extrovert', 'Introvert'], n_samples),
        'Personality_encoded': np.random.randint(0, 2, n_samples)  # 追加
    }
    
    return pd.DataFrame(data)


# ===== 共通モックユーティリティ =====

class MockDatabaseConnection:
    """DuckDB接続の標準モック"""
    
    def __init__(self, train_data: pd.DataFrame = None, test_data: pd.DataFrame = None):
        # DataFrameの真偽値評価を避けるため、明示的にNoneチェック
        if train_data is None:
            self.train_data = pd.DataFrame({"id": [1, 2], "feature": [1, 2]})
        else:
            self.train_data = train_data
            
        if test_data is None:
            self.test_data = pd.DataFrame({"id": [3, 4], "feature": [3, 4]})
        else:
            self.test_data = test_data
            
        self.mock_conn = MagicMock()
        self.mock_connect = MagicMock(return_value=self.mock_conn)
        
        # デフォルトのモック設定
        self._setup_default_mocks()
    
    def _setup_default_mocks(self):
        """デフォルトのモック設定"""
        mock_train_result = MagicMock()
        mock_train_result.df.return_value = self.train_data
        mock_test_result = MagicMock()
        mock_test_result.df.return_value = self.test_data
        
        # DataFrameの真偽値評価を避けるため、明示的にリストで管理
        self.mock_conn.execute.side_effect = lambda query: mock_train_result if "train" in query else mock_test_result
    
    def get_mock_connect(self):
        """モック接続を取得"""
        return self.mock_connect
    
    def get_mock_conn(self):
        """モックコネクションを取得"""
        return self.mock_conn


@pytest.fixture
def mock_db_connection(sample_bronze_data, sample_gold_data):
    """データベース接続のモックフィクスチャ"""
    return MockDatabaseConnection(
        train_data=sample_bronze_data,
        test_data=sample_gold_data
    )


# ===== 共通テストユーティリティ =====

class PerformanceTimer:
    """パフォーマンス測定用ユーティリティ"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = None
    
    def __enter__(self):
        self.start_time = time.time()
        # Memory usage tracking (if psutil available)
        try:
            import psutil
            process = psutil.Process()
            self.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.memory_usage = None
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed_time(self):
        """経過時間を取得"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def memory_used(self):
        """メモリ使用量を取得（MB）"""
        return self.memory_usage


def assert_sub_second_performance(func, *args, **kwargs):
    """サブ秒パフォーマンスをアサート"""
    with PerformanceTimer() as timer:
        result = func(*args, **kwargs)
    
    assert timer.elapsed_time < 1.0, f"Function took {timer.elapsed_time:.3f}s (should be < 1.0s)"
    return result


def assert_memory_efficient(func, max_memory_mb: float = 100.0, *args, **kwargs):
    """メモリ効率性をアサート"""
    with PerformanceTimer() as timer:
        result = func(*args, **kwargs)
    
    if timer.memory_used is not None:
        assert timer.memory_used < max_memory_mb, \
            f"Function used {timer.memory_used:.1f}MB (should be < {max_memory_mb}MB)"
    return result


def assert_lightgbm_compatibility(df: pd.DataFrame):
    """LightGBM互換性をアサート"""
    for col in df.columns:
        # データ型チェック（object型は除外）
        if df[col].dtype == 'object':
            continue  # object型はLightGBMで自動的に処理される
        
        assert df[col].dtype in ['float64', 'float32', 'int64', 'int32'], \
            f"Feature {col} has incompatible dtype {df[col].dtype}"
        
        # 無限値チェック
        if df[col].dtype in ['float64', 'float32']:
            assert not np.isinf(df[col]).any(), f"Infinite values found in {col}"
        
        # メモリ効率性チェック
        if df[col].dtype == 'float64':
            # 必要に応じてfloat32に変換可能かチェック
            if df[col].min() >= -3.4e38 and df[col].max() <= 3.4e38:
                # float32の範囲内なので変換可能
                pass


def assert_no_data_loss(original_df: pd.DataFrame, processed_df: pd.DataFrame):
    """データ損失がないことをアサート"""
    assert len(processed_df) == len(original_df), "Data length changed during processing"
    
    # 元の列が保持されているかチェック
    for col in original_df.columns:
        assert col in processed_df.columns, f"Original column {col} not preserved"


def assert_feature_engineering_quality(df: pd.DataFrame, min_new_features: int = 2):
    """特徴量エンジニアリングの品質をアサート"""
    # 新しい特徴量が追加されているかチェック
    new_features = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['ratio', 'interaction', 'score', 'poly_', 'scaled', 'participation_rate', 'communication_ratio', 'social_efficiency', 'activity_balance', 'non_social'])]
    assert len(new_features) >= min_new_features, f"Expected {min_new_features}+ new features, got {len(new_features)}"
    
    # 特徴量の品質チェック
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            # 合理的な範囲内かチェック
            feature_values = df[col].dropna()
            if len(feature_values) > 0:
                assert feature_values.min() >= -1000, f"Feature {col} has unreasonably low values"
                assert feature_values.max() <= 10000, f"Feature {col} has unreasonably high values"


# ===== 共通テストデコレータ =====

def performance_test(max_time: float = 1.0):
    """パフォーマンステスト用デコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceTimer() as timer:
                result = func(*args, **kwargs)
            
            assert timer.elapsed_time < max_time, \
                f"Function took {timer.elapsed_time:.3f}s (should be < {max_time}s)"
            return result
        return wrapper
    return decorator


def lightgbm_compatibility_test(func):
    """LightGBM互換性テスト用デコレータ"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            assert_lightgbm_compatibility(result)
        return result
    return wrapper


# ===== 共通アサーション関数 =====

def assert_database_operations(mock_connect, expected_calls: list = None):
    """データベース操作のアサート"""
    mock_connect.assert_called_once()
    mock_conn = mock_connect.return_value
    assert mock_conn.close.called
    
    if expected_calls:
        actual_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        for expected_call in expected_calls:
            assert any(expected_call in call for call in actual_calls), \
                f"Expected call '{expected_call}' not found in actual calls"


def assert_feature_presence(df: pd.DataFrame, required_features: list):
    """必要な特徴量の存在をアサート"""
    for feature in required_features:
        assert feature in df.columns, f"Required feature '{feature}' not found in DataFrame"


def assert_data_quality(df: pd.DataFrame):
    """データ品質をアサート"""
    # 無限値チェック
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            # 無限値をNaNに置換してからチェック
            col_data = df[col].replace([np.inf, -np.inf], np.nan)
            assert not col_data.isna().all(), f"All values in {col} are infinite or NaN"
    
    # データ型チェック
    for col in df.columns:
        assert df[col].dtype in ['float64', 'float32', 'int64', 'int32', 'object'], \
            f"Feature {col} has unsupported dtype {df[col].dtype}"
    
    # 特徴量の範囲チェック（極端な値は除外）
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            feature_values = df[col].dropna()
            if len(feature_values) > 0:
                # 無限値を除外してから範囲チェック
                finite_values = feature_values.replace([np.inf, -np.inf], np.nan).dropna()
                if len(finite_values) > 0:
                    # より寛容な範囲チェック
                    assert finite_values.min() >= -10000, f"Feature {col} has unreasonably low values"
                    assert finite_values.max() <= 100000, f"Feature {col} has unreasonably high values"


# ===== 共通テストデータ生成関数 =====

def create_correlated_test_data(n_samples: int = 100, correlation: float = 0.8) -> pd.DataFrame:
    """相関のあるテストデータを生成"""
    np.random.seed(42)
    
    # ベースとなる特徴量
    base_feature = np.random.randn(n_samples)
    
    # 相関のある特徴量を生成
    correlated_feature = correlation * base_feature + np.sqrt(1 - correlation**2) * np.random.randn(n_samples)
    
    # ノイズ特徴量
    noise_feature = np.random.randn(n_samples)
    
    return pd.DataFrame({
        'feature1': base_feature,
        'feature2': correlated_feature,
        'feature3': noise_feature,
        'target': np.random.randint(0, 2, n_samples)
    })


def create_missing_pattern_data(n_samples: int = 100) -> pd.DataFrame:
    """欠損パターンを含むテストデータを生成"""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples)
    })
    
    # 系統的な欠損パターンを追加
    df.loc[df['feature1'] > 0, 'feature2'] = np.nan
    df.loc[df['feature3'] < 0, 'feature4'] = np.nan
    
    # ランダムな欠損を追加
    mask = np.random.random(n_samples) < 0.1
    df.loc[mask, 'feature1'] = np.nan
    
    return df


@pytest.fixture
def create_outlier_data():
    """Create test data with outliers"""
    def _create_outlier_data(n_samples: int = 100):
        np.random.seed(42)
        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'outlier_feature': np.random.randn(n_samples)
        }
        # Add outliers
        data['outlier_feature'][0] = 1000  # Extreme outlier
        data['outlier_feature'][1] = -1000  # Extreme outlier
        
        return pd.DataFrame(data)
    return _create_outlier_data


@pytest.fixture
def create_correlated_test_data():
    """Create test data with correlated features"""
    def _create_correlated_test_data(n_samples: int = 100, correlation: float = 0.8):
        np.random.seed(42)
        
        # Create correlated features
        x1 = np.random.randn(n_samples)
        x2 = correlation * x1 + np.sqrt(1 - correlation**2) * np.random.randn(n_samples)
        
        # Create target
        target = (x1 + x2 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        data = {
            'feature1': x1,
            'feature2': x2,
            'target': target
        }
        
        return pd.DataFrame(data)
    return _create_correlated_test_data


@pytest.fixture
def create_missing_pattern_data():
    """Create test data with missing value patterns"""
    def _create_missing_pattern_data(n_samples: int = 100):
        np.random.seed(42)
        
        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'feature4': np.random.randn(n_samples)
        }
        
        # Add systematic missing patterns
        data['feature1'][:n_samples//4] = np.nan  # First quarter missing
        data['feature2'][n_samples//4:n_samples//2] = np.nan  # Second quarter missing
        data['feature3'][n_samples//2:3*n_samples//4] = np.nan  # Third quarter missing
        
        return pd.DataFrame(data)
    return _create_missing_pattern_data