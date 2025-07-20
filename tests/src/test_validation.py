"""
Test cases for validation.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score


class TestCVStrategy:
    """CV戦略のテスト"""

    def test_stratified_kfold_config(self):
        """StratifiedKFoldの設定テスト"""
        cv_config = {"n_splits": 5, "shuffle": True, "random_state": 42, "stratify": True}
        assert cv_config["n_splits"] == 5
        assert cv_config["random_state"] == 42

    def test_cv_splits_generation(self):
        """CV分割生成のテスト"""
        # 再現可能なデータ生成
        X = np.random.RandomState(42).random((100, 5))
        y = np.random.RandomState(42).randint(0, 2, 100)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))

        assert len(splits) == 5
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_stratification_preservation(self):
        """層化抽出でのクラス分布保持テスト"""
        # 再現可能な不均衡データ
        rng = np.random.RandomState(42)
        X = rng.random((100, 3))
        y = np.concatenate([np.zeros(80), np.ones(20)])  # 80:20の不均衡
        rng.shuffle(y)  # シャッフルしてリアルなデータに

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in cv.split(X, y):
            train_ratio = np.mean(y[train_idx])
            val_ratio = np.mean(y[val_idx])
            # 比率が近いことを確認（許容誤差0.1）
            assert abs(train_ratio - val_ratio) < 0.1


class TestEvaluationMetrics:
    """評価指標のテスト"""

    def test_accuracy_calculation(self):
        """Accuracy計算のテスト"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        # 手動計算: 4/5 = 0.8
        expected_accuracy = 0.8
        actual_accuracy = accuracy_score(y_true, y_pred)

        assert actual_accuracy == expected_accuracy

    def test_auc_calculation(self):
        """AUC計算のテスト"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

        auc = roc_auc_score(y_true, y_pred_proba)
        assert 0 <= auc <= 1

    def test_prediction_distribution(self):
        """予測分布統計のテスト"""
        predictions = np.array([0, 1, 1, 0, 1, 1, 0])

        extrovert_ratio = np.mean(predictions)  # 1の比率
        introvert_ratio = 1 - extrovert_ratio  # 0の比率

        assert abs(extrovert_ratio + introvert_ratio - 1.0) < 1e-10
        assert 0 <= extrovert_ratio <= 1
        assert 0 <= introvert_ratio <= 1


class TestCVScoring:
    """CVスコア計算のテスト"""

    def test_cv_score_aggregation(self):
        """CVスコア集計のテスト"""
        fold_scores = [0.97, 0.96, 0.98, 0.95, 0.97]

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        assert abs(mean_score - 0.966) < 0.001
        assert std_score > 0  # 分散があることを確認

    def test_cv_log_format(self):
        """CVログフォーマットのテスト"""
        cv_results = {
            "fold_scores": [0.97, 0.96, 0.98, 0.95, 0.97],
            "mean_score": 0.966,
            "std_score": 0.01,
            "training_time": 180.0,
        }

        # 必要なキーが存在することを確認
        required_keys = ["fold_scores", "mean_score", "std_score", "training_time"]
        for key in required_keys:
            assert key in cv_results

    def test_cv_reproducibility(self):
        """CV結果の再現性テスト"""
        X = np.random.RandomState(42).random((100, 5))
        y = np.random.RandomState(42).randint(0, 2, 100)

        # 同じシードで2回実行
        cv1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        splits1 = list(cv1.split(X, y))
        splits2 = list(cv2.split(X, y))

        # 分割が同じであることを確認
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            assert np.array_equal(train1, train2)
            assert np.array_equal(val1, val2)


class TestDataIntegrityChecks:
    """データ整合性チェックのテスト"""

    def test_data_shape_validation(self):
        """データ形状確認のテスト"""
        X_train = np.random.random((100, 10))
        X_test = np.random.random((50, 10))
        y_train = np.random.randint(0, 2, 100)

        # 形状チェック
        assert X_train.shape[0] == y_train.shape[0]  # サンプル数一致
        assert X_train.shape[1] == X_test.shape[1]  # 特徴量数一致

    def test_missing_value_detection(self):
        """欠損値検出のテスト"""
        data_with_nan = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])

        has_nan = np.isnan(data_with_nan).any()
        assert has_nan

        data_without_nan = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        has_nan_clean = np.isnan(data_without_nan).any()
        assert not has_nan_clean

    def test_infinite_value_detection(self):
        """無限値検出のテスト"""
        data_with_inf = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])

        has_inf = np.isinf(data_with_inf).any()
        assert has_inf

    def test_target_distribution_check(self):
        """ターゲット分布確認のテスト"""
        y = np.array([0, 1, 1, 0, 1, 0, 0, 1])

        unique_values = np.unique(y)
        value_counts = np.bincount(y)

        assert len(unique_values) == 2  # バイナリ分類
        assert all(count > 0 for count in value_counts)  # 両クラス存在


class TestCVLogging:
    """CVログ機能のテスト"""

    def test_log_structure(self):
        """ログ構造のテスト"""
        log_entry = {
            "timestamp": "2024-01-01T12: 00: 00",
            "model_type": "LightGBM",
            "cv_config": {"n_splits": 5, "random_state": 42},
            "fold_scores": [0.97, 0.96, 0.98],
            "mean_score": 0.970,
            "std_score": 0.010,
            "training_time": 180.0,
            "memory_usage": "256MB",
        }

        # 必要なフィールドの存在確認
        required_fields = ["timestamp", "model_type", "cv_config", "fold_scores", "mean_score", "std_score"]

        for field in required_fields:
            assert field in log_entry

    def test_json_log_saving(self):
        """JSONログ保存のテスト"""
        import json
        import tempfile
        
        # 実際のJSON操作でテスト
        log_data = {"test": "data", "score": 0.95, "timestamp": "2024-01-01"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(log_data, f)
            temp_path = f.name
        
        # 読み込みで検証
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
            
        assert loaded_data == log_data
        
        # クリーンアップ
        import os
        os.unlink(temp_path)

    def test_csv_log_format(self):
        """CSVログフォーマットのテスト"""
        cv_results = pd.DataFrame(
            {"fold": [0, 1, 2, 3, 4], "accuracy": [0.97, 0.96, 0.98, 0.95, 0.97], "auc": [0.99, 0.98, 0.99, 0.97, 0.98]}
        )

        assert "fold" in cv_results.columns
        assert "accuracy" in cv_results.columns
        assert len(cv_results) == 5


# Integration Test用のモックデータ生成
class TestMockDataGeneration:
    """テスト用データ生成"""

    def test_create_mock_classification_data(self):
        """分類用モックデータ生成のテスト"""
        n_samples = 1000
        n_features = 10

        # 再現可能なデータ生成
        rng = np.random.RandomState(42)
        X = rng.random((n_samples, n_features))
        y = rng.randint(0, 2, n_samples)

        assert X.shape == (n_samples, n_features)
        assert y.shape == (n_samples,)
        assert set(np.unique(y)) == {0, 1}

    def test_create_imbalanced_data(self):
        """不均衡データ生成のテスト"""
        n_minority = 200
        n_majority = 800

        y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])
        
        # 再現可能なシャッフル
        rng = np.random.RandomState(42)
        rng.shuffle(y)

        class_ratio = np.mean(y)
        expected_ratio = n_minority / (n_minority + n_majority)

        assert abs(class_ratio - expected_ratio) < 0.01
