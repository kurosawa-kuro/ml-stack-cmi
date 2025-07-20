# テストコード整理整頓の設計案

## 📊 現状分析サマリー
- **テストファイル数**: 27ファイル
- **主な問題**: 
  - 重複テストパターン多数
  - 過剰なモック使用（147箇所）
  - 実装されていない機能のテスト
  - コードの重複による保守性低下

## 🎯 整理目標
1. **カバレッジ維持**: 現在の72%カバレッジを維持
2. **重複排除**: 同一テスト内容の統合
3. **モック最適化**: 不要なモックの削除
4. **保守性向上**: 共通化によるメンテナンス簡素化

## 📋 段階的整理計画

### Phase 1: 重複ファイルの統合
```
削除対象ファイル（9個）:
├── test_data_enhanced.py     → test_data_comprehensive.py に統合
├── test_models_enhanced.py   → test_models_comprehensive.py に統合  
├── test_validation_enhanced.py → test_validation_comprehensive.py に統合
├── test_utilities_enhanced.py → test_util_comprehensive.py に統合
├── test_silver_gold_enhanced.py → test_silver_functions.py + test_gold_functions.py に統合
├── test_coverage_focused.py  → 各対応テストファイルに分散統合
├── test_data_real.py         → test_database_integration.py に統合
├── test_validation_real.py   → test_validation_comprehensive.py に統合
└── test_error_scenarios_integration.py → 各テストファイルにエラーケース統合
```

### Phase 2: 共通ユーティリティの作成
```python
# tests/conftest.py (新規作成)
@pytest.fixture
def sample_classification_data():
    """分類用テストデータ生成"""
    n_samples, n_features = 200, 10
    X = np.random.RandomState(42).random((n_samples, n_features))
    y = np.random.RandomState(42).randint(0, 2, n_samples)
    return X, y

@pytest.fixture  
def feature_names():
    """特徴量名生成"""
    return [f"feature_{i:02d}" for i in range(10)]

class MockHelpers:
    @staticmethod
    def create_duckdb_mock(train_data, test_data):
        """DuckDB接続の標準モック"""
        
    @staticmethod
    def create_lightgbm_mock():
        """LightGBMモデルの標準モック"""
```

### Phase 3: モック最適化
```python
# 削除対象モック例:
- 実装されていない機能のモック（42箇所）
- 過度に複雑なモック連鎖（15箇所）
- 実際のオブジェクトで代用可能なモック（23箇所）

# 保持対象モック:
- 外部API呼び出し（requests, webhooks）
- ファイルシステム操作
- データベース接続（必要な場合のみ）
```

## 🏗️ 最終的なテスト構造

```
tests/
├── conftest.py                     # 共通フィクスチャ・ユーティリティ
├── test_models.py                  # シンプル化されたモデルテスト
├── test_models_comprehensive.py   # 統合されたモデル包括テスト
├── test_validation.py              # 基本的なバリデーション
├── test_validation_comprehensive.py # 統合されたバリデーション包括テスト
├── test_data_comprehensive.py     # 統合されたデータ処理テスト
├── test_database_integration.py   # 実際のDB接続テスト
├── test_train_script.py           # トレーニングスクリプトテスト
├── test_notifications.py          # 通知システムテスト
├── test_time_tracker.py           # 時間追跡テスト
├── test_util_comprehensive.py     # 統合されたユーティリティテスト
├── test_bronze.py                  # ブロンズデータ処理
├── test_silver_functions.py       # シルバーデータ処理
├── test_gold_functions.py         # ゴールドデータ処理
├── test_integration_comprehensive.py # 包括的統合テスト
├── test_ml_pipeline_integration.py   # MLパイプライン統合テスト
├── test_pipeline_integration.py   # パイプライン統合テスト
└── test_notification_integration.py # 通知統合テスト

削減: 27ファイル → 18ファイル（33%削減）
```

## ✅ 実装ステップ

1. **conftest.py作成** - 共通フィクスチャ定義
2. **enhanced系ファイル削除** - 内容をcomprehensive系に統合  
3. **モック最適化** - 不要なモック削除・簡素化
4. **重複テスト統合** - 同一内容のテスト統合
5. **カバレッジ検証** - 整理後のカバレッジ確認

## 🎁 期待効果
- **ファイル数**: 27 → 18ファイル（33%削減）
- **重複コード**: 約60%削減見込み
- **モック箇所**: 147 → 約80箇所（45%削減）
- **保守性**: 大幅向上
- **カバレッジ**: 72%維持

## 📊 発見された問題の詳細

### 1. 重複している共通のテストパターン

#### テスト用データ生成パターン
```python
# 複数のファイルで同じパターン
n_samples, n_features = 200, 10
X = np.random.random((n_samples, n_features))
y = np.random.randint(0, 2, n_samples)
```

#### 特徴量名生成パターン
```python
# 複数のファイルで同じロジック
feature_names = [f"feature_{i}" for i in range(5)]
```

### 2. 類似のモックオブジェクト使用箇所

#### LightGBMモデルのモック
```python
@patch("lightgbm.LGBMClassifier")
def test_model_training(self, mock_lgb):
    mock_model = Mock()
    mock_lgb.return_value = mock_model
```

#### DuckDB接続のモック
```python
@patch("duckdb.connect")
def test_load_data(self, mock_connect):
    mock_conn = Mock()
    mock_connect.return_value = mock_conn
```

### 3. 同じテスト対象を複数のファイルでテストしている箇所

- Accuracy計算のテスト（3ファイルで重複）
- AUC計算のテスト（3ファイルで重複）
- CV結果検証パターン（5ファイルで類似）

### 4. 過剰なモックの例

#### 不必要な複雑なモック
```python
# 問題: モックのモックを作成
mock_conn.execute.side_effect = [
    Mock(df=Mock(return_value=train_data)),
    Mock(df=Mock(return_value=test_data)),
]
```

#### 実装されていない機能のモック
```python
# 問題: コメントアウトされたコードにモックが残存
# model = LightGBMModel()
# model.fit(X_train, y_train)
# mock_model.fit.assert_called_once()
```

#### テストの意図が不明確なモック
```python
# 問題: 実際の処理がなく、モックの動作だけをテスト
@patch("time.time")
def test_training_time_measurement(self, mock_time):
    mock_time.side_effect = [1000.0, 1180.0]
    training_time = end_time - start_time
    assert training_time == 180.0
```