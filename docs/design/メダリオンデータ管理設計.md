# ブロンズ、シルバー、ゴールド級データ管理パターンの段階的設計

## 概要
Kaggle S5E7 Personality Prediction コンペティション向けの段階的データ管理アーキテクチャ。
プロジェクトの成長に応じてブロンズ→シルバー→ゴールドへスケールアップ可能な設計。

## アーキテクチャ方針

### 設計原則
1. **段階的スケーラビリティ**: 必要に応じてレベルアップ
2. **下位互換性**: 上位レベルでも下位機能を利用可能
3. **明確な責任分離**: 各レベルで明確な役割定義
4. **実装コスト最適化**: 必要最小限から開始

## ファイル構成
```
src/data/
├── bronze.py          # レベル1: シンプル・直接的
├── silver.py          # レベル2: 構造化・再利用可能
├── gold.py            # レベル3: 企業級・フル機能
└── __init__.py        # 統合インターフェース
```

## レベル別詳細設計

### Bronze レベル (bronze.py)
**目的**: 高速プロトタイピング・最小限の実装

#### 特徴
- **シンプル**: 関数ベース、最小限のクラス
- **直接的**: DuckDBから直接読み込み
- **高速開発**: インライン処理、最小限の抽象化
- **適用場面**: 初期探索、ベースライン構築

#### 実装パターン
```python
# シンプルな関数ベース
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """DuckDBから直接データ読み込み"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM playground_series_s5e7.train").df()
    test = conn.execute("SELECT * FROM playground_series_s5e7.test").df()
    return train, test

def quick_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """最小限の前処理（インライン）"""
    # 欠損値処理
    df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].median())
    # カテゴリ変換
    df['Stage_fear_encoded'] = (df['Stage_fear'] == 'Yes').astype(int)
    return df

def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """基本的な特徴量生成"""
    df['social_ratio'] = df['Social_event_attendance'] / (df['Time_spent_Alone'] + 1)
    df['activity_sum'] = df['Going_outside'] + df['Social_event_attendance']
    return df
```

#### 利用例
```python
from src.data.bronze import load_data, quick_preprocess, basic_features

# 使用方法
train, test = load_data()
train = quick_preprocess(train)
train = basic_features(train)
```

### Silver レベル (silver.py)
**目的**: 構造化された再利用可能な中規模システム

#### 特徴
- **構造化**: クラスベース設計
- **再利用性**: モジュール化された処理
- **設定可能**: パラメータ調整可能
- **適用場面**: 本格的なモデル開発、チーム開発

#### 実装パターン
```python
class DataPipeline:
    """構造化されたデータ処理パイプライン"""

    def __init__(self, db_path: str, config: Dict = None):
        self.db_path = db_path
        self.config = config or {}
        self.conn = duckdb.connect(db_path)

    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Raw データ読み込み"""
        train = self.conn.execute("SELECT * FROM playground_series_s5e7.train").df()
        test = self.conn.execute("SELECT * FROM playground_series_s5e7.test").df()
        return train, test

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """設定可能な前処理"""
        strategy = self.config.get('missing_strategy', 'median')
        # 設定に基づく処理
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量エンジニアリング"""
        features = self.config.get('features', ['basic'])
        # 設定に基づく特徴量生成
        return df

class FeatureStore:
    """特徴量の保存・管理"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)

    def save_features(self, df: pd.DataFrame, name: str):
        """特徴量セットの保存"""
        df.to_sql(f"features_{name}", self.conn, if_exists='replace')

    def load_features(self, name: str) -> pd.DataFrame:
        """特徴量セットの読み込み"""
        return self.conn.execute(f"SELECT * FROM features_{name}").df()
```

#### 利用例
```python
from src.data.silver import DataPipeline, FeatureStore

# 設定ベースの使用
config = {
    'missing_strategy': 'median',
    'features': ['basic', 'interaction']
}

pipeline = DataPipeline(DB_PATH, config)
train, test = pipeline.load_raw()
train = pipeline.preprocess(train)
train = pipeline.engineer_features(train)

# 特徴量保存
store = FeatureStore(DB_PATH)
store.save_features(train, 'v1_basic')
```

### Gold レベル (gold.py)
**目的**: 企業級の完全機能データ管理システム

#### 特徴
- **バージョン管理**: データの履歴追跡
- **キャッシュシステム**: 計算結果の再利用
- **メタデータ追跡**: 処理過程の完全記録
- **監査機能**: 変更履歴とトレーサビリティ
- **設定管理**: YAML/JSON設定ファイル対応
- **適用場面**: 本番環境、大規模チーム、継続的運用

#### 実装パターン
```python
class DataManager:
    """企業級データ管理システム"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.db_path = self.config['database']['path']
        self.conn = duckdb.connect(self.db_path)
        self.cache = CacheManager(self.config['cache'])
        self.versioning = VersionManager(self.conn)
        self.metadata = MetadataTracker(self.conn)

    def get_data(self,
                 version: str = "latest",
                 cache: bool = True,
                 features: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """バージョン管理・キャッシュ付きデータ取得"""

        cache_key = f"data_{version}_{hash(str(features))}"

        if cache and self.cache.exists(cache_key):
            return self.cache.load(cache_key)

        # データ処理
        train, test = self._process_data(version, features)

        if cache:
            self.cache.save(cache_key, (train, test))

        # メタデータ記録
        self.metadata.record_access(version, features, cache_key)

        return train, test

    def create_feature_version(self,
                              features_func: Callable,
                              version_name: str,
                              description: str = ""):
        """新しい特徴量バージョンの作成"""

        # 処理実行
        train, test = self._apply_features(features_func)

        # バージョン保存
        version_id = self.versioning.create_version(
            version_name, description, features_func.__name__
        )

        # データ保存
        self._save_versioned_data(train, test, version_id)

        return version_id

class CacheManager:
    """インテリジェントキャッシュシステム"""

    def __init__(self, config: Dict):
        self.cache_dir = config['directory']
        self.max_size = config['max_size_gb']
        self.ttl = config['ttl_hours']

    def exists(self, key: str) -> bool:
        """キャッシュ存在確認（TTL考慮）"""
        pass

    def save(self, key: str, data: Any) -> None:
        """データキャッシュ保存"""
        pass

    def load(self, key: str) -> Any:
        """キャッシュデータ読み込み"""
        pass

class VersionManager:
    """データバージョン管理"""

    def __init__(self, conn):
        self.conn = conn
        self._init_version_tables()

    def create_version(self, name: str, description: str, features: str) -> str:
        """新バージョン作成"""
        pass

    def list_versions(self) -> List[Dict]:
        """バージョン一覧取得"""
        pass

    def get_version_info(self, version: str) -> Dict:
        """バージョン詳細情報"""
        pass

class MetadataTracker:
    """メタデータ・監査ログ管理"""

    def __init__(self, conn):
        self.conn = conn
        self._init_metadata_tables()

    def record_access(self, version: str, features: List[str], cache_key: str):
        """データアクセス記録"""
        pass

    def record_transformation(self, input_data: str, output_data: str, transform: str):
        """データ変換記録"""
        pass

    def get_lineage(self, data_id: str) -> Dict:
        """データ系譜追跡"""
        pass
```

#### 利用例
```python
from src.data.gold import DataManager

# 設定ファイル使用
manager = DataManager("configs/data_config.yaml")

# バージョン指定でデータ取得
train, test = manager.get_data(
    version="v2.1",
    cache=True,
    features=['basic', 'advanced', 'interaction']
)

# 新特徴量バージョン作成
def my_features(df):
    # カスタム特徴量
    return df

version_id = manager.create_feature_version(
    my_features,
    "v2.2_custom",
    "カスタム交互作用特徴量追加"
)
```

## 統合インターフェース (__init__.py)

### レベル選択機能
```python
from .bronze import load_data as bronze_load, quick_preprocess, basic_features
from .silver import DataPipeline, FeatureStore
from .gold import DataManager

def get_data_loader(level: str = "bronze", **kwargs):
    """レベル選択可能なデータローダー"""

    if level == "bronze":
        return {
            'load': bronze_load,
            'preprocess': quick_preprocess,
            'features': basic_features
        }
    elif level == "silver":
        return DataPipeline(**kwargs)
    elif level == "gold":
        return DataManager(**kwargs)
    else:
        raise ValueError(f"Unsupported level: {level}")

# 便利関数
def quick_start(level: str = "bronze") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """レベル別クイックスタート"""

    if level == "bronze":
        train, test = bronze_load()
        train = quick_preprocess(train)
        train = basic_features(train)
        test = quick_preprocess(test)
        test = basic_features(test)
        return train, test

    # 他のレベルの処理
```

### 使用例
```python
# レベル別使用法
from src.data import get_data_loader, quick_start

# Bronze: 最速開始
train, test = quick_start("bronze")

# Silver: 構造化開発
pipeline = get_data_loader("silver", db_path=DB_PATH, config=my_config)
train, test = pipeline.load_raw()

# Gold: フル機能
manager = get_data_loader("gold", config_path="configs/data.yaml")
train, test = manager.get_data(version="latest")
```

## 移行戦略

### Phase 1: Bronze 実装
1. 基本的な関数ベース実装
2. 最小限の前処理・特徴量生成
3. 高速プロトタイピング環境構築

### Phase 2: Silver 拡張
1. クラスベース構造化
2. 設定可能な処理パイプライン
3. 特徴量保存・管理機能

### Phase 3: Gold 完成
1. バージョン管理システム
2. キャッシュ・メタデータ機能
3. 企業級運用機能

## 設定ファイル例

### Gold レベル用設定 (configs/data_config.yaml)
```yaml
database:
  path: "/path/to/kaggle_datasets.duckdb"
  schema: "playground_series_s5e7"

cache:
  directory: "/tmp/ml_cache"
  max_size_gb: 10
  ttl_hours: 24

features:
  basic:
    - numerical_scaling
    - categorical_encoding
  advanced:
    - interaction_terms
    - polynomial_features
  custom:
    - domain_specific_features

preprocessing:
  missing_strategy: "median"
  outlier_method: "iqr"
  scaling_method: "standard"

versioning:
  auto_version: true
  retention_days: 30
```

## 期待効果

### 開発効率
- **Bronze**: 即座に開始、最短でベースライン構築
- **Silver**: 構造化により保守性向上、チーム開発可能
- **Gold**: 企業級機能で大規模運用・継続的改善

### スケーラビリティ
- プロジェクト成長に応じた段階的拡張
- 既存コードの再利用性確保
- 学習コスト最小化

### 品質保証
- 各レベルでの明確な責任範囲
- バージョン管理によるトレーサビリティ
- 監査機能による品質管理

＝＝＝＝＝＝＝＝＝＝＝＝＝

 Medallion Architecture テーブル詳細

  🥉 BRONZE層 (基本前処理)

  - train: 18,524行、11列
  - test: 6,175行、10列
  - 処理内容: カテゴリカル変数のエンコーディング
    - Stage_fear: "No"/"None"→0, "Yes"→1
    - Drained_after_socializing: "No"/"None"→0, "Yes"→1

  🥈 SILVER層 (特徴量エンジニアリング)

  - train: 18,524行、36列
  - test: 6,175行、35列
  - 特徴量カテゴリ:
    - 元特徴量(7): Time_spent_Alone, Stage_fear, Social_event_attendance, Going_outside, Drained_after_socializing, Friends_circle_size, Post_frequency
    - エンコード済み(2): Stage_fear_encoded, Drained_after_socializing_encoded
    - 新規作成(9): social_ratio, activity_sum, total_activity, avg_activity, activity_std, post_per_friend, fear_drain_interaction, extrovert_score, introvert_score
    - スケール済み(16): 全特徴量の標準化版

  🥇 GOLD層 (ML-Ready)

  - train: 18,524行、13列
  - test: 6,175行、11列
  - 厳選された10特徴量: extrovert_score, introvert_score, Social_event_attendance, Time_spent_Alone, Drained_after_socializing_encoded, Stage_fear_encoded, social_ratio, Friends_circle_size, Going_outside, Post_frequency

  ターゲット分布

  - Extrovert: 13,699件 (74.0%)
  - Introvert: 4,825件 (26.0%)

  キー特徴量統計

  - extrovert_score: 0-32 (平均17.3、標準偏差7.3)
  - introvert_score: 0-15 (平均3.9、標準偏差4.2)
  - social_ratio: 0-10 (平均2.4、標準偏差2.2)

  完璧なmladllionアーキテクチャで、すぐにモデル学習に使用可能です！
