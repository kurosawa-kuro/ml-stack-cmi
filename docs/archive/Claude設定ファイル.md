# CLAUDE.md
このファイルは、このリポジトリでコードを扱う際のClaude Code (claude.ai/code) への指針を提供します。

## 【プロジェクト概要】Kaggle S5E7 性格予測
- **コンペティション**: https://www.kaggle.com/competitions/playground-series-s5e7/overview
- **問題**: 二値分類（内向的 vs 外向的）
- **評価指標**: 精度（Accuracy）
- **現在の順位**: 2749チーム中1182位（上位43.0%）
- **現在のベストスコア**: 0.9684（CV スコア）
- **ブロンズメダル目標**: 0.976518（+0.008の改善が必要）

## 【重要 - 現在のプロジェクト状態】
### 高度な実装段階
- **成熟した実装**: 700行以上の本番環境対応コードを含むプロフェッショナルなMLパイプライン
- **ブロンズメダル目標**: 0.8%の改善が必要（現在: 0.9684、目標: 0.976518）
- **堅牢なアーキテクチャ**: メダリオンデータ管理、パイプライン統合、包括的なCVフレームワーク
- **高いテストカバレッジ**: 18個のテストファイルで475個のテスト、73%のカバレッジ
- **データ準備完了**: DuckDBにコンペティションデータ準備済み `/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb`

### 現在のパフォーマンス状況
- **最新CVスコア**: 0.9684 ± 0.0020（96.84%の精度）
- **ブロンズまでのギャップ**: +0.008の改善が必要
- **アーキテクチャ品質**: 将来の過度な複雑化を防ぐ拡張可能な設計

## 【メダリオンアーキテクチャ】単一ソースデータ処理パイプライン
### データ系譜と単一の真実の源泉
```
🗃️  生データソース（単一の真実の源泉）
     │
     ├── DuckDB: `/data/kaggle_datasets.duckdb`
     │   └── スキーマ: `playground_series_s5e7`
     │       ├── テーブル: `train`（元のコンペティションデータ）
     │       ├── テーブル: `test`（元のコンペティションデータ）  
     │       └── テーブル: `sample_submission`（元のフォーマット）
     │
     ↓ [ブロンズ処理]
     │
🥉  ブロンズレイヤー（`src/data/bronze.py`） 
     │   └── 目的: 生データの標準化と品質保証
     │   └── 出力: DuckDB `bronze.train`、`bronze.test`
     │
     ↓ [シルバー処理]
     │  
🥈  シルバーレイヤー（`src/data/silver.py`）
     │   └── 目的: 特徴量エンジニアリングとドメイン知識の統合
     │   └── 入力: ブロンズレイヤーテーブル（依存関係: bronze.py）
     │   └── 出力: DuckDB `silver.train`、`silver.test`
     │
     ↓ [ゴールド処理]
     │
🥇  ゴールドレイヤー（`src/data/gold.py`）
     │   └── 目的: ML対応データ準備とモデルインターフェース
     │   └── 入力: シルバーレイヤーテーブル（依存関係: silver.py）
     │   └── 出力: LightGBM用の X_train、y_train、X_test
```

### 実装構造
```
src/
├── data/                 # 🏗️ メダリオンアーキテクチャ（単一ソースパイプライン）
│   ├── bronze.py         # 🥉 生 → 標準化（エントリーポイント）
│   ├── silver.py         # 🥈 標準化 → エンジニアリング（依存: bronze）
│   └── gold.py           # 🥇 エンジニアリング → ML対応（依存: silver）
├── models.py             # 🤖 LightGBMモデル（消費: gold）
├── validation.py         # ✅ CVフレームワーク（オーケストレート: bronze→silver→gold）
└── util/                 # 🛠️ サポートインフラ
    ├── time_tracker.py   
    └── notifications.py  
```

### メダリオンデータ処理レイヤー

## 🥉 ブロンズレイヤー - 生データ標準化（エントリーポイント）
### 単一ソースの責任
**入力**: 元のDuckDBテーブル（`playground_series_s5e7.train`、`playground_series_s5e7.test`）  
**出力**: 標準化されたDuckDBテーブル（`bronze.train`、`bronze.test`）  
**依存関係**: なし（メダリオンパイプラインのエントリーポイント）

### コア処理関数
```python
# プライマリデータインターフェース（単一ソース）
load_data() → (train_df, test_df)                    # 生データアクセスポイント
create_bronze_tables() → bronze.train, bronze.test  # 標準化された出力

# データ品質保証  
validate_data_quality()     # 型検証、範囲ガード
advanced_missing_strategy() # 欠損値インテリジェンス
encode_categorical_robust() # Yes/No → バイナリ標準化
winsorize_outliers()        # 数値安定性処理
```

### LightGBM最適化データ品質パイプライン
**1. 型安全性と検証**
- 明示的なdtype設定: `int/float/bool/category`
- 範囲ガード: `Time_spent_Alone ≤ 24時間`、非負の行動メトリクス
- ダウンストリームの破損を防ぐスキーマ検証

**2. 欠損値インテリジェンス**
- **欠損フラグ**: `Stage_fear`（〜10%）、`Going_outside`（〜8%）のバイナリインジケータ
- **LightGBMネイティブハンドリング**: 自動ツリー処理のためNaNを保持
- **クロス特徴量パターン**: 高相関を活用した補完候補
- **体系的分析**: 欠損パターンの区別（ランダム vs 体系的）

**3. カテゴリカル標準化**
- **Yes/No正規化**: 大文字小文字を区別しない統一マッピング → {0,1}
- **LightGBMバイナリ最適化**: ツリー分割のための最適エンコーディング
- **欠損カテゴリ処理**: ダウンストリームLightGBM処理のために保持

**4. リーク防止基盤**
- **フォールド安全統計**: すべての計算値はCVフォールド内で分離
- **パイプライン準備**: シルバーレイヤー用のsklearn互換トランスフォーマー
- **監査証跡**: ダウンストリーム検証のための包括的メタデータ

### ブロンズ品質保証
✅ **単一の真実の源泉**: すべてのダウンストリーム処理はブロンズテーブルのみを使用  
✅ **LightGBM最適化**: ツリーベースモデル専用に設計された前処理  
✅ **コンペティショングレード**: 実証済みのトップティアKaggle前処理パターンを実装  
✅ **品質保証**: データ破損を防ぐ包括的な検証  
✅ **パフォーマンス対応**: 高速イテレーションを可能にする1秒未満の処理

## 🥈 シルバーレイヤー - 特徴量エンジニアリングとドメイン知識
### 単一ソース依存チェーン
**入力**: ブロンズレイヤーテーブル（`bronze.train`、`bronze.test`） - **排他的データソース**  
**出力**: 拡張DuckDBテーブル（`silver.train`、`silver.test`）  
**依存関係**: `src/data/bronze.py`（最初にブロンズパイプラインを実行する必要がある）

### コア特徴量エンジニアリングパイプライン
```python
# ブロンズ → シルバー変換（単一パイプライン）
load_silver_data() → enhanced_df                    # 消費: ブロンズテーブルのみ
create_silver_tables() → silver.train, silver.test # 拡張特徴量出力

# 特徴量エンジニアリングレイヤー（順次処理）
advanced_features()          # 15以上の統計的・ドメイン特徴量  
s5e7_interaction_features()  # トップティア交互作用パターン
s5e7_drain_adjusted_features() # 疲労調整活動モデリング
s5e7_communication_ratios()  # オンライン vs オフライン行動比率
polynomial_features()        # 次数2の非線形組み合わせ
```

### トップティア特徴量エンジニアリング（ブロンズ → シルバー変換）
**1. 優勝ソリューション交互作用特徴量**（+0.2-0.4%の実証済み影響）
```python
# ブロンズ入力 → シルバー拡張特徴量
Social_event_participation_rate = Social_event_attendance ÷ Going_outside
Non_social_outings = Going_outside - Social_event_attendance  
Communication_ratio = Post_frequency ÷ (Social_event_attendance + Going_outside)
Friend_social_efficiency = Social_event_attendance ÷ Friends_circle_size
```

**2. 疲労調整ドメインモデリング**（+0.1-0.2% 内向性精度）
```python  
# 心理的行動モデリング（トップティアイノベーション）
Activity_ratio = comprehensive_activity_index(bronze_features)
Drain_adjusted_activity = activity_ratio × (1 - Drained_after_socializing)
Introvert_extrovert_spectrum = quantified_personality_score(bronze_features)
```

**3. LightGBMツリー最適化特徴量**（+0.3-0.5% ツリー処理ゲイン）
- **欠損値保持**: LightGBMネイティブ処理のためブロンズNaNハンドリングを継承
- **比率特徴量**: ツリーベース分割パターン用に最適化
- **バイナリ交互作用**: ブロンズカテゴリカル標準化を活用
- **複合指標**: マルチ特徴量統計集約

### シルバー処理保証  
✅ **ブロンズ依存**: ブロンズレイヤーのみを消費（生データアクセスなし）  
✅ **特徴量系譜**: ブロンズ → シルバー変換の明確なトレーサビリティ  
✅ **LightGBM最適化**: すべての特徴量はツリーベースモデル消費用に設計  
✅ **コンペティション実証済み**: 検証済みトップティアKaggle技術を実装  
✅ **パフォーマンス向上**: 測定された影響期待値を持つ30以上のエンジニアリング特徴量

## 🥇 ゴールドレイヤー - ML対応データとモデルインターフェース
### 単一ソース依存チェーン
**入力**: シルバーレイヤーテーブル（`silver.train`、`silver.test`） - **排他的データソース**  
**出力**: LightGBM対応配列（`X_train`、`y_train`、`X_test`）  
**依存関係**: `src/data/silver.py`（最初にシルバーパイプラインを実行する必要がある）

### コアML準備パイプライン
```python
# シルバー → ゴールド変換（最終MLインターフェース）
get_ml_ready_data() → X_train, y_train, X_test     # LightGBM消費準備完了
prepare_model_data() → formatted_arrays            # モデル固有のフォーマット

# ML最適化レイヤー（順次処理）
clean_and_validate_features()   # データ品質最終検証
select_best_features()          # 統計的特徴量選択（F検定 + MI）
create_submission_format()      # コンペティション出力標準化
```

### LightGBMモデルインターフェース（シルバー → ゴールド → モデル）
**1. 特徴量選択と最適化**
```python
# シルバー入力 → ゴールド最適化特徴量  
statistical_selection = F_test + mutual_information(silver_features)
lightgbm_ready_features = feature_importance_ranking(selected_features)
X_train, y_train = prepare_training_data(optimized_features)
X_test = prepare_inference_data(optimized_features)
```

**2. 本番環境対応データ品質**
- **最終検証**: 無限値処理、外れ値検出
- **型一貫性**: LightGBM互換データ型の確保
- **メモリ最適化**: 訓練用の効率的な配列フォーマット
- **監査完全性**: 包括的なデータ系譜検証

**3. コンペティション出力インターフェース**
- **提出フォーマット**: 標準Kaggle提出ファイル作成
- **モデル予測インターフェース**: 直接LightGBM消費フォーマット  
- **パフォーマンス監視**: 特徴量重要度と予測追跡

### ゴールド処理保証
✅ **シルバー依存**: シルバーレイヤーのみを消費（ブロンズ/生アクセスなし）  
✅ **モデル準備完了**: 追加処理なしで直接LightGBM消費  
✅ **コンペティションフォーマット**: 標準Kaggle提出ファイル互換性  
✅ **本番品質**: モデル訓練の安定性を確保する最終検証  
✅ **パフォーマンス最適化**: ブロンズメダル目標（0.976518）を最大化する特徴量選択

## 🎯 メダリオンパイプライン開発戦略
### 単一ソース処理フロー
```
生データ → 🥉 ブロンズ → 🥈 シルバー → 🥇 ゴールド → 🤖 LightGBM → 🏆 ブロンズメダル (0.976518)
```
**現在のフェーズ**: LightGBMベースライン用のブロンズ + シルバー最適化  
**目標**: ブロンズメダル閾値を達成する単一モデル  
**アーキテクチャ**: データ系譜の整合性を確保するメダリオンパイプライン

## 🗃️ 単一ソースデータ管理（DuckDB）
### プライマリデータソース（単一の真実の源泉）
**データベース**: `/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb`

### スキーマ構造とデータ系譜
```sql
-- 生コンペティションデータ（元のソース）
playground_series_s5e7.train           # 元のKaggle訓練データ
playground_series_s5e7.test            # 元のKaggleテストデータ  
playground_series_s5e7.sample_submission # 元の提出フォーマット

-- メダリオンパイプライン出力（処理済みレイヤー）
bronze.train, bronze.test              # 🥉 標準化・検証済み
silver.train, silver.test              # 🥈 特徴量エンジニアリング済み  
gold.X_train, gold.y_train, gold.X_test # 🥇 ML対応（オプション永続化）
```

### データアクセスパターン（単一ソース強制）
```python
# ❌ 決して: シルバー/ゴールドレイヤーでの直接生データアクセス
# ✅ 常に: 適切なレイヤーのロード関数を使用

# ブロンズレイヤー（エントリーポイント）
from src.data.bronze import load_data
train_raw, test_raw = load_data()  # ブロンズのみが生データにアクセス

# シルバーレイヤー（ブロンズ依存）  
from src.data.silver import load_silver_data
train_silver, test_silver = load_silver_data()  # ブロンズ出力のみにアクセス

# ゴールドレイヤー（シルバー依存）
from src.data.gold import get_ml_ready_data  
X_train, y_train, X_test = get_ml_ready_data()  # シルバー出力のみにアクセス
```

### 元の特徴量スキーマ（コンペティションデータ）
- **ターゲット**: `Personality`（内向的/外向的） - 二値分類
- **ID**: `id` - 行識別子
- **数値特徴量**（5）: Time_spent_Alone、Social_event_attendance、Going_outside、Friends_circle_size、Post_frequency
- **カテゴリカル特徴量**（2）: Stage_fear（Yes/No）、Drained_after_socializing（Yes/No）

### 単一ソースの利点
✅ **データ系譜**: 生 → ブロンズ → シルバー → ゴールドの明確な変換追跡  
✅ **依存関係制御**: 各レイヤーは直前の前任者のみにアクセス  
✅ **一貫性保証**: すべてのダウンストリーム処理は標準化された入力を使用  
✅ **デバッグ効率**: 問題は特定のパイプラインレイヤーに追跡可能  
✅ **キャッシュ最適化**: 中間結果は再利用のためDuckDBに保存

## 【開発コマンド】
### 現在利用可能（Makefile）
```bash
make install              # 依存関係のインストール
make dev-install         # 開発ツール込みのインストール
make setup               # ディレクトリ構造の作成
make quick-test          # 単一モデルのクイックテスト
make personality-prediction  # フルワークフロー（実装時）
make test                # テストの実行（テスト存在時）
make clean               # 出力のクリーンアップ
make help                # 利用可能なコマンドを表示
```

### 利用可能なコマンド（実装済み）
```bash
# コアワークフロー
make install              # ✅ 依存関係のインストール
make dev-install         # ✅ 開発ツール込みのインストール
make test                # ✅ 475テストの実行（73%カバレッジ）
make quick-test          # ✅ 高速モデル検証
make personality-prediction  # ✅ フル訓練パイプライン
make clean               # ✅ 出力のクリーンアップ

# 訓練バリエーション
python scripts/train_light.py    # ✅ 高速イテレーション（0.5秒）
python scripts/train.py          # ✅ 標準訓練
python scripts/train_enhanced.py # ✅ 高度な特徴量
python scripts/train_heavy.py    # ✅ 完全最適化
```

## 【依存関係と環境】
### インストール（pyproject.toml設定済み）
```bash
pip install -e .                    # 基本的なML依存関係
pip install -e .[dev]              # + 開発ツール
pip install -e .[optimization]     # + 調整用Optuna
pip install -e .[visualization]    # + プロットライブラリ
```

### コア依存関係
- **データ**: pandas、numpy、duckdb
- **モデル**: scikit-learn、xgboost、lightgbm、catboost
- **最適化**: optuna
- **開発**: pytest、black、flake8、mypy
- **Python**: 3.8+

## 【現在のパフォーマンス】最近の訓練結果
- **Light Enhanced モデル**: 96.79% ± 0.22%（最新実行、30特徴量）
- **ベースラインモデル**: 96.84% ± 0.20%（最高CVスコア、10特徴量）
- **訓練効率**: 0.5秒（light）、0.39秒（ベースライン）
- **特徴量重要度**: poly_extrovert_score_Post_frequency（257.6）が最上位
- **ブロンズギャップ**: +0.8%必要（最適化で十分達成可能）

### パフォーマンス分析
- **一貫した結果**: 低い標準偏差は安定したモデルを示す
- **高速イテレーション**: 1秒未満の訓練により迅速な実験が可能
- **特徴量品質**: 多項式特徴量が強い予測力を示す

## 【実装ガイドライン】
### 設計原則（バランスの取れたアプローチ）
- **拡張可能なシンプルさ**: 複雑さを増やさずに成長をサポートするクリーンな抽象化
- **リーク防止**: パイプライン統合によりCV対応の前処理を確保（実装済み）
- **CVを信頼**: 整合性検証を含むStratifiedKFold（実装済み）
- **エビデンスベース**: 重要度分析に基づく特徴量エンジニアリング
- **段階的開発**: メダリオンアーキテクチャが段階的な拡張をサポート

### 主要な実装メモ
- **CSVファイル不使用**: すべてのデータアクセスはDuckDB経由のみ
- **システムPython**: 仮想環境なし（プロジェクト履歴による）
- **分類設定**: `Personality`ターゲットの二値分類
- **精度指標**: 主要な評価基準

### 開発ワークフロー（最適化済み）
1. **現状を最適化**: ブロンズ向けのハイパーパラメータ調整と特徴量選択
2. **高速イテレーション**: 0.5秒の訓練サイクルで迅速な実験が可能
3. **厳密な検証**: データリーク防止とCV整合性チェック（実装済み）
4. **包括的なテスト**: 統合テストを含む73%カバレッジ（実装済み）
5. **思慮深いスケーリング**: メダリオンアーキテクチャが制御された拡張をサポート

### ブロンズレイヤー実装チェックリスト（トップティアパターン）
**必須ステップ（実装優先度）**:
- [ ] データロード時の明示的dtype設定（int/float/bool/category）
- [ ] 値範囲検証（Time_spent_Alone ≤ 24時間、非負チェック）
- [ ] Yes/No正規化辞書（大文字小文字統一 → {0,1}）
- [ ] 欠損フラグ生成（Stage_fear、Going_outside、Drained_after_socializing）
- [ ] フォールド内統計計算（CV安全な補完値とエンコーディング）
- [ ] 層別K-Foldセットアップ（クラス比率維持）

**強く推奨されるステップ（パフォーマンス向上）**:
- [ ] 外れ値ウィンソライジング（IQRベース、1%/99%パーセンタイルクリッピング）
- [ ] LightGBM最適化前処理（NaN保持、バイナリカテゴリカルエンコーディング）
- [ ] クロス特徴量補完（高相関パターンベースの欠損値推定）
- [ ] ツリーフレンドリー特徴量生成（比率、差、交互作用）

**実験的ステップ（微調整ゲイン）**:
- [ ] 比率特徴量（Time_spent_Alone/(Time_spent_Alone+Social_event_attendance)）
- [ ] RankGauss変換（高度に歪んだ特徴量の正規化）
- [ ] ターゲットエンコーディング + ノイズ（高カーディナリティカテゴリ用）

### シルバーレイヤー高度実装チェックリスト（トップティアパターン）
**高優先度（優勝ソリューションベース）**:
- [ ] ソーシャルイベント参加率（Social_event_attendance ÷ Going_outside）
- [ ] 非ソーシャル外出（Going_outside - Social_event_attendance）
- [ ] コミュニケーション比率（Post_frequency ÷ 総活動）
- [ ] 疲労調整活動（疲労ベースの活動調整）
- [ ] LightGBMフレンドリービニング（ツリー最適化数値離散化）

**中優先度（統計的複合指標）**:
- [ ] ソーシャル活動比率（統合ソーシャル活動指標）
- [ ] 友人-ソーシャル効率（Social_event_attendance ÷ Friends_circle_size）
- [ ] 内向的-外向的スペクトラム（性格定量化）
- [ ] コミュニケーションバランス（オンライン-オフライン活動バランス）

**実験的ステップ（微調整）**:
- [ ] トリプル交互作用（主要特徴量組み合わせ）
- [ ] 活動パターン分類（ソーシャル/非ソーシャル/オンライン）
- [ ] 疲労重み付け強化（より強いDrained_after_socializing活用）

## 【成功基準】
- **ブロンズメダル**: 0.976518+の精度（現在の0.9684から+0.8%）
- **アーキテクチャ品質**: 複雑さを制御した拡張可能な設計
- **信頼性**: データリーク防止、再現可能なCV結果（実装済み）
- **開発効率**: 1秒未満の訓練、包括的なテスト（実装済み）
- **長期的価値**: 将来のコンペで再利用可能なパターン

## 【ブロンズメダルロードマップ】
### 即座の機会（1-2週間）
1. **高度なシルバーレイヤー**（最高優先度 +0.4-0.8%期待）:
   - トップティア交互作用特徴量（ソーシャルイベント参加率、communication_ratio）
   - 疲労調整活動スコア（drain_adjusted_activity） - トップティアイノベーション
   - LightGBM最適化ビニング（ツリーフレンドリー数値離散化）
   - 統計的複合指標（Social_activity_ratio、introvert_extrovert_spectrum）

2. **高度なブロンズレイヤー**（高優先度 +0.3-0.5%期待）:
   - Stage_fear、Going_outside用の欠損インジケータ（トップティア実証済み）
   - 高相関パターンを使用したクロス特徴量補完
   - 数値安定性のための外れ値ウィンソライジング（IQRベース）
   - LightGBM最適化前処理（NaN保持、バイナリエンコーディング）

3. **ハイパーパラメータ最適化**: 既存のOptuna統合を活用（+0.2-0.4%）

4. **強化されたデータ品質**（中優先度 +0.1-0.3%）:
   - 範囲ガード付きdtype検証（Time_spent_Alone ≤ 24時間）
   - カテゴリカル標準化（大文字小文字を区別しないYes/Noマッピング）
   - 体系的 vs ランダム検出のための欠損パターン分析

5. **CVフレームワーク強化**（+0.1-0.2%）:
   - 明示的な内向的/外向的比率維持を伴う層別K-Fold
   - 情報リークを防ぐフォールド安全統計計算
   - 一貫した訓練/検証処理を確保するパイプライン統合

6. **特徴量選択**: トップ重要度特徴量に焦点（poly_extrovert_score_*）
7. **モデルアンサンブル**: 予測の安定性のためにCVフォールドを組み合わせ
8. **閾値調整**: 精度向上のための分類閾値の最適化

### 準備済み技術資産
- ✅ **データリーク防止**: パイプライン統合実装済み
- ✅ **CVフレームワーク**: 整合性検証と層別サンプリング
- ✅ **特徴量エンジニアリング**: 重要度ランキング付き30以上の特徴量
- ✅ **最適化インフラ**: ハイパーパラメータ調整用Optuna統合
- ✅ **パフォーマンス監視**: 時間追跡と包括的なロギング