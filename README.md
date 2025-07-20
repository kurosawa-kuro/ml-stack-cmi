# CLAUDE.md
このファイルは、このリポジトリでコードを扱う際のClaude Code (claude.ai/code) への指針を提供します。

## 【プロジェクト概要】Kaggle CMI - 行動検出
- **コンペティション**: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data
- **問題**: 多クラス + 二値分類（センサーデータからのBFRB行動検出）
- **評価指標**: 0.5 × (Binary F1 + Macro F1) - カスタムF1複合スコア
- **現在の順位**: 未定（新規プロジェクト）
- **ブロンズメダル目標**: ~0.60+（約360チーム中上位60%推定）
- **締切**: 2025年8月26日（残り約5週間）

## 【重要 - 現在のプロジェクト状態】
### 初期セットアップ段階
- **コンペティション種別**: 時系列多モーダルセンサー分類
- **データ規模**: ~200参加者 × 数百セッション（~1.5GB）
- **センサーチャンネル**: IMU（加速度/ジャイロ）、ToF距離（4ch）、サーモパイル温度（5ch）
- **サンプリングレート**: 50Hz
- **ブロンズメダル目標**: LB ~0.60+（推定順位109-216/360チーム）

### 主要課題
- **多モーダル融合**: IMU + ToF + サーモパイルセンサー統合
- **クラス不均衡**: 二値検出 + 多クラスジェスチャー分類
- **時間的依存性**: 50Hzでの連続行動パターン
- **参加者リーク**: GroupKFold(participant_id)の使用必須

## 【データ構造】多モーダルセンサーストリーム

### 生データスキーマ
```
センサーデータ（50Hz サンプリング）:
├── timestamp                 # 時間参照
├── acc_x, acc_y, acc_z      # 加速度計（IMU）
├── gyro_x, gyro_y, gyro_z   # ジャイロスコープ（IMU）
├── tof_0, tof_1, tof_2, tof_3  # 飛行時間距離センサー
├── thermopile_0...4         # 温度アレイセンサー
├── participant_id           # 被験者識別子（GroupKFoldキー）
├── series_id               # セッション識別子
└── label                   # ターゲット: BFRB行動クラス
```

### ターゲットラベル
- **二値タスク**: BFRB存在（1）vs なし（0）
- **多クラスタスク**: 特定のBFRBジェスチャータイプ
- **評価**: 両方のタスクのバランスを必要とする複合F1スコア

## 【メダリオンアーキテクチャ】時系列用に適応

### センサー処理用データパイプライン
```
🗃️  生センサーデータ
     │
     ├── DuckDB: data/kaggle_datasets.duckdb
     │   └── 50Hz多モーダルセンサーストリーム
     │
     ↓ [ブロンズ処理]
     │
🥉  ブロンズレイヤー（src/data/bronze.py）
     │   └── センサー品質チェック & 正規化
     │   └── 欠損値処理（ToF/サーモパイル）
     │   └── 参加者対応分割
     │
     ↓ [シルバー処理]
     │  
🥈  シルバーレイヤー（src/data/silver.py）
     │   └── 時系列特徴量エンジニアリング
     │   └── FFT/統計特徴量（tsfresh）
     │   └── 多モーダルチャンネル融合
     │
     ↓ [ゴールド処理]
     │
🥇  ゴールドレイヤー（src/data/gold.py）
     │   └── ML対応特徴量/シーケンス
     │   └── GroupKFoldによる訓練/検証分割
     │   └── 正規化 & スケーリング
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

### DuckDBデータソース
- **データベースパス**: bronze.pyで参照（`DB_PATH` 変数）
- **プライマリテーブル**: DuckDB形式で保存された生コンペティションデータ
- **メダリオンパイプライン**: bronze → silver → gold レイヤー処理
- **GroupKFold戦略**: リークを防ぐ参加者対応CV

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

### センサー処理ガイドライン

#### ブロンズレイヤー - センサーデータ品質
**コア処理:**
- IMU正規化（チャンネル毎Z-score）
- ToF欠損値処理（0埋めまたはマスキング）
- サーモパイルノイズ除去
- タイムスタンプ検証（50Hz一貫性）
- CV用参加者グループ化

#### シルバーレイヤー - 特徴量エンジニアリング
**時系列特徴量:**
- 統計特徴量（tsfresh統合）
- FFT/周波数領域解析
- モーダル間センサー相関
- 時間パターン検出

**ドメイン固有特徴量:**
- 動作パターン（IMU自己相関）
- 近接パターン（ToF距離追跡）
- 熱シグネチャ（温度勾配）

#### ゴールドレイヤー - モデル準備
**データフォーマット:**
- 木ベースモデル用表形式特徴量（LightGBM/CatBoost）
- 深層学習用シーケンスデータ（CNN/RNN）
- GroupKFoldクロスバリデーション設定

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

## 【ブロンズメダルロードマップ】5週間計画

### 第1週: 基盤 & ベースライン（目標: LB 0.50+）
1. **EDA & センサー理解**
   - チャンネル毎統計と分布
   - ラベル頻度解析
   - 参加者/セッションデータ品質

2. **ブロンズレイヤー実装**
   - センサー正規化パイプライン
   - 欠損値戦略
   - GroupKFold CV設定

3. **クイックベースライン**
   - tsfresh特徴量 + LightGBM
   - シンプルスライディングウィンドウアプローチ

### 第2週: 深層学習統合（目標: LB 0.57-0.60）
1. **1D CNN実装**
   - InceptionTimeまたは類似アーキテクチャ
   - 単一モーダル（IMUのみ）ベースライン
   
2. **データ拡張**
   - 時間シフト
   - 回転拡張（IMU）
   - ノイズ注入

3. **GPU最適化**
   - バッチ処理
   - 混合精度訓練

### 第3週: 多モーダル融合（目標: LB 0.62+）
1. **多ブランチアーキテクチャ**
   - IMU/ToF/サーモパイル用個別ブランチ
   - 特徴量レベル融合
   
2. **高度な特徴量**
   - モーダル間相関
   - 注意機構
   
3. **アンサンブル戦略**
   - 5-fold × マルチシード
   - モデル多様性（CNN + 木ベース）

### 第4-5週: 最適化 & 堅牢性
1. **特徴量選択**
   - チャンネル重要度解析
   - 推論速度最適化
   
2. **指標最適化**
   - 二値 vs 多クラス重み付け
   - 閾値調整
   
3. **Private LB準備**
   - CV-LBアライメント
   - 堅牢な検証戦略

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