# Makefile 使い方ガイド

CMI競技用のMakefileコマンドの詳細な使い方を説明します。

## 📌 基本的な使い方

```bash
# ヘルプを表示（利用可能なコマンド一覧）
make help

# 特定のコマンドを実行
make <コマンド名>
```

## 🚀 セットアップ・初期化

### `make install`
基本的な依存関係をインストールします。
```bash
make install
# → requirements.txt の内容をインストール
```

### `make dev-install`
開発用ツールを含めてインストールします。
```bash
make dev-install
# → requirements-dev.txt の内容をインストール（black, flake8, mypy等）
```

### `make setup`
プロジェクト環境を初期化します。
```bash
make setup
# 実行内容:
# 1. 必要なディレクトリを作成
# 2. project_setup.pyを実行して環境を検証
# 3. データ接続テスト
```

## 📊 データ理解・品質チェック

### `make eda`
探索的データ分析（EDA）を実行します。
```bash
make eda
# 実行内容:
# - センサーデータの分布分析
# - 欠損値パターンの可視化
# - ラベル分布とクラス不均衡の分析
# - 時系列グラフの生成
# 出力: outputs/figures/eda/
```

### `make data-check`
データ品質を検証します。
```bash
make data-check
# 実行内容:
# - センサー値の範囲チェック
# - 異常値検出
# - タイムスタンプの連続性確認
# - train/testの一貫性チェック
# 出力: outputs/reports/data_quality/
```

## ⚙️ データ処理パイプライン（Medallion Architecture）

### `make bronze`
生データをクリーニング・正規化します。
```bash
make bronze
# Bronze Layer処理:
# - センサーデータの正規化（Z-score）
# - 欠損値処理（IMU補間、ToF/温度0埋め）
# - タイムスタンプ検証
# 出力: data/processed/bronze/
```

### `make silver`
特徴量エンジニアリングを実行します。
```bash
make silver
# Silver Layer処理:
# - tsfresh統計特徴量の抽出
# - FFT周波数領域特徴量
# - センサー間相関特徴量
# - ドメイン固有特徴量
# 出力: data/processed/silver/
```

### `make gold`
機械学習用のデータを準備します。
```bash
make gold
# Gold Layer処理:
# - ML用フォーマット変換
# - GroupKFold分割準備
# - 正規化・スケーリング
# 出力: data/processed/gold/
```

## 🤖 モデル訓練

### `make train-lgb`
LightGBMベースラインモデルを訓練します。
```bash
make train-lgb
# 実行内容:
# - GroupKFold交差検証（5-fold）
# - Binary + Multiclass予測
# - 複合F1スコア評価
# 出力: outputs/models/lgb_baseline/
```

### `make train-cnn`
1D CNNモデルを訓練します。
```bash
make train-cnn
# 実行内容:
# - InceptionTime風アーキテクチャ
# - マルチモーダルセンサー融合
# - データ拡張（時間シフト、ノイズ）
# 出力: outputs/models/cnn_1d/
```

## 📈 評価・分析

### `make evaluate`
モデルの包括的評価を実行します。
```bash
make evaluate
# 実行内容:
# - 混同行列の詳細分析
# - 参加者別性能評価
# - 誤分類パターンの特定
# - モデル比較（複数モデルがある場合）
# 出力: outputs/evaluation/
```

### `make feature-importance`
特徴量重要度を分析します。
```bash
make feature-importance
# 実行内容:
# - LightGBM特徴量重要度
# - Permutation重要度
# - 特徴量相関分析
# - センサー別重要度ランキング
# 出力: outputs/feature_analysis/
```

### `make validate-cv`
クイック交差検証を実行します。
```bash
make validate-cv
# 簡易的なCV検証（開発中の確認用）
```

## 🎭 最終ステップ

### `make ensemble`
アンサンブルモデルを訓練します。
```bash
make ensemble
# 実行内容:
# - LightGBM + CNN + XGBoost統合
# - 重み付け平均・スタッキング
# - 最適重み探索
# 出力: outputs/models/ensemble/
```

### `make submit`
Kaggle提出用ファイルを生成します。
```bash
make submit
# 実行内容:
# - 最終予測の生成
# - submission.csvの作成
# 出力: outputs/submissions/
```

## 🔄 週別ワークフロー（一括実行）

### `make week1-baseline`
Week 1の全工程を自動実行します。
```bash
make week1-baseline
# 実行順序:
# 1. setup      → 環境初期化
# 2. eda        → データ探索
# 3. data-check → 品質確認
# 4. bronze     → データクリーニング
# 5. silver     → 特徴量生成
# 6. gold       → ML準備
# 7. train-lgb  → ベースライン訓練
# 8. evaluate   → 性能評価
```

### `make week2-deep-learning`
Week 2のCNN訓練を実行します。
```bash
make week2-deep-learning
# 実行順序:
# 1. train-cnn  → CNN訓練
# 2. evaluate   → 性能評価
```

### `make week3-final`
Week 3の最終最適化を実行します。
```bash
make week3-final
# 実行順序:
# 1. feature-importance → 特徴量分析
# 2. ensemble          → アンサンブル
# 3. submit            → 提出ファイル生成
```

## 🔧 コード品質・保守

### `make lint`
コード品質をチェックします。
```bash
make lint
# チェック内容:
# - black（コードフォーマット）
# - flake8（スタイルガイド）
# - mypy（型チェック）
```

### `make format`
コードを自動フォーマットします。
```bash
make format
# blackを使用してコードを整形
```

### `make test`
テストスイートを実行します。
```bash
make test
# pytestでテストを実行（カバレッジ73%）
```

### `make clean`
出力ファイルをクリーンアップします。
```bash
make clean
# 削除対象:
# - outputs/
# - submissions/
# - logs/
# - __pycache__
# - .pytest_cache
```

## 💡 使用例

### 初めてプロジェクトを実行する場合
```bash
# 1. 環境セットアップ
make install
make setup

# 2. Week 1の作業を一括実行
make week1-baseline

# 3. 結果を確認
ls outputs/models/lgb_baseline/
```

### 日々の開発フロー
```bash
# 1. データ処理のみ実行
make bronze
make silver
make gold

# 2. モデル訓練
make train-lgb

# 3. 評価
make evaluate
```

### 最終提出前
```bash
# 1. 全体最適化
make week3-final

# 2. 提出ファイル確認
ls outputs/submissions/

# 3. Kaggleにアップロード
```

## ⚠️ 注意事項

1. **データの存在確認**: 初回実行前に競技データがダウンロードされていることを確認
2. **順序の重要性**: データ処理は bronze → silver → gold の順序で実行
3. **GroupKFold必須**: participant_idによるリーク防止が重要
4. **GPU利用**: CNN訓練時はGPU環境推奨

## 🎯 Bronze Medal達成への推奨フロー

```bash
# Week 1（基盤構築）
make week1-baseline
# → 期待スコア: LB 0.50+

# Week 2（深層学習）
make train-cnn
make evaluate
# → 期待スコア: LB 0.57-0.60

# Week 3（最終調整）
make week3-final
# → 期待スコア: LB 0.62+ (Bronze Medal!)
```

---

詳細な開発ガイドラインは [CLAUDE.md](CLAUDE.md) を参照してください。