# Kaggle Playground Series S5E7 - ベースライン実装

## 概要
Kaggle Playground Series Season 5, Episode 7の性格予測（内向的/外向的）2クラス分類問題のベースライン実装です。

## 実装内容

### 1. データ探索 (notebooks/)
- `01_data_exploration.py`: DuckDBからのデータ読み込みと基本統計
- `02_eda_analysis.py`: 詳細なEDA（特徴量相関、ターゲット分布など）

### 2. 前処理パイプライン (src/preprocessing/)
- `custom_transformers.py`: カスタム特徴量エンジニアリング
  - Social Score: 社交活動スコア
  - Social Network: ソーシャルネットワーク指標
  - 欠損値フラグ
- `BinaryCategoricalEncoder`: Yes/Noのバイナリエンコーディング

### 3. モデル実装
- Random Forest
- LightGBM
- XGBoost

### 4. 評価システム (src/evaluation/)
- `validation.py`: 包括的な評価指標
  - Accuracy (主要指標)
  - Precision, Recall, F1-score
  - ROC-AUC
  - 混同行列
  - クラス別メトリクス

### 5. 予測・提出 (scripts/)
- `baseline_workflow.py`: 基本的なワークフロー
- `predict.py`: 予測専用スクリプト
- `integrated_workflow.py`: 統合ワークフロー

## 結果

### Cross-Validation スコア
- Random Forest: 96.77% (+/- 0.21%)
- LightGBM: **96.90%** (+/- 0.24%) ← 最良
- XGBoost: 96.86% (+/- 0.23%)

### 予測分布
- Extrovert: 4,610 (74.7%)
- Introvert: 1,565 (25.3%)

## 使用方法

### 1. ベースライン実行
```bash
python3 scripts/baseline_workflow.py
```

### 2. 統合ワークフロー実行
```bash
python3 scripts/integrated_workflow.py --config baseline
```

### 3. 予測のみ実行
```bash
python3 scripts/predict.py --model-dir artifacts/models --ensemble-method soft_voting
```

## ディレクトリ構造
```
.
├── notebooks/              # データ探索・分析
├── scripts/               # 実行スクリプト
├── src/                   # ソースコード
│   ├── config/           # 設定
│   ├── data/             # データローダー
│   ├── preprocessing/    # 前処理
│   └── evaluation/       # 評価
├── submissions/          # 提出ファイル
└── artifacts/           # 学習済みモデル・結果
```

## 主な特徴量
1. **数値特徴量** (5個)
   - Time_spent_Alone: 一人で過ごす時間
   - Social_event_attendance: 社交イベント参加度
   - Going_outside: 外出頻度
   - Friends_circle_size: 友人の数
   - Post_frequency: 投稿頻度

2. **カテゴリカル特徴量** (2個)
   - Stage_fear: ステージ恐怖症 (Yes/No)
   - Drained_after_socializing: 社交後の疲労 (Yes/No)

3. **エンジニアリング特徴量** (7個)
   - social_score: 社交活動スコア
   - social_network: ソーシャルネットワーク指標
   - 各数値特徴量の欠損値フラグ (5個)

## 改善のアイデア
1. **特徴量エンジニアリング**
   - 多項式特徴量
   - 交互作用特徴量
   - より高度な集約特徴量

2. **モデル最適化**
   - ハイパーパラメータチューニング (Optuna)
   - より高度なアンサンブル手法 (Stacking)
   - Neural Network モデルの追加

3. **前処理改善**
   - より洗練された欠損値処理
   - 外れ値検出と処理
   - 特徴量選択
