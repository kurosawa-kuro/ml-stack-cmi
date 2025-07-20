# Training Scripts

機械学習モデル訓練用スクリプト集

## スクリプト一覧

### 1. train_lightgbm.py
**用途**: LightGBMベースライン訓練
- GroupKFold交差検証
- Binary + Multiclass予測戦略
- tsfresh特徴量統合
- Optuna最適化オプション
- 複合F1スコア評価

**実行タイミング**: Week 1 Day 7 / Week 2（ベースライン構築）

```bash
python scripts/training/train_lightgbm.py
```

### 2. train_cnn.py
**用途**: 1D CNN深層学習パイプライン
- InceptionTime風アーキテクチャ
- マルチモーダル融合（IMU/ToF/Thermopile）
- データ拡張（時間シフト、ノイズ注入）
- 混合精度訓練
- TensorFlow/Keras実装

**実行タイミング**: Week 2 Day 8-11（深層学習導入）

```bash
python scripts/training/train_cnn.py
```

### 3. train_baseline.py
**用途**: 汎用訓練スクリプト（既存）
- 基本的な訓練パイプライン
- モデル非依存設計

```bash
python scripts/training/train_baseline.py
```

## モデル比較

| モデル | 特徴 | 適用場面 | 期待性能 |
|--------|------|----------|----------|
| **LightGBM** | 表形式特徴量に強い | tsfresh特徴量 | Week 1: 0.50+ |
| **1D CNN** | 時系列パターン認識 | 生センサー配列 | Week 2: 0.57+ |

## 実行戦略

### Week 1: ベースライン構築
```bash
# 1. データ準備完了後
python scripts/training/train_lightgbm.py

# 2. 初回Kaggle提出
# submission_lgb_*.csv をアップロード
```

### Week 2: 深層学習
```bash
# 1. CNN訓練
python scripts/training/train_cnn.py

# 2. 性能比較
python scripts/evaluation/model_evaluation.py
```

## 出力ファイル

- `outputs/models/lgb_baseline/` - LightGBMモデル群
- `outputs/models/cnn_1d/` - CNNモデル群
- `outputs/models/*/cv_results.json` - 交差検証結果
- `outputs/models/*/oof_predictions.csv` - Out-of-fold予測
- `submission_*.csv` - Kaggle提出ファイル

## 重要な設計原則

- **GroupKFold必須**: 参加者リーク防止
- **複合F1最適化**: Binary F1 + Macro F1
- **再現性確保**: random seed固定
- **段階的改善**: Week毎の明確な目標設定