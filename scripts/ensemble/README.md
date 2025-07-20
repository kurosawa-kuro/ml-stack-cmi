# Ensemble Scripts

アンサンブル学習・提出用スクリプト集

## スクリプト一覧

### 1. train_ensemble.py
**用途**: 高度なアンサンブル訓練パイプライン
- マルチモデル統合（LightGBM + CNN + XGBoost）
- スタッキングアンサンブル（メタ学習器）
- CV性能ベース重み付け平均
- 重み最適化（scipy.optimize）
- Test Time Augmentation（TTA）

**実行タイミング**: Week 3 Day 17-20（最終最適化）

```bash
python scripts/ensemble/train_ensemble.py
```

### 2. generate_submission.py
**用途**: Kaggle提出ファイル生成（既存）
- 基本的な提出ファイル作成
- フォーマット検証

```bash
python scripts/ensemble/generate_submission.py
```

## アンサンブル戦略

### 1. 重み付け平均
```
最終予測 = w₁×LightGBM + w₂×CNN + w₃×XGBoost
重み = CV性能に基づく自動算出
```

### 2. スタッキング
```
レベル1: 基本モデル群（LightGBM, CNN, XGBoost）
　　　　　　　↓
レベル2: メタ学習器（LogisticRegression）
　　　　　　　↓
最終予測: 統合予測結果
```

### 3. 最適化手法
- **重み最適化**: scipy.optimize で F1 スコア最大化
- **TTA適用**: 予測時の複数回実行・平均化
- **GroupKFold**: 一貫した参加者別分割

## 実行フロー

### Phase 1: 基本モデル準備
```bash
# 必要な基本モデルの訓練完了を確認
ls outputs/models/lgb_baseline/oof_predictions.csv
ls outputs/models/cnn_1d/oof_predictions.csv
```

### Phase 2: アンサンブル構築
```bash
python scripts/ensemble/train_ensemble.py
```

### Phase 3: 最終提出
```bash
# 生成された提出ファイルをKaggleにアップロード
ls outputs/models/ensemble/submission_ensemble_*.csv
```

## 期待性能向上

| 手法 | 期待CV改善 | 理由 |
|------|------------|------|
| **重み付け平均** | +0.01-0.02 | モデル多様性活用 |
| **スタッキング** | +0.02-0.03 | 高次特徴量学習 |
| **TTA** | +0.005-0.01 | 予測安定性向上 |
| **重み最適化** | +0.01-0.015 | 最適重み発見 |

## 出力ファイル

- `outputs/models/ensemble/` - アンサンブルモデル群
- `submission_ensemble_*.csv` - 最終提出ファイル
- `ensemble_config_*.json` - アンサンブル設定
- `ENSEMBLE_REPORT.md` - 詳細分析レポート

## Bronze Medal戦略

### 目標: LB 0.60+
1. **個別モデル**: 各々0.55+を達成
2. **アンサンブル**: +0.03-0.05の改善
3. **最終予測**: 0.58-0.62の範囲を目指す

### 提出戦略
1. **Primary**: 最適化アンサンブル
2. **Backup**: 最高性能単体モデル
3. **Conservative**: 安定性重視の重み付け平均