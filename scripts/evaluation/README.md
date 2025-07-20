# Evaluation Scripts

モデル評価・分析用スクリプト集

## スクリプト一覧

### 1. model_evaluation.py
**用途**: 包括的モデル評価とエラー分析
- 詳細混同行列分析
- 参加者別性能評価
- 誤分類パターン特定
- モデル比較フレームワーク
- 性能可視化・レポート生成

**実行タイミング**: 各モデル訓練後（継続的評価）

```bash
python scripts/evaluation/model_evaluation.py
```

### 2. feature_analysis.py
**用途**: 特徴量重要度分析
- LightGBM特徴量重要度
- Permutation重要度分析
- 特徴量相関分析
- センサー別重要度ランキング
- 特徴量選択推奨

**実行タイミング**: Week 3 Day 17-18（特徴量最適化）

```bash
python scripts/evaluation/feature_analysis.py
```

### 3. validate_quick_cv.py
**用途**: 高速交差検証（既存）
- クイック性能確認
- 開発時の迅速検証

```bash
python scripts/evaluation/validate_quick_cv.py
```

## 評価指標

### 主要指標
- **複合F1スコア**: (Binary F1 + Macro F1) / 2（競技指標）
- **Binary F1**: 行動有無の二値分類性能
- **Macro F1**: 行動種別の多クラス分類性能

### 補助指標
- Precision/Recall詳細
- 混同行列分析
- 参加者別性能分布
- 特徴量重要度ランキング

## 分析フレームワーク

### 1. 性能分析
```bash
# モデル訓練後の包括評価
python scripts/evaluation/model_evaluation.py
```

### 2. エラー分析
- 誤分類パターンの特定
- 困難ケースの分析
- 参加者固有の課題発見

### 3. 特徴量分析
```bash
# 重要特徴量の特定
python scripts/evaluation/feature_analysis.py
```

## 実行順序

1. **モデル訓練後**: `model_evaluation.py`
2. **特徴量最適化時**: `feature_analysis.py`
3. **開発中の確認**: `validate_quick_cv.py`

## 出力ファイル

- `outputs/evaluation/` - 評価レポート・可視化
- `outputs/evaluation/plots/` - 性能グラフ群
- `outputs/feature_analysis/` - 特徴量分析結果
- `*_REPORT.md` - 詳細分析レポート

## 活用戦略

### Week 2: モデル改善
- エラー分析 → データ拡張戦略
- 特徴量重要度 → 特徴量エンジニアリング

### Week 3: 最終最適化
- 参加者別分析 → 個別対応戦略
- モデル比較 → アンサンブル設計