# CMI Competition - Scripts Directory

CMI Detect Behavior with Sensor Data競技用の包括的スクリプト集

## 📁 フォルダ構成

```
scripts/
├── setup/                   # プロジェクト初期化・データ理解
├── data_processing/         # データパイプライン（Bronze→Silver→Gold）
├── training/               # モデル訓練（LightGBM, CNN）
├── evaluation/             # モデル評価・分析
└── ensemble/               # アンサンブル・最終提出
```

## 🚀 実行フロー（5週間ロードマップ）

### Week 1: 基盤構築
```bash
# 1. 環境初期化
python scripts/setup/project_setup.py

# 2. データ理解
python scripts/setup/exploratory_analysis.py

# 3. データ品質確認
python scripts/setup/data_quality_check.py

# 4. データパイプライン
python scripts/data_processing/bronze_layer.py
python scripts/data_processing/silver_layer.py
python scripts/data_processing/gold_layer.py

# 5. ベースライン構築
python scripts/training/train_lightgbm.py
```

### Week 2: モデル改善
```bash
# 1. 深層学習導入
python scripts/training/train_cnn.py

# 2. 包括的評価
python scripts/evaluation/model_evaluation.py
```

### Week 3: 最終最適化
```bash
# 1. 特徴量分析
python scripts/evaluation/feature_analysis.py

# 2. アンサンブル構築
python scripts/ensemble/train_ensemble.py
```

## 📊 期待性能推移

| Week | 手法 | 期待CVスコア | 期待LBスコア |
|------|------|-------------|-------------|
| 1 | tsfresh + LightGBM | 0.48-0.52 | 0.47-0.51 |
| 2 | + 1D CNN + 拡張 | 0.57-0.60 | 0.56-0.59 |
| 3 | + アンサンブル | 0.62-0.65 | 0.61-0.64 |

## 🎯 Bronze Medal戦略

### 目標: LB 0.60+ (Bronze Medal圏)
- **推定順位**: 109-216位 / 360チーム
- **成功の鍵**: GroupKFold + マルチモーダル融合

### 実行チェックリスト
- [ ] Week 1: ベースライン LB 0.50+
- [ ] Week 2: CNN統合 LB 0.57+
- [ ] Week 3: アンサンブル LB 0.62+

## 🔧 各フォルダの詳細

### [setup/](./setup/) - プロジェクト基盤
- 環境初期化・検証
- 包括的EDA・可視化
- データ品質検証

### [data_processing/](./data_processing/) - データパイプライン
- Medallion Architecture実装
- センサー固有前処理
- 特徴量エンジニアリング

### [training/](./training/) - モデル訓練
- LightGBMベースライン
- 1D CNN深層学習
- GroupKFold交差検証

### [evaluation/](./evaluation/) - 性能分析
- 包括的モデル評価
- エラー分析・改善提案
- 特徴量重要度分析

### [ensemble/](./ensemble/) - 最終統合
- 高度なアンサンブル手法
- TTA・重み最適化
- Kaggle提出ファイル生成

## ⚠️ 重要な実装原則

### データリーク防止
- **GroupKFold必須**: participant_id による分割
- **時系列考慮**: 連続性を保った分割
- **テスト分離**: 訓練時の participant_id 使用禁止

### 競技特化設計
- **複合F1最適化**: (Binary F1 + Macro F1) / 2
- **マルチモーダル**: IMU + ToF + Thermopile 統合
- **50Hz時系列**: サンプリング周波数対応

### 産業品質
- **再現性**: random seed 固定・設定保存
- **スケーラビリティ**: メモリ効率・並列処理
- **監査可能**: 包括的ログ・レポート生成

## 📈 成功指標

- **技術目標**: Bronze medal (LB 0.60+)
- **学習目標**: 時系列多モーダル分析手法習得
- **品質目標**: 産業レベルMLパイプライン構築