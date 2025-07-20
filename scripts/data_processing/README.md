# Data Processing Scripts

Medallion Architecture（Bronze→Silver→Gold）に基づくデータ処理パイプライン

## スクリプト一覧

### 1. bronze_layer.py
**用途**: 生データの標準化と品質管理（Bronze Layer）
- センサーデータ正規化
- 欠損値処理（IMU補間、ToF/温度0埋め）
- タイムスタンプ検証
- 参加者グループ分割準備

**実行タイミング**: Week 1 Day 3-4（データクリーニング）

```bash
python scripts/data_processing/bronze_layer.py
```

### 2. silver_layer.py
**用途**: 特徴量エンジニアリング（Silver Layer）
- tsfresh統計特徴量抽出
- FFT周波数領域特徴量
- センサー間相関特徴量
- ドメイン知識ベース特徴量

**実行タイミング**: Week 1 Day 5-6（特徴量生成）

```bash
python scripts/data_processing/silver_layer.py
```

### 3. gold_layer.py
**用途**: ML対応データ準備（Gold Layer）
- 機械学習用フォーマット変換
- GroupKFold分割データ
- 正規化・スケーリング
- 訓練/テストデータ統合

**実行タイミング**: Week 1 Day 7（モデル訓練前）

```bash
python scripts/data_processing/gold_layer.py
```

## データパイプライン

```
🗃️ Raw Sensor Data (50Hz)
     ↓
🥉 Bronze Layer: 品質管理・正規化
     ↓
🥈 Silver Layer: 特徴量エンジニアリング
     ↓
🥇 Gold Layer: ML対応データ
```

## 実行順序

1. `bronze_layer.py` - データクリーニング
2. `silver_layer.py` - 特徴量生成
3. `gold_layer.py` - ML準備

## 出力データ

- `data/processed/bronze/` - クリーンセンサーデータ
- `data/processed/silver/` - 特徴量データ
- `data/processed/gold/` - ML対応データ

## 重要な設計原則

- **GroupKFold必須**: participant_id漏洩防止
- **センサー別処理**: IMU/ToF/Thermopile固有の前処理
- **メモリ効率**: 大容量データの段階的処理