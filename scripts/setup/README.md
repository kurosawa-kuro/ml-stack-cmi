# Setup Scripts

プロジェクト初期化とデータ理解のためのスクリプト集

## スクリプト一覧

### 1. project_setup.py
**用途**: プロジェクト環境の初期化と検証
- ディレクトリ作成
- 必要パッケージの確認・インストール
- データ接続テスト
- 環境設定の検証

**実行タイミング**: プロジェクト開始時（最初に実行）

```bash
python scripts/setup/project_setup.py
```

### 2. exploratory_analysis.py
**用途**: 包括的探索的データ分析（EDA）
- センサーデータ分布分析
- 欠損値パターン分析
- ラベル分布とクラス不均衡分析
- 時系列可視化
- 参加者別分析

**実行タイミング**: Week 1 Day 1-2（データ理解段階）

```bash
python scripts/setup/exploratory_analysis.py
```

### 3. data_quality_check.py
**用途**: データ品質検証と異常値検出
- センサー値範囲の妥当性検証
- 統計的異常値検出
- タイムスタンプ連続性確認
- データ一貫性チェック

**実行タイミング**: Week 1 Day 3-4（前処理前の検証）

```bash
python scripts/setup/data_quality_check.py
```

## 実行順序

1. `project_setup.py` - 環境準備
2. `exploratory_analysis.py` - データ理解
3. `data_quality_check.py` - 品質確認

## 出力ファイル

- `outputs/figures/eda/` - EDA可視化結果
- `outputs/reports/data_quality/` - データ品質レポート
- 各種CSV・JSON形式の分析結果