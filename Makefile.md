# Makefile Usage Guide

## Overview
S5E7 Personality Prediction プロジェクトのMakefileコマンド使用方法ガイド

## Available Commands

### Setup Commands

#### `make install`
基本的な依存関係をインストールします。
```bash
make install
```

#### `make dev-install`
開発用ツール、最適化、可視化ライブラリを含む全ての依存関係をインストールします。
```bash
make dev-install
```

#### `make setup`
プロジェクトに必要なディレクトリ構造を作成します。
```bash
make setup
```

### Validation Commands

#### `make validate-cv`
クイックCV検証を実行します。
```bash
make validate-cv
```

### Training Commands

#### `make train-fast-dev`
高速開発用トレーニングを実行します。
- 用途: 開発時の素早いテスト
- 実行時間: 短時間
```bash
make train-fast-dev
```

#### `make train-full-optimized`
完全最適化トレーニングを実行します。
- 用途: 本格的なモデル訓練
- 実行時間: 中程度
```bash
make train-full-optimized
```

#### `make train-max-performance`
最大パフォーマンストレーニングを実行します。
- 用途: Bronze Medal獲得を目指すトレーニング
- 実行時間: 長時間
```bash
make train-max-performance
```

### Prediction Commands

#### `make predict-basic`
基本提出用の予測を実行します。
```bash
make predict-basic
```

#### `make predict-gold`
Gold layer機能を使用した提出予測を実行します。
```bash
make predict-gold
```

### Code Quality Commands

#### `make lint`
コード品質チェックを実行します（black, flake8, mypy）。
```bash
make lint
```

#### `make format`
blackを使用してコードフォーマットを実行します。
```bash
make format
```

#### `make lint-fix`
コードフォーマットを適用し、結果を表示します。
```bash
make lint-fix
```

### Testing Commands

#### `make test`
全てのテストを実行します。
```bash
make test
```

### Maintenance Commands

#### `make clean`
出力ファイル、キャッシュファイルを削除します。
```bash
make clean
```

#### `make help`
利用可能なコマンド一覧を表示します。
```bash
make help
```

## Typical Workflow

### 1. 初期セットアップ
```bash
make dev-install
make setup
```

### 2. 開発サイクル
```bash
# 高速テスト
make train-fast-dev

# CV検証
make validate-cv

# コード品質チェック
make lint-fix

# テスト実行
make test
```

### 3. 本格的なトレーニング
```bash
# 最適化トレーニング
make train-full-optimized

# 最大パフォーマンストレーニング（Bronze Medal狙い）
make train-max-performance
```

### 4. 提出ファイル生成
```bash
# 基本提出
make predict-basic

# Gold layer提出
make predict-gold
```

### 5. クリーンアップ
```bash
make clean
```

## Notes

- 全てのPythonスクリプトは`PYTHONPATH=.`環境で実行されます
- 出力ファイルは`outputs/`, `submissions/`, `logs/`ディレクトリに保存されます
- 開発時は`make train-fast-dev`で素早くテストし、本格的な実験では`make train-max-performance`を使用してください

## File Structure

```
/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/
├── Makefile                           # このガイドで説明するMakefile
├── scripts/
│   ├── train_fast_dev.py             # 高速開発用トレーニング
│   ├── train_full_optimized.py       # 完全最適化トレーニング
│   ├── train_max_performance.py      # 最大パフォーマンストレーニング
│   ├── predict_basic_submission.py   # 基本提出予測
│   ├── predict_gold_submission.py    # Gold提出予測
│   └── validate_quick_cv.py          # CV検証
├── data/                             # データディレクトリ
├── outputs/                          # 出力ディレクトリ
├── submissions/                      # 提出ファイルディレクトリ
└── logs/                            # ログディレクトリ
```

## Performance Expectations

### Training Commands Performance
- **train-fast-dev**: ~0.5秒 - 素早い開発用テスト
- **train-full-optimized**: ~2-5分 - 本格的な最適化済みトレーニング
- **train-max-performance**: ~10-30分 - Bronze Medal狙いの最大パフォーマンス

### Current Project Status
- **Bronze Medal Target**: 0.976518 accuracy
- **Current Best**: 0.9684 accuracy (Gap: +0.008)
- **Architecture**: Medallion data pipeline (Bronze → Silver → Gold)
- **Test Coverage**: 73% (475 tests)

## Quick Reference

| Command | Purpose | Time | Use Case |
|---------|---------|------|----------|
| `make train-fast-dev` | 高速開発 | ~0.5s | アイデア検証 |
| `make train-full-optimized` | 最適化済み | ~5min | 本格実験 |
| `make train-max-performance` | 最大性能 | ~30min | Bronze Medal狙い |
| `make predict-basic` | 基本予測 | ~1min | 標準提出 |
| `make predict-gold` | Gold予測 | ~2min | 高品質提出 |
| `make validate-cv` | CV検証 | ~1min | 結果確認 |