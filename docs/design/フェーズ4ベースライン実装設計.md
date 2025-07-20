# フェーズ4: ベースライン実装設計

## 概要

段階的データ管理パターン構築（フェーズ3）完了後の次ステップとして、LightGBMを用いたベースラインモデルの実装を行います。目標は**CV 0.975**を達成し、安定した検証基盤を確立することです。

## 実装対象ファイル

### 1. `src/models.py` - モデル定義・学習

```python
"""
モデル定義・学習・予測機能
- LightGBMベースラインモデル
- クロスバリデーション対応
- ハイパーパラメータ管理
- モデル保存・読み込み機能
"""
```

#### 主要機能

1. **LightGBMModelクラス**
   - ハイパーパラメータ管理
   - 学習・予測メソッド
   - モデル保存・読み込み
   - 特徴量重要度取得

2. **CrossValidationTrainerクラス**
   - StratifiedKFoldベースのCV実行
   - Out-of-Fold予測生成
   - CVスコア計算・ログ出力
   - 乱数シード固定

3. **ユーティリティ関数**
   - モデル評価メトリクス（accuracy, AUC）
   - 予測結果の統計情報出力
   - 学習時間計測

#### 設計方針

- **シンプル性重視**: 複雑なファクトリパターンは避け、直接的な実装
- **再現性確保**: 乱数シード固定、同一環境での同一結果保証
- **拡張性考慮**: 将来的なXGBoost/CatBoost追加に対応可能な構造

### 2. `src/validation.py` - クロスバリデーション

```python
"""
クロスバリデーション戦略・評価指標
- StratifiedKFold設定
- CVスコア計算・集計
- 学習履歴ログ機能
- データリーク防止チェック
"""
```

#### 主要機能

1. **CVStrategy設定**
   - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   - 層化抽出による分布バランス保持
   - 再現可能性のための乱数シード固定

2. **評価指標計算**
   - Accuracy（主要指標）
   - AUC-ROC（補助指標）
   - 予測分布統計（Extrovert/Introvert比率）

3. **CVログ管理**
   - フォールド別スコア記録
   - 平均・標準偏差計算
   - 学習時間・メモリ使用量記録
   - 結果をJSON/CSVで保存

4. **データ整合性チェック**
   - train/testデータの形状確認
   - 欠損値・無限値チェック
   - ターゲット分布確認

### 3. `scripts/train.py` - 学習実行スクリプト

```python
"""
ベースラインモデル学習実行
- データ読み込み（Goldレベル）
- モデル学習・CV実行
- 結果出力・保存
- 通知・ログ出力
"""
```

#### 実行フロー

1. **初期化**
   - 時間トラッカー開始
   - 通知システム初期化
   - ログ設定

2. **データ準備**
   - Goldレベルデータ読み込み
   - 特徴量・ターゲット分離
   - データ整合性チェック

3. **モデル学習**
   - LightGBMベースライン学習
   - 5-Fold CV実行
   - Out-of-Fold予測生成

4. **結果出力**
   - CVスコア計算・表示
   - モデル保存（`outputs/models/`）
   - 学習ログ保存（`outputs/logs/`）
   - 通知送信（完了報告）

## ファイル構成

```
src/
├── models.py           # モデル定義・学習機能
├── validation.py       # CV戦略・評価指標
└── data/              # 既存のMedallion Architecture

scripts/
├── train.py           # ベースライン学習実行
└── test_pipeline.py   # 既存の統合テスト

outputs/               # 学習結果出力
├── models/            # 学習済みモデル
├── logs/              # 学習ログ
└── submissions/       # 提出ファイル（後のフェーズ）
```

## 技術仕様

### モデル設定

```python
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
```

### CV設定

```python
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42,
    'stratify': True
}
```

### 評価基準

- **主要指標**: Accuracy（Kaggleコンペの評価指標）
- **目標値**: CV 0.975以上
- **許容誤差**: CV標準偏差 ≤ 0.002（安定性確保）

## 実装順序

1. **validation.py実装**
   - CV戦略定義
   - 評価指標計算機能
   - ログ機能

2. **models.py実装**
   - LightGBMModelクラス
   - CrossValidationTrainerクラス
   - ユーティリティ関数

3. **train.py実装**
   - メインフロー実装
   - データ読み込み・前処理
   - 結果出力・保存

4. **統合テスト**
   - `make quick-test`で動作確認
   - CV結果の妥当性チェック
   - 既存テストとの整合性確認

## 成功判定基準

### 定量的基準
- ✅ CV Accuracy ≥ 0.975
- ✅ CV標準偏差 ≤ 0.002
- ✅ 学習時間 ≤ 5分（フル5-fold CV）
- ✅ 全テストパス（pytest）

### 定性的基準
- ✅ コードの可読性・保守性
- ✅ エラーハンドリングの適切性
- ✅ ログ・通知の有用性
- ✅ 再現性の確保

## 次フェーズへの準備

このベースライン実装完了により、次の展開が可能となります：

- **フェーズ5**: 特徴量リファクタリング
- **フェーズ6**: 検証戦略ロック
- **フェーズ7**: ハイパーパラメータ最適化

安定したベースラインCV結果が、後続フェーズでの改善効果測定の基準となります。

## 依存関係詳細仕様

### src.utilモジュール構成

#### 1. `src/util/time_tracker.py` - 学習時間計測
```python
"""
WorkflowTimeTracker: 学習フロー全体の時間計測
- start_workflow(): ワークフロー開始時間記録
- start_phase(name): フェーズ開始時間記録
- end_phase(name): フェーズ終了・経過時間計算
- get_summary(): 全体サマリー取得
- to_dict(): 辞書形式での時間データ出力
"""
```

**主要機能**:
- CV全体の実行時間計測（5分以内基準）
- フォールド別学習時間記録
- メモリ使用量監視（オプション）
- 時間統計の自動計算・ログ出力

#### 2. `src/util/notifications.py` - 通知システム
```python
"""
WebhookNotifier: Discord Webhook通知
- notify_training_start(model, config): 学習開始通知
- notify_training_complete(model, score, duration): 完了通知
- notify_error(error_message): エラー通知
- format_metrics(metrics): メトリクス整形
"""
```

**主要機能**:
- 学習開始・完了・エラーの自動通知
- CV結果の整形・送信
- 通知失敗時の graceful degradation
- 設定ファイルによるWebhook URL管理

## 定性的基準の測定方法

### コード品質評価基準

#### 1. 可読性指標
- **関数長**: 50行以内（複雑な処理は分割）
- **クラス長**: 200行以内（単一責任原則）
- **コメント率**: 重要部分に適切な説明
- **命名規則**: PEP8準拠、意図明確な変数名

#### 2. 保守性指標
- **循環複雑度**: 関数あたり10以下
- **結合度**: モジュール間の依存関係最小化
- **凝集度**: 関連機能の適切なグループ化
- **テストカバレッジ**: 主要パス95%以上

#### 3. 測定ツール・手法
```bash
# コード品質チェック
flake8 src/ tests/           # PEP8準拠確認
mypy src/                    # 型チェック
pytest --cov=src tests/      # テストカバレッジ測定
```

#### 4. レビューチェックリスト
- [ ] 関数・クラスのDocstring完備
- [ ] エラーハンドリング適切
- [ ] ログ出力の有用性
- [ ] 設定の外部化
- [ ] ハードコード値の排除

### 再現性保証基準

#### 1. 必須要件
- **乱数シード固定**: 全ランダム処理で`random_state=42`
- **環境依存性排除**: バージョン固定、相対パス使用
- **データアクセス一貫性**: DuckDB経由のみ、CSVファイル禁止

#### 2. 検証手法
- 同一環境での複数回実行結果比較
- CV結果の標準偏差≤0.002達成確認
- ログファイルによる実行履歴追跡

## 参考情報

- **既存成果**: データ管理システム（Bronze/Silver/Gold）
- **利用データ**: Gold層データ（`src/data/gold.py`）
- **実行環境**: `make quick-test`コマンド経由
- **通知**: Discord Webhook（オプション）
- **時間計測**: `src/util/time_tracker.py`活用

============

## 設計書とテストコード整合性確認 ✅

### 🟢 解決済み項目

1. **LightGBMパラメータ統一完了**
   - 設計書仕様（learning_rate: 0.1, num_leaves: 31等）にテストコード準拠
   - 全必須パラメータ（metric, boosting_type, feature_fraction等）テスト確認済み

2. **学習時間基準統一完了**
   - 設計書仕様：5分以内 ← テストコード：5分以内（300秒）に修正完了

3. **依存関係仕様明文化完了**
   - src.utilモジュール（time_tracker, notifications）の詳細仕様追加
   - API仕様・主要機能・使用方法を具体化

4. **定性的基準測定方法定義完了**
   - コード品質評価基準（可読性・保守性指標）具体化
   - 測定ツール・チェックリスト・検証手法明記
   - 再現性保証基準の定量化

### ✅ 一貫して整合している項目

- 基本的なCV戦略・評価指標（StratifiedKFold, Accuracy）
- ファイル構成・実装順序（validation.py → models.py → train.py）
- 核となる成功基準（CV 0.975、標準偏差≤0.002）
- データアクセスパターン（DuckDB Gold層データ利用）

### 📋 実装準備完了

全ての不整合が解決され、以下が保証されます：
- **パラメータ一貫性**: 設計書⟷テストコード完全同期
- **時間基準統一**: 5分以内の学習時間制約
- **品質基準明確化**: 具体的測定手法・ツール定義
- **依存関係透明化**: utilモジュール仕様完全文書化

実装時の混乱・手戻りリスクが最小化され、スムーズな開発進行が可能です。
