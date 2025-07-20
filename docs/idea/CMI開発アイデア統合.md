# CMI開発アイデア統合ドキュメント

## 🎯 プロジェクト概要

**目標**: Kaggle CMI - センサーデータでBFRB行動検出でブロンズメダル達成（LB 0.60+）
- **コンペティション**: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data
- **問題**: 多モーダルセンサーデータからのBFRB行動検出（二値+多クラス分類）
- **評価指標**: 0.5 × (Binary F1 + Macro F1) - カスタムF1複合スコア
- **締切**: 2025年8月26日（残り約5週間）
- **データ規模**: ~200参加者 × 数百セッション（~1.5GB）

---

## 📊 時系列センサーデータの特徴

### データ構造
- **サンプリングレート**: 50Hz
- **IMU**: acc_x/y/z, gyro_x/y/z (加速度・ジャイロスコープ)
- **ToF**: tof_0-3 (4チャンネル距離センサー)
- **サーモパイル**: thermopile_0-4 (5チャンネル温度センサー)
- **GroupKFold必須**: participant_idによる参加者別クロスバリデーション

### 主要課題
1. **多モーダル融合**: IMU + ToF + サーモパイル統合
2. **クラス不均衡**: 二値検出 + 多クラス分類
3. **時間依存性**: 50Hzでの連続行動パターン
4. **参加者リーク防止**: GroupKFold(participant_id)の厳守

---

## 🛠️ 技術的実装アイデア

### 1. 時間記録システム
```python
# JSONベースの軽量実行時間トラッカー
class WorkflowTimeTracker:
    def __init__(self, db_path: str = "workflow_times.json"):
        self.db_path = db_path
        self.data = self._load_data()

    def start_workflow(self, workflow_name: str) -> float:
        """ワークフロー開始 + 推定完了時刻表示"""
        estimated_duration = self.get_estimated_duration(workflow_name)
        if estimated_duration:
            estimated_end = datetime.now() + timedelta(seconds=estimated_duration)
            print(f"⏱️  Estimated completion: {estimated_end.strftime('%H:%M:%S')}")
        return time.time()

    def end_workflow(self, workflow_name: str, start_time: float):
        """実行時間記録 + 統計更新"""
        duration = time.time() - start_time
        # 履歴保存（最新100件のみ）
        self._update_statistics(workflow_name)
```

### 2. センサー特徴量ブレーカー
```python
# 時系列センサーデータ用高度特徴量
def create_sensor_features(df):
    """多モーダルセンサー統合特徴量"""
    
    # IMU特徴量
    df['imu_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['gyro_magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
    
    # ToF距離パターン
    df['tof_avg'] = df[['tof_0', 'tof_1', 'tof_2', 'tof_3']].mean(axis=1)
    df['tof_std'] = df[['tof_0', 'tof_1', 'tof_2', 'tof_3']].std(axis=1)
    
    # サーモパイル温度パターン
    df['thermopile_avg'] = df[['thermopile_0', 'thermopile_1', 'thermopile_2', 
                               'thermopile_3', 'thermopile_4']].mean(axis=1)
    
    # 多モーダル交互作用
    df['imu_tof_ratio'] = df['imu_magnitude'] / (df['tof_avg'] + 1e-6)
    df['thermal_movement'] = df['thermopile_avg'] * df['gyro_magnitude']
    
    return df
```

### 3. ウィンドウベース特徴量抽出
```python
def extract_windowed_features(df, window_size_sec=2, overlap=0.5):
    """時系列ウィンドウ特徴量（50Hzサンプリング対応）"""
    window_samples = int(window_size_sec * 50)  # 2秒 = 100サンプル
    stride = int(window_samples * (1 - overlap))
    
    features = []
    for i in range(0, len(df) - window_samples, stride):
        window = df.iloc[i:i+window_samples]
        
        # 統計特徴量
        stat_feats = {
            'mean': window.select_dtypes(include=[np.number]).mean(),
            'std': window.select_dtypes(include=[np.number]).std(),
            'min': window.select_dtypes(include=[np.number]).min(),
            'max': window.select_dtypes(include=[np.number]).max(),
        }
        
        # FFT特徴量（周波数成分）
        for col in ['acc_x', 'acc_y', 'acc_z']:
            fft_vals = np.fft.fft(window[col].values)
            power_spectrum = np.abs(fft_vals[:len(fft_vals)//2])
            stat_feats[f'{col}_fft_peak'] = np.argmax(power_spectrum)
            stat_feats[f'{col}_fft_power'] = np.sum(power_spectrum)
        
        features.append(stat_feats)
    
    return pd.DataFrame(features)
```

---

## 🏗️ 開発・実験管理

### ブランチ運用戦略
```bash
# 実験用ブランチ命名規則
exp/20250720_imu_baseline        # 日付 + 内容
exp/multimodal_fusion_v1         # 機能名 + バージョン
feat/tof_feature_engineering     # 特徴量追加
fix/groupkfold_leakage          # バグ修正

# 提出用タグ
git tag -a sub_20250720_01 -m "LB 0.58 - IMU baseline"
git tag -a sub_20250721_02 -m "LB 0.61 - Multimodal fusion"
```

### MLflow統合（実験管理）
```python
# 最小限のMLflow統合
from utils.mlflow_setup import init_mlflow
import mlflow.lightgbm

with init_mlflow(run_name="multimodal_lgb_v1"):
    # モデル訓練
    mlflow.lightgbm.autolog()
    model = train_model(X_train, y_train)
    
    # カスタムメトリクス
    mlflow.log_metric("cv_binary_f1", cv_binary_f1)
    mlflow.log_metric("cv_macro_f1", cv_macro_f1)
    mlflow.log_metric("cv_composite_f1", cv_composite_f1)
    mlflow.log_metric("lb_score", lb_score)  # 提出後に追記
    
    # 特徴量重要度
    mlflow.log_artifact("feature_importance.png")
```

---

## 🚀 実装優先順位（5週間計画）

### Week 1: 基盤構築（目標: LB 0.50+）
1. **Bronze層実装**
   - センサー正規化パイプライン
   - 欠損値処理（ToF/サーモパイル）
   - GroupKFold CV設定

2. **Silver層ベースライン**
   - tsfresh統計特徴量
   - LightGBM単一モデル
   - 時間記録システム導入

### Week 2: 深層学習統合（目標: LB 0.57-0.60）
1. **1D CNN実装**
   - InceptionTimeアーキテクチャ
   - 単一モーダル（IMU）ベースライン

2. **データ拡張**
   - 時間シフト
   - 回転拡張（IMU）
   - ノイズ注入

### Week 3: 多モーダル融合（目標: LB 0.62+）
1. **多ブランチアーキテクチャ**
   - IMU/ToF/サーモパイル別ブランチ
   - 特徴量レベル融合

2. **高度特徴量**
   - モーダル間相関
   - 注意機構

### Week 4-5: 最適化（ブロンズメダル確保）
1. **特徴量選択**
   - チャンネル重要度解析
   - 推論速度最適化

2. **メトリクス最適化**
   - 二値 vs 多クラス重み調整
   - 閾値チューニング

---

## 🎯 具体的実装コマンド

### 開発ワークフロー
```bash
# 基本セットアップ
make install              # 依存関係インストール
make setup               # ディレクトリ構造作成

# 訓練パイプライン
make train-fast-dev      # 高速開発訓練
make train-full-optimized # 完全最適化訓練
make train-max-performance # 最大性能訓練

# 検証・予測
make validate-cv         # クロスバリデーション
make predict-basic       # 基本提出
make predict-gold        # 最適化提出

# 実験管理
make mlflow-ui           # MLflow UI起動
make time-report         # 実行時間レポート
```

### 期待スコア改善
| フェーズ | 機能 | 期待改善 | 累積目標 |
|---------|-----|---------|---------|
| Week 1 | 基盤 + tsfresh | +0.10 | 0.50 |
| Week 2 | 1D CNN | +0.07-0.10 | 0.57-0.60 |
| Week 3 | 多モーダル融合 | +0.02-0.05 | 0.62+ |
| Week 4-5 | 最適化 | +0.02-0.03 | 0.64+ (ブロンズ確保) |

---

## 🔧 ユーティリティ機能

### 1. メモリ監視
```python
class MemoryMonitor:
    def monitor(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            before = self.get_memory_usage()
            result = func(*args, **kwargs)
            after = self.get_memory_usage()
            print(f"💾 {func.__name__}: {before:.1f}MB → {after:.1f}MB")
            return result
        return wrapper
```

### 2. データ品質チェック
```python
class SensorDataQualityChecker:
    def check_sensor_data(self, df):
        """センサーデータ固有の品質チェック"""
        issues = []
        
        # サンプリングレートチェック（50Hz期待）
        if 'timestamp' in df.columns:
            time_diffs = df['timestamp'].diff().dropna()
            expected_interval = 1/50  # 50Hz = 20ms間隔
            actual_interval = time_diffs.median()
            if abs(actual_interval - expected_interval) > 0.001:
                issues.append(f"Irregular sampling rate: {actual_interval:.3f}s vs {expected_interval:.3f}s")
        
        # センサー範囲チェック
        for sensor_prefix in ['acc_', 'gyro_', 'tof_', 'thermopile_']:
            sensor_cols = [col for col in df.columns if col.startswith(sensor_prefix)]
            for col in sensor_cols:
                if df[col].isnull().sum() > len(df) * 0.1:  # 10%以上欠損
                    issues.append(f"High missing rate in {col}: {df[col].isnull().mean():.1%}")
        
        return issues
```

### 3. 提出ファイル検証
```python
class CMISubmissionValidator:
    def validate_cmi_submission(self, submission_df):
        """CMI固有の提出ファイル検証"""
        issues = []
        
        # 二値 + 多クラス予測の検証
        if 'binary_prediction' in submission_df.columns:
            binary_vals = submission_df['binary_prediction'].unique()
            if not set(binary_vals).issubset({0, 1}):
                issues.append(f"Binary predictions must be 0 or 1, got: {binary_vals}")
        
        # 予測確率の範囲チェック
        prob_cols = [col for col in submission_df.columns if col.startswith('prob_')]
        for col in prob_cols:
            if (submission_df[col] < 0).any() or (submission_df[col] > 1).any():
                issues.append(f"Probabilities in {col} out of [0,1] range")
        
        return issues
```

---

## 💡 成功のための重要ポイント

### 1. GroupKFoldの徹底
- **必須**: participant_idによる厳格な分割
- **データリーク防止**: 参加者間の情報漏洩を完全遮断
- **CV-LB一貫性**: 適切なGroupKFoldでPrivate LBとの整合性確保

### 2. 多モーダル統合戦略
- **段階的統合**: IMU → IMU+ToF → IMU+ToF+サーモパイル
- **ドメイン知識活用**: 各センサーの物理的意味を理解した特徴量設計
- **融合レベル**: データレベル・特徴量レベル・決定レベルの組み合わせ

### 3. 時系列処理最適化
- **ウィンドウサイズ**: 行動継続時間（2-5秒）に基づく設定
- **オーバーラップ**: 訓練時80%、推論時50%のオーバーラップ
- **拡張手法**: 時系列特有の拡張（時間シフト、速度変更等）

---

## 🏆 ブロンズメダル達成戦略

**目標**: LB 0.60+（推定順位109-216/360チーム）

### 成功要因
1. **技術的優位性**: 多モーダル融合による情報統合
2. **実装品質**: GroupKFoldによるリーク防止
3. **時間管理**: 5週間での段階的スコア向上
4. **実験効率**: MLflow + 時間記録による効率的PDCAサイクル

### リスク対策
1. **計算資源**: GPU不要の軽量化優先
2. **過学習**: GroupKFoldによる堅牢な検証
3. **締切管理**: 週次マイルストーンによる進捗管理
4. **バックアップ**: 複数アプローチの並行開発

---

このアイデア統合により、CMIコンペティションでのブロンズメダル達成に向けた包括的な開発戦略が整備されました。