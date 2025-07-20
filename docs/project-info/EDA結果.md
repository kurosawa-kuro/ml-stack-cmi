# CMI BFRB Detection - EDA総括（Claude-Code用）

## 1. データセット概要

### 基本構造
```yaml
train_data:
  rows: 574,945
  participants: 81
  sequences: 8,151
  sampling_rate: 50Hz
  avg_sequence_duration: 1.41秒
  
test_data:
  rows: 107
  participants: 2
  sequences: 2
  
participant_overlap: なし（データリークなし）
```

### センサー構成
```yaml
sensors:
  IMU:
    - acc_x, acc_y, acc_z  # 加速度センサー
    - rot_w, rot_x, rot_y, rot_z  # 回転クォータニオン
    missing_rate: 0.0%（acc）, 0.64%（rot）
    
  Thermopile:
    - thm_1, thm_2, thm_3, thm_4  # 欠損1-2%
    - thm_5  # 欠損5.79%（要注意）
    
  ToF:
    - tof_1, tof_2, tof_3, tof_4  # 各64チャンネル、欠損1%
    - tof_5  # 欠損5.24%（要注意）
```

## 2. ターゲット変数

### 分類タスク
```yaml
binary_classification:
  Target: 59.84%
  Non-Target: 40.16%
  評価指標: Binary F1
  
multi_class_classification:
  classes: 18ジェスチャー
  imbalance_ratio: 5.9:1
  most_frequent: "Text on phone" (10.17%)
  least_frequent: "Pinch knee/leg skin" (1.71%)
  評価指標: Macro F1
  
combined_score: 0.5 × (Binary F1 + Macro F1)
```

### 主要ジェスチャー（頻度順）
1. Text on phone (10.17%)
2. Neck - scratch (9.85%)
3. Eyebrow - pull hair (7.71%)
4. Forehead - scratch (7.12%)
5. Forehead - pull hairline (7.10%)

## 3. 重要な発見事項

### データ品質
```yaml
良好:
  - IMUセンサー: 欠損ほぼゼロ、信頼性高い
  - 時系列連続性: シーケンス内でギャップなし
  - 参加者カバレッジ: 全参加者が全ジェスチャー実行
  
要注意:
  - thm_5とtof_5: 5%以上欠損（センサー故障？）
  - 欠損の共起: thm_5とtof_5の欠損が94%一致
  - 特定参加者: SUBJ_044680とSUBJ_016552は100%欠損
```

### クロスバリデーション設計
```yaml
推奨設定:
  method: GroupKFold
  n_splits: 5
  group_by: subject（参加者ID）
  
理由:
  - 参加者間でデータリークを防ぐ
  - 各foldに約16人の参加者
  - train/testの参加者重複なし
```

## 4. 特徴工学推奨事項

### 優先度1（必須）
```python
# IMU特徴量
features = {
    'acc_magnitude': 'sqrt(acc_x^2 + acc_y^2 + acc_z^2)',
    'velocity': 'diff(acc_x/y/z)',
    'jerk': 'diff(diff(acc_x/y/z))',
    'rolling_stats': 'mean/std/min/max over 5,10,20 timesteps'
}

# 欠損値処理
missing_features = {
    'thm_5_available': 'binary indicator',
    'tof_5_available': 'binary indicator',
    'imputation': 'median of thm_1-4 for thm_5'
}
```

### 優先度2（高インパクト）
```python
# 周波数領域特徴
fft_features = {
    'spectral_energy': 'FFT power in bands',
    'dominant_freq': 'peak frequency',
    'frequency_bands': '[0-2Hz, 2-5Hz, 5-10Hz, 10-25Hz]'
}

# ToF次元削減
tof_features = {
    'pca_components': '8-16 per sensor',
    'statistical_summary': 'mean/std/min/max per sensor',
    'proximity': 'min(tof_N_v0..63)'
}
```

### 優先度3（最適化）
```python
# マルチモーダル融合
fusion_features = {
    'imu_temp_correlation': 'correlation(acc_magnitude, thm_mean)',
    'motion_proximity': 'correlation(acc_jerk, tof_min_distance)',
    'lag_features': 'temperature[t-1] vs acc[t]'
}
```

## 5. モデル戦略

### フェーズ1（Week 1）
```yaml
目標: CV 0.50+, LB 0.50+
手法:
  - GroupKFold CV セットアップ
  - 基本特徴量（IMU magnitude, rolling stats）
  - LightGBM baseline
```

### フェーズ2（Week 2-3）
```yaml
目標: CV 0.58+, LB 0.57+
手法:
  - FFT spectrum features
  - ToF PCA次元削減
  - 1D CNN on raw sensors
  - アンサンブル準備
```

### フェーズ3（Week 4-5）
```yaml
目標: CV 0.62+, LB 0.60+（銅メダル圏）
手法:
  - Multi-branch CNN（モダリティ別）
  - モデルアンサンブル
  - ハイパーパラメータ最適化
```

## 6. リスクと対策

### 主要リスク
```yaml
risks:
  - Macro F1の困難さ: 18クラス不均衡で0.5以下の可能性
  - センサー5欠損: 情報損失の影響
  - CV-LBギャップ: 参加者ベース分割での乖離
  
対策:
  - focal_lossでクラス重み調整
  - 欠損値インジケータ＋適切な補完
  - 複数シードでCV安定性確認
```

## 7. 実装チェックリスト

### 即座に実装
- [ ] GroupKFold CV環境構築
- [ ] IMU magnitude + 微分特徴量
- [ ] 欠損値インジケータ作成
- [ ] ベースラインモデル（LightGBM）

### 今週中に実装
- [ ] Rolling window statistics
- [ ] FFT特徴量抽出
- [ ] ToF PCA実装
- [ ] 時系列拡張（tsfresh）

### 最適化フェーズ
- [ ] 1D CNN実装
- [ ] マルチモーダル融合
- [ ] アンサンブル構築
- [ ] 疑似LB実験

## 8. 予想スコアとマイルストーン

```yaml
week1:
  target: CV 0.50, LB 0.50
  status: ベースライン確立
  
week2-3:
  target: CV 0.58, LB 0.57
  status: 特徴工学完了
  
week4-5:
  target: CV 0.62, LB 0.60
  status: 銅メダル圏到達
  
final:
  binary_f1: 0.65-0.70
  macro_f1: 0.55-0.65
  combined: 0.60-0.68
```

## 9. Claude-Code実装時の注意事項

### データ処理
```python
# 必ずGroupKFoldを使用
cv = GroupKFold(n_splits=5)
groups = train_data['subject']

# 欠損値処理を明示的に
train_data['thm_5_available'] = train_data['thm_5'].notna().astype(int)
train_data['tof_5_available'] = train_data['tof_5_v0'].notna().astype(int)
```

### メモリ効率
```python
# 大規模データのため型最適化
float_cols = ['acc_x', 'acc_y', 'acc_z', ...]
train_data[float_cols] = train_data[float_cols].astype('float32')

# ToFデータは必要に応じてPCA
from sklearn.decomposition import PCA
pca = PCA(n_components=16)
```

### 評価関数
```python
def combined_f1_score(y_true, y_pred_binary, y_pred_multi):
    binary_f1 = f1_score(y_true_binary, y_pred_binary)
    macro_f1 = f1_score(y_true_multi, y_pred_multi, average='macro')
    return 0.5 * (binary_f1 + macro_f1)
```

## 10. 次のアクション

1. **今すぐ**: GroupKFold CV実装とベースライン作成
2. **今日中**: IMU特徴量エンジニアリング
3. **明日**: FFT特徴量とToF次元削減
4. **週末**: 1D CNNプロトタイプ作成

---
**注意**: このEDAは2025年1月時点の分析です。コンペティション進行に応じて戦略を調整してください。