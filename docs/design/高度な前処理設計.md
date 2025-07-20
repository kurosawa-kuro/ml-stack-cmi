# 高度な前処理設計ドキュメント

## 概要

Kaggle Playground Series S5E7（性格予測コンペティション）におけるブロンズ層の精度向上を目的とした、3つの高度な前処理手法の詳細設計書。

**目標**: 現在のスコア0.9749から銅メダルライン0.976518（+0.0016）を確実に達成

## 実装対象

### 1. 高度な欠損パターン分析
### 2. クロス特徴量補完戦略  
### 3. 異常値検出の高度化

---

## 1. 高度な欠損パターン分析

### 設計思想
- **体系的欠損 vs ランダム欠損**の区別
- **条件付き欠損フラグ**による情報活用
- **相関パターン**を活用した欠損パターン識別

### 実装詳細

```python
def advanced_missing_pattern_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    高度な欠損パターン分析と条件付き欠損フラグ生成
    
    Args:
        df: 入力データフレーム
        
    Returns:
        欠損パターンフラグが追加されたデータフレーム
        
    Expected Impact: +0.3-0.5%
    """
    df = df.copy()
    
    # 1. 基本的な欠損フラグ（既存実装の拡張）
    high_impact_features = [
        'Stage_fear', 'Drained_after_socializing', 'Going_outside',
        'Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size'
    ]
    
    for col in high_impact_features:
        if col in df.columns:
            missing_flag_col = f"{col}_missing"
            df[missing_flag_col] = df[col].isna().astype('int32')
    
    # 2. 条件付き欠損フラグ（高相関パターン）
    if 'Stage_fear' in df.columns and 'Drained_after_socializing' in df.columns:
        # 両方同時欠損（社会的不安の完全回避パターン）
        df['social_anxiety_complete_missing'] = (
            df['Stage_fear'].isna() & df['Drained_after_socializing'].isna()
        ).astype('int32')
        
        # 片方のみ欠損（部分的回避パターン）
        df['social_anxiety_partial_missing'] = (
            df['Stage_fear'].isna() ^ df['Drained_after_socializing'].isna()  # XOR
        ).astype('int32')
        
        # 社会的疲労関連欠損（内向性指標）
        df['social_fatigue_missing'] = (
            df['Drained_after_socializing'].isna() & 
            (df['Social_event_attendance'] > df['Social_event_attendance'].quantile(0.7))
        ).astype('int32')
    
    # 3. 行動パターン関連欠損
    if 'Time_spent_Alone' in df.columns and 'Social_event_attendance' in df.columns:
        # 高孤独時間 + ソーシャル欠損（極端な内向性）
        df['extreme_introvert_missing'] = (
            (df['Time_spent_Alone'] > df['Time_spent_Alone'].quantile(0.8)) & 
            df['Social_event_attendance'].isna()
        ).astype('int32')
        
        # 低孤独時間 + ソーシャル欠損（矛盾パターン）
        df['contradictory_social_missing'] = (
            (df['Time_spent_Alone'] < df['Time_spent_Alone'].quantile(0.2)) & 
            df['Social_event_attendance'].isna()
        ).astype('int32')
    
    # 4. 外出パターン関連欠損
    if 'Going_outside' in df.columns:
        # 高外出頻度 + ソーシャルイベント欠損（非ソーシャル外出）
        df['non_social_outing_missing'] = (
            (df['Going_outside'] > df['Going_outside'].quantile(0.7)) & 
            df['Social_event_attendance'].isna()
        ).astype('int32')
    
    # 5. コミュニケーション関連欠損
    if 'Post_frequency' in df.columns and 'Friends_circle_size' in df.columns:
        # 高投稿頻度 + 友達数欠損（オンライン中心）
        df['online_centric_missing'] = (
            (df['Post_frequency'] > df['Post_frequency'].quantile(0.7)) & 
            df['Friends_circle_size'].isna()
        ).astype('int32')
    
    # 6. 複合欠損パターン（3つ以上の同時欠損）
    social_cols = ['Stage_fear', 'Drained_after_socializing', 'Social_event_attendance']
    missing_counts = df[social_cols].isna().sum(axis=1)
    df['multiple_social_missing'] = (missing_counts >= 2).astype('int32')
    
    return df
```

### 期待効果の根拠
- **社会的不安パターン**: 内向性予測の重要な指標
- **行動矛盾パターン**: データ品質問題の検出
- **複合欠損**: より複雑な性格パターンの識別

---

## 2. クロス特徴量補完戦略

### 設計思想
- **相関ベース**のインテリジェント補完
- **行動パターン**を活用した推定
- **LightGBM最適化**を考慮した補完値選択

### 実装詳細

```python
def cross_feature_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    クロス特徴量を活用したインテリジェント補完戦略
    
    Args:
        df: 入力データフレーム
        
    Returns:
        補完されたデータフレーム
        
    Expected Impact: +0.2-0.4%
    """
    df = df.copy()
    
    # 1. 社会的疲労関連補完（Stage_fear ↔ Drained_after_socializing）
    if 'Stage_fear' in df.columns and 'Drained_after_socializing' in df.columns:
        # 相関計算（非欠損データのみ）
        valid_mask = ~(df['Stage_fear'].isna() | df['Drained_after_socializing'].isna())
        if valid_mask.sum() > 10:  # 十分なデータがある場合
            correlation = df.loc[valid_mask, ['Stage_fear', 'Drained_after_socializing']].corr().iloc[0,1]
            
            # 高相関の場合（>0.3）、片方から他方を推定
            if abs(correlation) > 0.3:
                # Stage_fear欠損 → Drained_after_socializingから推定
                stage_missing = df['Stage_fear'].isna() & ~df['Drained_after_socializing'].isna()
                if stage_missing.any():
                    # 相関に基づく推定
                    if correlation > 0:
                        df.loc[stage_missing, 'Stage_fear'] = df.loc[stage_missing, 'Drained_after_socializing']
                    else:
                        df.loc[stage_missing, 'Stage_fear'] = 1 - df.loc[stage_missing, 'Drained_after_socializing']
                
                # Drained_after_socializing欠損 → Stage_fearから推定
                drain_missing = df['Drained_after_socializing'].isna() & ~df['Stage_fear'].isna()
                if drain_missing.any():
                    if correlation > 0:
                        df.loc[drain_missing, 'Drained_after_socializing'] = df.loc[drain_missing, 'Stage_fear']
                    else:
                        df.loc[drain_missing, 'Drained_after_socializing'] = 1 - df.loc[drain_missing, 'Stage_fear']
    
    # 2. 行動パターンベース補完
    if 'Going_outside' in df.columns and 'Social_event_attendance' in df.columns:
        # 外出頻度からソーシャルイベント参加を推定
        outside_high = df['Going_outside'] > df['Going_outside'].quantile(0.7)
        social_missing = df['Social_event_attendance'].isna()
        
        if (outside_high & social_missing).any():
            # 高外出頻度グループのソーシャルイベント参加率
            high_outside_social = df.loc[outside_high & ~social_missing, 'Social_event_attendance'].median()
            df.loc[outside_high & social_missing, 'Social_event_attendance'] = high_outside_social
        
        # 逆方向の推定
        social_high = df['Social_event_attendance'] > df['Social_event_attendance'].quantile(0.7)
        outside_missing = df['Going_outside'].isna()
        
        if (social_high & outside_missing).any():
            # 高ソーシャル参加グループの外出頻度
            high_social_outside = df.loc[social_high & ~outside_missing, 'Going_outside'].median()
            df.loc[social_high & outside_missing, 'Going_outside'] = high_social_outside
    
    # 3. 時間配分ベース補完
    if 'Time_spent_Alone' in df.columns:
        # 孤独時間から他の活動を推定
        alone_high = df['Time_spent_Alone'] > df['Time_spent_Alone'].quantile(0.8)
        
        # 高孤独時間 → 低ソーシャル活動
        if 'Social_event_attendance' in df.columns:
            social_missing_alone = df['Social_event_attendance'].isna() & alone_high
            if social_missing_alone.any():
                low_social_value = df.loc[alone_high & ~df['Social_event_attendance'].isna(), 'Social_event_attendance'].quantile(0.25)
                df.loc[social_missing_alone, 'Social_event_attendance'] = low_social_value
        
        # 高孤独時間 → 低外出頻度
        if 'Going_outside' in df.columns:
            outside_missing_alone = df['Going_outside'].isna() & alone_high
            if outside_missing_alone.any():
                low_outside_value = df.loc[alone_high & ~df['Going_outside'].isna(), 'Going_outside'].quantile(0.25)
                df.loc[outside_missing_alone, 'Going_outside'] = low_outside_value
    
    # 4. 友達数ベース補完
    if 'Friends_circle_size' in df.columns:
        # 友達数からソーシャル活動を推定
        friends_high = df['Friends_circle_size'] > df['Friends_circle_size'].quantile(0.7)
        
        if 'Social_event_attendance' in df.columns:
            social_missing_friends = df['Social_event_attendance'].isna() & friends_high
            if social_missing_friends.any():
                high_social_value = df.loc[friends_high & ~df['Social_event_attendance'].isna(), 'Social_event_attendance'].quantile(0.75)
                df.loc[social_missing_friends, 'Social_event_attendance'] = high_social_value
    
    # 5. 投稿頻度ベース補完
    if 'Post_frequency' in df.columns:
        # 高投稿頻度 → 高友達数（オンライン社交性）
        post_high = df['Post_frequency'] > df['Post_frequency'].quantile(0.7)
        
        if 'Friends_circle_size' in df.columns:
            friends_missing_post = df['Friends_circle_size'].isna() & post_high
            if friends_missing_post.any():
                high_friends_value = df.loc[post_high & ~df['Friends_circle_size'].isna(), 'Friends_circle_size'].quantile(0.75)
                df.loc[friends_missing_post, 'Friends_circle_size'] = high_friends_value
    
    return df
```

### 期待効果の根拠
- **相関ベース推定**: 統計的に妥当な補完
- **行動パターン活用**: ドメイン知識の反映
- **段階的補完**: 情報の段階的活用

---

## 3. 異常値検出の高度化

### 設計思想
- **Isolation Forest**による異常値検出
- **統計的手法**との組み合わせ
- **LightGBM最適化**を考慮した異常値処理

### 実装詳細

```python
def advanced_outlier_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    高度な異常値検出と処理
    
    Args:
        df: 入力データフレーム
        
    Returns:
        異常値フラグが追加されたデータフレーム
        
    Expected Impact: +0.2-0.3%
    """
    from sklearn.ensemble import IsolationForest
    import numpy as np
    
    df = df.copy()
    
    # 対象となる数値特徴量
    numeric_cols = [
        'Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
        'Friends_circle_size', 'Post_frequency'
    ]
    
    # 1. Isolation Forest による異常値検出
    try:
        # 数値データのみでIsolation Forest実行
        numeric_data = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # パラメータ調整（contamination=0.05で5%を異常値として検出）
        iso_forest = IsolationForest(
            contamination=0.05,  # 5%を異常値として検出
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        # 異常値スコアの計算
        outlier_scores = iso_forest.fit_predict(numeric_data)
        
        # 異常値フラグ（-1が異常値）
        df['isolation_forest_outlier'] = (outlier_scores == -1).astype('int32')
        
        # 異常値スコアの保存（連続値）
        df['isolation_forest_score'] = iso_forest.decision_function(numeric_data)
        
    except Exception as e:
        print(f"Isolation Forest failed: {e}")
        df['isolation_forest_outlier'] = 0
        df['isolation_forest_score'] = 0
    
    # 2. 統計的異常値検出（改良版IQR）
    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 10:
            # 基本統計量の計算
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # より厳密な境界設定（従来の1.5 → 2.5）
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            
            # 異常値フラグ
            outlier_flag = f"{col}_statistical_outlier"
            df[outlier_flag] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype('int32')
            
            # 異常値スコア（境界からの距離）
            df[f"{col}_outlier_score"] = np.where(
                df[col] < lower_bound,
                (lower_bound - df[col]) / IQR,
                np.where(
                    df[col] > upper_bound,
                    (df[col] - upper_bound) / IQR,
                    0
                )
            )
    
    # 3. Z-score ベース異常値検出
    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 10:
            # Z-score計算
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                z_scores = np.abs((df[col] - mean_val) / std_val)
                
                # Z-score > 3 を異常値として検出
                z_outlier_flag = f"{col}_zscore_outlier"
                df[z_outlier_flag] = (z_scores > 3).astype('int32')
                
                # Z-score値の保存
                df[f"{col}_zscore"] = z_scores
    
    # 4. 複合異常値フラグ
    outlier_flags = [col for col in df.columns if col.endswith('_outlier')]
    if outlier_flags:
        # 複数の異常値検出手法で検出された異常値
        df['multiple_outlier_detected'] = df[outlier_flags].sum(axis=1) >= 2
        df['multiple_outlier_detected'] = df['multiple_outlier_detected'].astype('int32')
        
        # 異常値の総数
        df['total_outlier_count'] = df[outlier_flags].sum(axis=1)
    
    # 5. ドメイン特化異常値検出
    if 'Time_spent_Alone' in df.columns:
        # 24時間を超える異常値
        df['time_alone_extreme'] = (df['Time_spent_Alone'] > 24).astype('int32')
        
        # 負の値（データエラー）
        df['time_alone_negative'] = (df['Time_spent_Alone'] < 0).astype('int32')
    
    if 'Friends_circle_size' in df.columns:
        # 極端に大きな友達数（現実的でない値）
        df['friends_extreme'] = (df['Friends_circle_size'] > 1000).astype('int32')
    
    if 'Post_frequency' in df.columns:
        # 極端に高い投稿頻度
        df['post_frequency_extreme'] = (df['Post_frequency'] > 100).astype('int32')
    
    return df
```

### 期待効果の根拠
- **Isolation Forest**: 非線形な異常値パターンの検出
- **統計的手法**: 線形な異常値の検出
- **ドメイン特化**: 現実的な制約に基づく検出

---

## 統合実装

### 実装順序

```python
def enhanced_bronze_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    3つの高度な前処理を統合したブロンズ層処理
    
    Args:
        df: 生データフレーム
        
    Returns:
        高度な前処理が適用されたデータフレーム
        
    Expected Total Impact: +0.7-1.2%
    """
    # 1. 高度な欠損パターン分析
    df = advanced_missing_pattern_analysis(df)
    
    # 2. クロス特徴量補完戦略
    df = cross_feature_imputation(df)
    
    # 3. 異常値検出の高度化
    df = advanced_outlier_detection(df)
    
    return df
```

### 既存コードとの統合

```python
# src/data/bronze.py の create_bronze_tables() に追加
def create_bronze_tables() -> None:
    """Creates standardized bronze.train, bronze.test tables"""
    conn = duckdb.connect(DB_PATH)
    
    # Create bronze schema
    conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    
    # Load raw data
    train_raw, test_raw = load_data()
    
    # Apply existing bronze layer processing
    train_bronze = encode_categorical_robust(train_raw)
    test_bronze = encode_categorical_robust(test_raw)
    
    train_bronze = advanced_missing_strategy(train_bronze)
    test_bronze = advanced_missing_strategy(test_bronze)
    
    train_bronze = winsorize_outliers(train_bronze)
    test_bronze = winsorize_outliers(test_bronze)
    
    # 新規: 高度な前処理の追加
    train_bronze = enhanced_bronze_preprocessing(train_bronze)
    test_bronze = enhanced_bronze_preprocessing(test_bronze)
    
    # Validate data quality
    train_validation = validate_data_quality(train_bronze)
    test_validation = validate_data_quality(test_bronze)
    
    # Create bronze tables
    conn.execute("DROP TABLE IF EXISTS bronze.train")
    conn.execute("DROP TABLE IF EXISTS bronze.test")
    
    conn.register("train_bronze_df", train_bronze)
    conn.register("test_bronze_df", test_bronze)
    
    conn.execute("CREATE TABLE bronze.train AS SELECT * FROM train_bronze_df")
    conn.execute("CREATE TABLE bronze.test AS SELECT * FROM test_bronze_df")
    
    print("Enhanced Bronze tables created:")
    print(f"- bronze.train: {len(train_bronze)} rows, {len(train_bronze.columns)} columns")
    print(f"- bronze.test: {len(test_bronze)} rows, {len(test_bronze.columns)} columns")
    print(f"- Enhanced features: {len(train_bronze.columns) - len(train_raw.columns)} new columns")
    
    conn.close()
```

---

## 期待効果と検証

### 期待される改善
- **合計期待効果**: +0.7-1.2%
- **現在スコア**: 0.9749
- **予想スコア**: 0.9819-0.9869
- **銅メダルライン**: 0.976518 ✅

### 検証方法
1. **CV スコア測定**: 5-fold StratifiedKFold
2. **CV-LB ギャップ確認**: 0.01以下を目標
3. **特徴量重要度分析**: 新規特徴量の貢献度確認
4. **処理時間測定**: 1秒以下を維持

### 実装チェックリスト
- [ ] 高度な欠損パターン分析の実装
- [ ] クロス特徴量補完戦略の実装
- [ ] 異常値検出の高度化の実装
- [ ] 既存コードとの統合
- [ ] テストケースの作成
- [ ] 効果測定の実行
- [ ] ドキュメントの更新

---

## 注意事項

### 実装時の注意点
1. **データリーク防止**: すべての統計量はCV fold内で計算
2. **再現性確保**: 乱数シードの固定
3. **エラーハンドリング**: 各処理での例外処理
4. **メモリ効率**: 大規模データセットでの最適化

### パフォーマンス考慮
- **処理時間**: 1秒以下を目標
- **メモリ使用量**: 既存の2倍以下
- **スケーラビリティ**: データサイズ増加への対応

この設計により、銅メダル獲得を確実にし、さらなる上位入賞への基盤を構築できます。
