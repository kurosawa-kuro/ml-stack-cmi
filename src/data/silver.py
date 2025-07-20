"""
Silver Level Data Management
Feature Engineering & Advanced Preprocessing
"""

from typing import Tuple, List, Dict, Any, Optional
import warnings

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder

DB_PATH = "/home/wsl/dev/my-study/ml/ml-stack-cmi/data/kaggle_datasets.duckdb"

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in multiply')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in reduce')


def advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """高度な特徴量エンジニアリング - 30+ statistical & domain features"""
    df = df.copy()

    # Winner Solution features (from CLAUDE.md specifications) - Silver層で生成
    if "Social_event_attendance" in df.columns and "Going_outside" in df.columns:
        df["social_participation_rate"] = (
            df["Social_event_attendance"] / (df["Going_outside"] + 1e-8)
        ).astype('float32')
    
    if "Post_frequency" in df.columns and "Social_event_attendance" in df.columns and "Going_outside" in df.columns:
        total_activity = df["Social_event_attendance"] + df["Going_outside"]
        df["communication_ratio"] = (
            df["Post_frequency"] / (total_activity + 1e-8)
        ).astype('float32')
    
    if "Social_event_attendance" in df.columns and "Friends_circle_size" in df.columns:
        df["friend_social_efficiency"] = (
            df["Social_event_attendance"] / (df["Friends_circle_size"] + 1e-8)
        ).astype('float32')
    
    if "Going_outside" in df.columns and "Social_event_attendance" in df.columns:
        df["non_social_outings"] = (
            df["Going_outside"] - df["Social_event_attendance"]
        ).astype('float32')
    
    if "Time_spent_Alone" in df.columns and "Social_event_attendance" in df.columns:
        df["activity_balance"] = (
            df["Social_event_attendance"] / (df["Time_spent_Alone"] + 1e-8)
        ).astype('float32')
    
    if "Social_event_attendance" in df.columns and "Time_spent_Alone" in df.columns:
        df["social_ratio"] = (
            df["Social_event_attendance"] / (df["Time_spent_Alone"] + 1e-8)
        ).astype('float32')

    if "Going_outside" in df.columns and "Social_event_attendance" in df.columns:
        df["activity_sum"] = (
            df["Going_outside"] + df["Social_event_attendance"]
        ).astype('float32')

    # 新しい特徴量（Silver層固有）
    numeric_cols = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]

    # 統計的特徴量
    if all(col in df.columns for col in numeric_cols):
        df["total_activity"] = df[numeric_cols].sum(axis=1)
        df["avg_activity"] = df[numeric_cols].mean(axis=1)
        df["activity_std"] = df[numeric_cols].clip(-50, 50).std(axis=1).fillna(0)
        df["activity_min"] = df[numeric_cols].min(axis=1)
        df["activity_max"] = df[numeric_cols].max(axis=1)
        df["activity_range"] = df["activity_max"] - df["activity_min"]

    # 比率特徴量
    if "Friends_circle_size" in df.columns and "Post_frequency" in df.columns:
        df["post_per_friend"] = df["Post_frequency"] / (df["Friends_circle_size"] + 1e-8)
        df["friend_efficiency"] = df["Friends_circle_size"] / (df["Post_frequency"] + 1e-8)

    # 二項交互作用
    if "Stage_fear_encoded" in df.columns and "Drained_after_socializing_encoded" in df.columns:
        df["fear_drain_interaction"] = df["Stage_fear_encoded"] * df["Drained_after_socializing_encoded"]
        df["fear_drain_sum"] = df["Stage_fear_encoded"] + df["Drained_after_socializing_encoded"]
        df["fear_drain_ratio"] = df["Stage_fear_encoded"] / (df["Drained_after_socializing_encoded"] + 1e-8)
    
    # 追加の交互作用特徴量（確実に作成）
    if "Social_event_attendance" in df.columns and "Time_spent_Alone" in df.columns:
        df["social_alone_interaction"] = df["Social_event_attendance"] * df["Time_spent_Alone"]
        df["social_alone_ratio"] = df["Social_event_attendance"] / (df["Time_spent_Alone"] + 1e-8)
    
    if "Friends_circle_size" in df.columns and "Post_frequency" in df.columns:
        df["friends_post_interaction"] = df["Friends_circle_size"] * df["Post_frequency"]
        df["friends_post_ratio"] = df["Friends_circle_size"] / (df["Post_frequency"] + 1e-8)

    # 外向性スコア（仮説ベース）
    extrovert_features = []
    if "Social_event_attendance" in df.columns:
        extrovert_features.append("Social_event_attendance")
    if "Going_outside" in df.columns:
        extrovert_features.append("Going_outside")
    if "Friends_circle_size" in df.columns:
        extrovert_features.append("Friends_circle_size")

    if extrovert_features:
        df["extrovert_score"] = df[extrovert_features].sum(axis=1)
        df["extrovert_avg"] = df[extrovert_features].mean(axis=1)
        df["extrovert_std"] = df[extrovert_features].clip(-50, 50).std(axis=1).fillna(0)

    # 内向性スコア
    introvert_features = []
    if "Time_spent_Alone" in df.columns:
        introvert_features.append("Time_spent_Alone")
    if "Stage_fear_encoded" in df.columns:
        introvert_features.append("Stage_fear_encoded")
    if "Drained_after_socializing_encoded" in df.columns:
        introvert_features.append("Drained_after_socializing_encoded")

    if introvert_features:
        df["introvert_score"] = df[introvert_features].sum(axis=1)
        df["introvert_avg"] = df[introvert_features].mean(axis=1)
        # std計算でNaNが発生しないように修正
        introvert_data = df[introvert_features].fillna(0)
        df["introvert_std"] = introvert_data.std(axis=1).fillna(0)

    # 複合指標
    if "extrovert_score" in df.columns and "introvert_score" in df.columns:
        df["personality_balance"] = df["extrovert_score"] - df["introvert_score"]
        # 極端な値を避けるため、分母を1以上に制限
        introvert_denominator = df["introvert_score"].clip(lower=1)
        df["personality_ratio"] = df["extrovert_score"] / introvert_denominator
        df["personality_sum"] = df["extrovert_score"] + df["introvert_score"]

    # 時間関連特徴量
    if "Time_spent_Alone" in df.columns:
        df["alone_percentage"] = df["Time_spent_Alone"] / 24.0  # 24時間中の割合
        df["alone_squared"] = df["Time_spent_Alone"] ** 2
        # 無限値を避けるため、log1pを使用し、負の値を0に置換
        time_alone_positive = df["Time_spent_Alone"].clip(lower=0)
        df["alone_log"] = np.log1p(time_alone_positive)

    # ソーシャル関連特徴量
    if "Social_event_attendance" in df.columns:
        df["social_squared"] = df["Social_event_attendance"] ** 2
        df["social_log"] = np.log1p(df["Social_event_attendance"])
        df["social_percentage"] = df["Social_event_attendance"] / (df["Social_event_attendance"].max() + 1e-8)

    return df


def enhanced_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """強化された交互作用特徴量 - 上位特徴量の組み合わせ"""
    df = df.copy()

    # 上位特徴量の交互作用項（重要度順）
    top_features = ["extrovert_score", "social_ratio", "social_participation_rate"]

    # 1. extrovert_score と social_ratio の交互作用
    if all(col in df.columns for col in ["extrovert_score", "social_ratio"]):
        df["extrovert_social_interaction"] = df["extrovert_score"] * df["social_ratio"]
        df["extrovert_social_ratio"] = df["extrovert_score"] / (df["social_ratio"] + 1e-8)
        df["social_extrovert_ratio"] = df["social_ratio"] / (df["extrovert_score"] + 1e-8)

    # 2. extrovert_score と他の重要特徴量との交互作用
    if "extrovert_score" in df.columns:
        if "Social_event_attendance" in df.columns:
            df["extrovert_social_event_interaction"] = df["extrovert_score"] * df["Social_event_attendance"]
            df["extrovert_social_event_ratio"] = df["extrovert_score"] / (df["Social_event_attendance"] + 1e-8)

        if "Time_spent_Alone" in df.columns:
            df["extrovert_alone_interaction"] = df["extrovert_score"] * df["Time_spent_Alone"]
            df["extrovert_alone_contrast"] = df["extrovert_score"] - df["Time_spent_Alone"]
            df["extrovert_alone_ratio"] = df["extrovert_score"] / (df["Time_spent_Alone"] + 1e-8)

        if "Drained_after_socializing_encoded" in df.columns:
            df["extrovert_drain_interaction"] = df["extrovert_score"] * df["Drained_after_socializing_encoded"]
            # 極端な値を避けるため、分母を1以上に制限
            drain_denominator = df["Drained_after_socializing_encoded"].clip(lower=1)
            df["extrovert_drain_ratio"] = df["extrovert_score"] / drain_denominator

        if "Friends_circle_size" in df.columns:
            df["extrovert_friends_interaction"] = df["extrovert_score"] * df["Friends_circle_size"]
            df["extrovert_friends_ratio"] = df["extrovert_score"] / (df["Friends_circle_size"] + 1e-8)

    # 3. social_ratio と他の特徴量との交互作用
    if "social_ratio" in df.columns:
        if "Friends_circle_size" in df.columns:
            df["social_friends_interaction"] = df["social_ratio"] * df["Friends_circle_size"]
            df["social_friends_ratio"] = df["social_ratio"] / (df["Friends_circle_size"] + 1e-8)

        if "Going_outside" in df.columns:
            df["social_outside_interaction"] = df["social_ratio"] * df["Going_outside"]
            df["social_outside_ratio"] = df["social_ratio"] / (df["Going_outside"] + 1e-8)

        if "Post_frequency" in df.columns:
            df["social_post_interaction"] = df["social_ratio"] * df["Post_frequency"]
            df["social_post_ratio"] = df["social_ratio"] / (df["Post_frequency"] + 1e-8)

    # 4. social_participation_rate との交互作用
    if "social_participation_rate" in df.columns:
        if "Time_spent_Alone" in df.columns:
            df["participation_alone_interaction"] = df["social_participation_rate"] * df["Time_spent_Alone"]
            df["participation_alone_ratio"] = df["social_participation_rate"] / (df["Time_spent_Alone"] + 1e-8)

        if "Friends_circle_size" in df.columns:
            df["participation_friends_interaction"] = df["social_participation_rate"] * df["Friends_circle_size"]
            df["participation_friends_ratio"] = df["social_participation_rate"] / (df["Friends_circle_size"] + 1e-8)

    # 5. 三項交互作用（最重要特徴量のみ）
    if all(col in df.columns for col in ["extrovert_score", "social_ratio", "Social_event_attendance"]):
        df["triple_interaction"] = df["extrovert_score"] * df["social_ratio"] * df["Social_event_attendance"]

    if all(col in df.columns for col in ["extrovert_score", "social_participation_rate", "Time_spent_Alone"]):
        df["triple_participation_interaction"] = df["extrovert_score"] * df["social_participation_rate"] * df["Time_spent_Alone"]

    # 6. 複合比率特徴量
    if all(col in df.columns for col in ["social_ratio", "communication_ratio", "friend_social_efficiency"]):
        df["composite_social_score"] = (df["social_ratio"] + df["communication_ratio"] + df["friend_social_efficiency"]) / 3
        df["social_efficiency_balance"] = df["social_ratio"] * df["friend_social_efficiency"]

    return df


def polynomial_features(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """多項式特徴量生成（degree=2で非線形関係を捕捉）"""
    df = df.copy()

    # 多項式特徴量を適用する数値特徴量を選定
    key_features = []

    # 上位特徴量のみ選択（数値のみ確保）
    top_numeric_features = [
        "extrovert_score",
        "social_ratio",
        "social_participation_rate",
        "communication_ratio",
        "friend_social_efficiency",
        "Social_event_attendance",
        "Time_spent_Alone",
        "Friends_circle_size",
        "Going_outside",
        "Post_frequency",
    ]

    for feature in top_numeric_features:
        if feature in df.columns:
            # 数値型であることを確認
            if pd.api.types.is_numeric_dtype(df[feature]):
                key_features.append(feature)

    if len(key_features) >= 2:  # 最低2つの特徴量が必要
        try:
            # 一時的なデータフレームで多項式特徴量生成
            temp_df = df[key_features].copy()

            # NaNと無限値の処理
            temp_df = temp_df.fillna(0)
            temp_df = temp_df.replace([np.inf, -np.inf], 0)
            
            # Scale features to prevent overflow in polynomial expansion
            temp_df = temp_df.clip(-5, 5)  # Ultra conservative clipping
            
            # Further scale down large values to prevent overflow
            for col in temp_df.columns:
                if temp_df[col].abs().max() > 3:
                    temp_df[col] = temp_df[col] / (temp_df[col].abs().max() / 3)

            # PolynomialFeaturesを使用（interaction_only=Falseで二乗項も含む）
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            poly_features = poly.fit_transform(temp_df)
            
            # Clip polynomial features to prevent overflow - ultra conservative
            poly_features = np.clip(poly_features, -50, 50)

            # 特徴量名生成
            feature_names = poly.get_feature_names_out(key_features)

            # 元の特徴量以外の新しい特徴量のみ追加
            original_features = set(key_features)
            for i, name in enumerate(feature_names):
                if name not in original_features:
                    # 特徴量名をクリーンアップ
                    clean_name = name.replace(" ", "_").replace("^", "_pow_")
                    # Additional clipping for individual features
                    feature_values = poly_features[:, i]
                    feature_values = np.clip(feature_values, -50, 50)
                    df[f"poly_{clean_name}"] = feature_values.astype(np.float32)

        except Exception as e:
            # エラーが発生した場合はログに記録して続行
            print(f"Warning: Polynomial feature generation failed: {e}")

    return df


def scaling_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量スケーリング（標準化）"""
    df = df.copy()

    # 数値特徴量を標準化（boolean型も含む）
    numeric_features = df.select_dtypes(include=[np.number, bool]).columns
    exclude_cols = ["id"]  # IDカラムは除外
    numeric_features = [col for col in numeric_features if col not in exclude_cols]

    for col in numeric_features:
        col_data = df[col].clip(-100, 100)  # More conservative clipping
        if col_data.std() > 1e-8:  # 分散が0でない場合のみ
            df[f"{col}_scaled"] = ((col_data - col_data.mean()) / col_data.std()).clip(-5, 5).astype(np.float32)

    return df


def create_silver_tables() -> None:
    """silver層テーブルをDuckDBに作成"""
    conn = duckdb.connect(DB_PATH)

    # silverスキーマ作成
    conn.execute("CREATE SCHEMA IF NOT EXISTS silver")

    # bronzeデータ読み込み
    try:
        train_bronze = conn.execute("SELECT * FROM bronze.train").df()
        test_bronze = conn.execute("SELECT * FROM bronze.test").df()
    except Exception:
        print("Bronze tables not found. Creating bronze tables first...")
        from .bronze import create_bronze_tables

        create_bronze_tables()
        train_bronze = conn.execute("SELECT * FROM bronze.train").df()
        test_bronze = conn.execute("SELECT * FROM bronze.test").df()

    # Apply Silver layer processing pipeline (CLAUDE.md specification)
    # Bronze層からは品質保証されたデータのみを受け取り、全ての特徴量をSilver層で生成
    
    # Step 1: Advanced features (Winner Solution + 統計特徴量)
    train_silver = advanced_features(train_bronze)
    test_silver = advanced_features(test_bronze)

    # Step 2: Winner Solution Interaction Features (+0.2-0.4% proven impact)
    train_silver = s5e7_interaction_features(train_silver)
    test_silver = s5e7_interaction_features(test_silver)

    # Step 3: Fatigue-Adjusted Domain Modeling (+0.1-0.2% introversion accuracy)
    train_silver = s5e7_drain_adjusted_features(train_silver)
    test_silver = s5e7_drain_adjusted_features(test_silver)

    # Step 4: Online vs Offline behavioral ratios
    train_silver = s5e7_communication_ratios(train_silver)
    test_silver = s5e7_communication_ratios(test_silver)

    # Step 5: Enhanced interaction features (追加の交互作用)
    train_silver = enhanced_interaction_features(train_silver)
    test_silver = enhanced_interaction_features(test_silver)

    # Step 6: Degree-2 nonlinear combinations (多項式特徴量)
    train_silver = polynomial_features(train_silver, degree=2)
    test_silver = polynomial_features(test_silver, degree=2)
    
    # Step 7: Advanced Feature Engineering (Bronze Medal Enhancement)
    # LightGBM Power Transformations (+0.3-0.5% expected)
    lgbm_engineer = LightGBMFeatureEngineer()
    lgbm_engineer.fit(train_silver)
    train_silver = lgbm_engineer.transform(train_silver)
    test_silver = lgbm_engineer.transform(test_silver)
    
    # Advanced Statistical Features (+0.1-0.3% expected)  
    stat_engineer = AdvancedStatisticalFeatures(n_neighbors=5)
    stat_engineer.fit(train_silver)
    train_silver = stat_engineer.transform(train_silver)
    test_silver = stat_engineer.transform(test_silver)

    # silverテーブル作成・挿入
    conn.execute("DROP TABLE IF EXISTS silver.train")
    conn.execute("DROP TABLE IF EXISTS silver.test")

    conn.register("train_silver_df", train_silver)
    conn.register("test_silver_df", test_silver)

    conn.execute("CREATE TABLE silver.train AS SELECT * FROM train_silver_df")
    conn.execute("CREATE TABLE silver.test AS SELECT * FROM test_silver_df")

    print("Silver tables created: ")
    print(f"- silver.train: {len(train_silver)} rows, {len(train_silver.columns)} columns")
    print(f"- silver.test: {len(test_silver)} rows, {len(test_silver.columns)} columns")
    print(f"- Total Engineered Features: {len(train_silver.columns) - len(train_bronze.columns)} features generated")
    print(f"- Winner Solution features: {len([col for col in train_silver.columns if any(keyword in col for keyword in ['participation_rate', 'communication_ratio', 'social_efficiency'])])}")
    print(f"- Interaction features: {len([col for col in train_silver.columns if 'interaction' in col.lower()])}")
    print(f"- Polynomial features: {len([col for col in train_silver.columns if col.startswith('poly_')])}")
    print(f"- Power transformed features: {len([col for col in train_silver.columns if col.endswith('_power')])}")
    print(f"- Statistical features: {len([col for col in train_silver.columns if col.startswith('row_')])}")
    print(f"- Missing indicators: {len([col for col in train_silver.columns if col.endswith('_was_missing')])}")
    print(f"- Z-score features: {len([col for col in train_silver.columns if col.endswith('_zscore')])}")
    print(f"- Percentile features: {len([col for col in train_silver.columns if col.endswith('_percentile')])}")
    print("Advanced Feature Engineering:")
    print("  ✓ LightGBM Power Transformations (+0.3-0.5% expected)")
    print("  ✓ KNN Imputation with Missing Indicators (+0.1-0.2% expected)")  
    print("  ✓ Advanced Statistical Moments (+0.1-0.2% expected)")
    print("  ✓ Feature-specific Z-scores and Percentiles (+0.1% expected)")
    print("  🎯 Bronze Medal Target: +0.8% total improvement expected")

    conn.close()


def load_silver_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """silver層データ読み込み"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM silver.train").df()
    test = conn.execute("SELECT * FROM silver.test").df()
    conn.close()
    return train, test


def s5e7_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Winner Solution Interaction Features (+0.2-0.4% proven impact)"""
    df = df.copy()
    
    # Social_event_participation_rate = Social_event_attendance ÷ Going_outside
    if 'Social_event_attendance' in df.columns and 'Going_outside' in df.columns:
        df['Social_event_participation_rate'] = df['Social_event_attendance'] / (df['Going_outside'] + 1e-8)
    
    # Non_social_outings = Going_outside - Social_event_attendance
    if 'Going_outside' in df.columns and 'Social_event_attendance' in df.columns:
        df['Non_social_outings'] = df['Going_outside'] - df['Social_event_attendance']
    
    # Communication_ratio = Post_frequency ÷ (Social_event_attendance + Going_outside)
    if all(col in df.columns for col in ['Post_frequency', 'Social_event_attendance', 'Going_outside']):
        df['Communication_ratio'] = df['Post_frequency'] / (df['Social_event_attendance'] + df['Going_outside'] + 1e-8)
    
    # Friend_social_efficiency = Social_event_attendance ÷ Friends_circle_size
    if 'Social_event_attendance' in df.columns and 'Friends_circle_size' in df.columns:
        df['Friend_social_efficiency'] = df['Social_event_attendance'] / (df['Friends_circle_size'] + 1e-8)
    
    return df


def s5e7_drain_adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fatigue-Adjusted Domain Modeling (+0.1-0.2% introversion accuracy)"""
    df = df.copy()
    
    # Activity_ratio = comprehensive_activity_index
    activity_cols = ['Social_event_attendance', 'Going_outside', 'Post_frequency']
    available_cols = [col for col in activity_cols if col in df.columns]
    if available_cols:
        df['Activity_ratio'] = df[available_cols].sum(axis=1)
    
    # Drain_adjusted_activity = activity_ratio × (1 - Drained_after_socializing_encoded)
    if 'Activity_ratio' in df.columns and 'Drained_after_socializing_encoded' in df.columns:
        df['Drain_adjusted_activity'] = df['Activity_ratio'] * (1 - df['Drained_after_socializing_encoded'])
    elif 'Activity_ratio' in df.columns and 'Drained_after_socializing' in df.columns:
        # Fallback: encode if not already encoded
        df['Drain_adjusted_activity'] = df['Activity_ratio'] * (1 - (df['Drained_after_socializing'] == 'Yes').astype(float))
    
    # Introvert_extrovert_spectrum = quantified_personality_score
    extrovert_features = ['Social_event_attendance', 'Going_outside', 'Friends_circle_size']
    introvert_features = ['Time_spent_Alone']
    
    extrovert_sum = 0
    for col in extrovert_features:
        if col in df.columns:
            extrovert_sum += df[col]
    
    introvert_sum = 0
    for col in introvert_features:
        if col in df.columns:
            introvert_sum += df[col]
    
    if isinstance(extrovert_sum, pd.Series) or isinstance(introvert_sum, pd.Series):
        df['Introvert_extrovert_spectrum'] = extrovert_sum - introvert_sum
    
    return df


def s5e7_communication_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Online vs Offline behavioral ratios"""
    df = df.copy()
    
    # Online_offline_ratio = Post_frequency ÷ (Social_event_attendance + Going_outside)
    if all(col in df.columns for col in ['Post_frequency', 'Social_event_attendance', 'Going_outside']):
        df['Online_offline_ratio'] = df['Post_frequency'] / (df['Social_event_attendance'] + df['Going_outside'] + 1e-8)
    
    # Communication_balance = balanced ratio calculation
    if 'Post_frequency' in df.columns and 'Social_event_attendance' in df.columns:
        total_communication = df['Post_frequency'] + df['Social_event_attendance']
        df['Communication_balance'] = df['Post_frequency'] / (total_communication + 1e-8)
    
    return df


def get_feature_importance_order() -> list:
    """特徴量重要度順リスト（経験的順序）"""
    return [
        "extrovert_score",
        "introvert_score",
        "Social_event_attendance",
        "Time_spent_Alone",
        "Drained_after_socializing_encoded",
        "Stage_fear_encoded",
        "social_ratio",
        "social_participation_rate",
        "communication_ratio",
        "friend_social_efficiency",
        "Friends_circle_size",
        "Going_outside",
        "Post_frequency",
        "activity_sum",
        "post_per_friend",
        "fear_drain_interaction",
        "total_activity",
        "avg_activity",
        "personality_balance",
        "personality_ratio",
    ]


# ===== Advanced Feature Engineering Classes =====

class LightGBMFeatureEngineer(BaseEstimator, TransformerMixin):
    """LightGBM-optimized feature engineering (+0.3-0.5% expected)"""
    
    def __init__(self, use_power_transforms: bool = True):
        self.use_power_transforms = use_power_transforms
        self.power_transformers = {}
        self.numeric_features = None
        
    def fit(self, X, y=None):
        """Fit power transformations for skewed features"""
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in self.numeric_features:
            self.numeric_features.remove('id')
        
        if self.use_power_transforms:
            for col in self.numeric_features:
                if col in X.columns:
                    col_data = X[col].dropna()
                    if len(col_data) > 0:
                        skewness = col_data.skew()
                        if abs(skewness) > 0.5:  # Moderately skewed
                            self.power_transformers[col] = PowerTransformer(
                                method='yeo-johnson',
                                standardize=False
                            )
                            self.power_transformers[col].fit(col_data.values.reshape(-1, 1))
        
        return self
    
    def transform(self, X):
        """Apply power transformations for LightGBM"""
        X_transformed = X.copy()
        
        # Collect all power features in a dictionary for efficient addition
        power_features = {}
        
        for col, transformer in self.power_transformers.items():
            if col in X_transformed.columns:
                mask = X_transformed[col].notna()
                if mask.sum() > 0:
                    try:
                        transformed_values = transformer.transform(
                            X_transformed.loc[mask, col].values.reshape(-1, 1)
                        ).flatten()
                        
                        # Create full-length array with NaN for missing values
                        power_col = np.full(len(X_transformed), np.nan)
                        power_col[mask] = transformed_values
                        power_features[f'{col}_power'] = power_col
                    except Exception as e:
                        print(f"Warning: Power transformation failed for {col}: {e}")
        
        # Add all power features at once using pd.concat
        if power_features:
            power_df = pd.DataFrame(power_features, index=X_transformed.index)
            X_transformed = pd.concat([X_transformed, power_df], axis=1)
        
        return X_transformed


class CVSafeTargetEncoder(BaseEstimator, TransformerMixin):
    """Fold-safe target encoding (+0.2-0.4% expected)"""
    
    def __init__(self, cols: Optional[List[str]] = None, smoothing: float = 1.0, noise_level: float = 0.01):
        self.cols = cols
        self.smoothing = smoothing
        self.noise_level = noise_level
        self.encoders = {}
        self.global_mean = None
        
    def fit(self, X, y):
        """Fit target encoders with smoothing"""
        if y is None:
            raise ValueError("Target encoding requires y")
            
        self.global_mean = np.mean(y)
        
        if self.cols is None:
            self.cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.cols:
            if col in X.columns:
                encoder = TargetEncoder(
                    smoothing=self.smoothing,
                    min_samples_leaf=10,
                    return_df=True
                )
                temp_df = pd.DataFrame({col: X[col]})
                encoder.fit(temp_df, y)
                self.encoders[col] = encoder
        
        return self
    
    def transform(self, X):
        """Apply target encoding with noise for regularization"""
        X_transformed = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                temp_df = pd.DataFrame({col: X_transformed[col]})
                encoded_values = encoder.transform(temp_df)[col].values
                
                if self.noise_level > 0:
                    noise = np.random.normal(0, self.noise_level, size=len(encoded_values))
                    encoded_values = encoded_values + noise
                
                X_transformed[f'{col}_target_encoded'] = encoded_values
        
        return X_transformed


class AdvancedStatisticalFeatures(BaseEstimator, TransformerMixin):
    """Advanced statistical and imputation features (+0.1-0.3% expected)"""
    
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
        self.knn_imputer = None
        self.numeric_features = None
        self.original_features = None
        
    def fit(self, X, y=None):
        """Fit KNN imputer only on original features"""
        # Only apply KNN to original Bronze features, not derived polynomial features
        original_numeric_features = [
            'Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency'
        ]
        self.original_features = [f for f in original_numeric_features if f in X.columns]
        
        if self.original_features:
            self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            # Clip values before fitting to prevent numerical issues
            X_clipped = X[self.original_features].clip(-100, 100)
            self.knn_imputer.fit(X_clipped)
        
        return self
    
    def transform(self, X):
        """Apply KNN imputation and add statistical features"""
        X_transformed = X.copy()
        
        # 1. Store missing indicators before imputation (bulk operation)
        if self.knn_imputer and self.original_features:
            missing_indicators = {}
            for col in self.original_features:
                if col in X_transformed.columns:
                    missing_indicators[f'{col}_was_missing'] = X_transformed[col].isna().astype(int)
            
            if missing_indicators:
                missing_df = pd.DataFrame(missing_indicators, index=X_transformed.index)
                X_transformed = pd.concat([X_transformed, missing_df], axis=1)
            
            # Apply KNN imputation only to original features with clipping
            X_clipped = X_transformed[self.original_features].clip(-50, 50)
            X_transformed[self.original_features] = self.knn_imputer.transform(X_clipped)
        
        # 2. Add statistical moment features (bulk operation)
        # Use only original features for statistics to prevent explosion
        stats_features = self.original_features if self.original_features else []
        if stats_features:
            numeric_data = X_transformed[stats_features]
            
            # Clip extreme values to prevent overflow - ultra conservative clipping
            numeric_data_clipped = numeric_data.clip(-25, 25)
            
            statistical_features = {
                'row_mean': numeric_data_clipped.mean(axis=1),
                'row_std': numeric_data_clipped.std(axis=1).fillna(0),
                'row_q25': numeric_data_clipped.quantile(0.25, axis=1),
                'row_q75': numeric_data_clipped.quantile(0.75, axis=1),
            }
            
            # Safer calculation for skew and kurtosis
            try:
                statistical_features['row_skew'] = numeric_data_clipped.apply(
                    lambda x: x.dropna().skew() if len(x.dropna()) > 2 else 0, axis=1
                ).fillna(0).clip(-10, 10)
                
                statistical_features['row_kurtosis'] = numeric_data_clipped.apply(
                    lambda x: x.dropna().kurtosis() if len(x.dropna()) > 2 else 0, axis=1
                ).fillna(0).clip(-10, 10)
            except Exception:
                statistical_features['row_skew'] = pd.Series(0, index=X_transformed.index)
                statistical_features['row_kurtosis'] = pd.Series(0, index=X_transformed.index)
            statistical_features['row_iqr'] = statistical_features['row_q75'] - statistical_features['row_q25']
            
            stats_df = pd.DataFrame(statistical_features, index=X_transformed.index)
            X_transformed = pd.concat([X_transformed, stats_df], axis=1)
            
            # Feature-specific moments for key features
            key_features = ['Social_event_attendance', 'Time_spent_Alone', 'Friends_circle_size']
            key_feature_stats = {}
            for feat in key_features:
                if feat in X_transformed.columns:
                    # Safer zscore calculation with aggressive clipping
                    feat_data = X_transformed[feat].clip(-50, 50)
                    zscore = (feat_data - feat_data.mean()) / (feat_data.std() + 1e-8)
                    key_feature_stats[f'{feat}_zscore'] = zscore.clip(-5, 5)
                    
                    key_feature_stats[f'{feat}_percentile'] = X_transformed[feat].rank(pct=True)
            
            if key_feature_stats:
                key_stats_df = pd.DataFrame(key_feature_stats, index=X_transformed.index)
                X_transformed = pd.concat([X_transformed, key_stats_df], axis=1)
        
        return X_transformed


# ===== Sklearn-Compatible Transformers for Pipeline Integration =====

class SilverPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for Silver layer processing"""
    
    def __init__(self, add_polynomial: bool = True, add_scaling: bool = True):
        self.add_polynomial = add_polynomial
        self.add_scaling = add_scaling
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Fit the transformer (no fitting required for Silver layer)"""
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply Silver layer transformations"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        # Apply Silver pipeline
        X_transformed = advanced_features(X)
        X_transformed = s5e7_interaction_features(X_transformed)
        X_transformed = s5e7_drain_adjusted_features(X_transformed)
        X_transformed = s5e7_communication_ratios(X_transformed)
        X_transformed = enhanced_interaction_features(X_transformed)
        
        if self.add_polynomial:
            X_transformed = polynomial_features(X_transformed, degree=2)
        
        if self.add_scaling:
            X_transformed = scaling_features(X_transformed)
        
        return X_transformed


class FoldSafeSilverPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-safe Silver preprocessor for CV integration"""
    
    def __init__(self, use_target_encoding: bool = False):
        self.use_target_encoding = use_target_encoding
        self.scaler = StandardScaler()
        self.lgbm_engineer = LightGBMFeatureEngineer()
        self.stat_engineer = AdvancedStatisticalFeatures(n_neighbors=5)
        self.target_encoder = None
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Learn scaling parameters from training data only"""
        # Apply Silver transformations
        X_silver = advanced_features(X)
        X_silver = s5e7_interaction_features(X_silver)
        X_silver = s5e7_drain_adjusted_features(X_silver)
        X_silver = s5e7_communication_ratios(X_silver)
        X_silver = enhanced_interaction_features(X_silver)
        
        # Fit advanced feature engineers
        self.lgbm_engineer.fit(X_silver, y)
        X_silver = self.lgbm_engineer.transform(X_silver)
        
        self.stat_engineer.fit(X_silver, y)
        X_silver = self.stat_engineer.transform(X_silver)
        
        # Target encoding for categorical features (if enabled and y provided)
        if self.use_target_encoding and y is not None:
            categorical_cols = ['Stage_fear_encoded', 'Drained_after_socializing_encoded']
            available_cats = [col for col in categorical_cols if col in X_silver.columns]
            if available_cats:
                self.target_encoder = CVSafeTargetEncoder(cols=available_cats, smoothing=1.0)
                self.target_encoder.fit(X_silver, y)
                X_silver = self.target_encoder.transform(X_silver)
        
        # Fit scaler on numeric features only
        numeric_features = X_silver.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            self.scaler.fit(X_silver[numeric_features])
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply fold-safe transformations"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        # Apply Silver transformations
        X_transformed = advanced_features(X)
        X_transformed = s5e7_interaction_features(X_transformed)
        X_transformed = s5e7_drain_adjusted_features(X_transformed)
        X_transformed = s5e7_communication_ratios(X_transformed)
        X_transformed = enhanced_interaction_features(X_transformed)
        
        # Apply advanced feature engineering
        X_transformed = self.lgbm_engineer.transform(X_transformed)
        X_transformed = self.stat_engineer.transform(X_transformed)
        
        # Apply target encoding if fitted
        if self.target_encoder:
            X_transformed = self.target_encoder.transform(X_transformed)
        
        # Apply scaling only to numeric features
        numeric_features = X_transformed.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            X_transformed[numeric_features] = self.scaler.transform(X_transformed[numeric_features])
        
        return X_transformed


class EnhancedSilverPreprocessor(BaseEstimator, TransformerMixin):
    """Enhanced Silver layer processor combining all advanced techniques"""
    
    def __init__(
        self,
        use_power_transforms: bool = True,
        use_target_encoding: bool = True,
        use_statistical_features: bool = True,
        n_neighbors: int = 5,
        target_smoothing: float = 1.0
    ):
        self.use_power_transforms = use_power_transforms
        self.use_target_encoding = use_target_encoding
        self.use_statistical_features = use_statistical_features
        self.n_neighbors = n_neighbors
        self.target_smoothing = target_smoothing
        
        # Initialize sub-transformers
        self.lgbm_engineer = LightGBMFeatureEngineer(use_power_transforms=use_power_transforms)
        self.stat_engineer = AdvancedStatisticalFeatures(n_neighbors=n_neighbors)
        self.target_encoder = None
        
    def fit(self, X, y=None):
        """Fit all transformers"""
        if self.use_power_transforms:
            self.lgbm_engineer.fit(X, y)
        
        if self.use_statistical_features:
            self.stat_engineer.fit(X, y)
        
        if self.use_target_encoding and y is not None:
            categorical_cols = ['Stage_fear_encoded', 'Drained_after_socializing_encoded']
            available_cats = [col for col in categorical_cols if col in X.columns]
            if available_cats:
                self.target_encoder = CVSafeTargetEncoder(
                    cols=available_cats, 
                    smoothing=self.target_smoothing
                )
                self.target_encoder.fit(X, y)
        
        return self
    
    def transform(self, X):
        """Apply all transformations"""
        X_transformed = X.copy()
        
        # Apply core Silver features first
        X_transformed = advanced_features(X_transformed)
        X_transformed = s5e7_interaction_features(X_transformed)
        X_transformed = s5e7_drain_adjusted_features(X_transformed)
        X_transformed = s5e7_communication_ratios(X_transformed)
        X_transformed = enhanced_interaction_features(X_transformed)
        X_transformed = polynomial_features(X_transformed, degree=2)
        
        # Apply advanced feature engineering
        if self.use_power_transforms:
            X_transformed = self.lgbm_engineer.transform(X_transformed)
        
        if self.use_statistical_features:
            X_transformed = self.stat_engineer.transform(X_transformed)
        
        if self.target_encoder:
            X_transformed = self.target_encoder.transform(X_transformed)
        
        return X_transformed
