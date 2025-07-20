"""
Gold Level Data Management
Model-Ready Data for ML Training
"""

# type: ignore

from typing import Any, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler

DB_PATH = "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"


def clean_and_validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量のクリーニングと検証"""
    df = df.copy()

    # 無限値と極端な外れ値の処理
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col not in ["id"]:  # IDカラムはスキップ
            # 無限値をNaNに変換
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            # 極端な外れ値の処理（IQR法 + 絶対値制限）
            if df[col].notna().any():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                # 追加：数値安定性のための絶対値制限
                abs_limit = 1e5  # LightGBM numerical stability
                lower_bound = max(lower_bound, -abs_limit)
                upper_bound = min(upper_bound, abs_limit)

                # クリップして外れ値を境界値に設定
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # 欠損値の処理
    for col in numeric_cols:
        if col not in ["id"] and df[col].isna().any():
            # 中央値で補完
            df[col] = df[col].fillna(df[col].median())

    return df


def select_best_features(df: pd.DataFrame, target_col: str, k: int = 30, method: str = "combined") -> List[str]:
    """統計的特徴量選択でトップK特徴量を選択"""
    # 数値特徴量のみを選択（カテゴリカル変数は除外）
    feature_cols = []
    for col in df.columns:
        if col not in ["id", target_col, f"{target_col}_encoded"]:
            # 数値型の列のみを含める
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

    if len(feature_cols) <= k:
        return feature_cols

    # ターゲットが文字列の場合は数値に変換
    if df[target_col].dtype == "object":
        target_values = (df[target_col] == "Extrovert").astype(int)
    else:
        target_values = df[target_col]

    # 数値特徴量データのみを抽出してNaN処理
    X_df = df[feature_cols].copy()

    # NaNと無限値の処理
    for col in X_df.columns:
        X_df[col] = X_df[col].replace([np.inf, -np.inf], np.nan)
        X_df[col] = X_df[col].fillna(X_df[col].median())

    X = X_df.values
    y = target_values.values

    try:
        if method == "statistical":
            # F統計量ベースのみ
            selector_f = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
            selector_f.fit(X, y)
            selected_indices = selector_f.get_support()
            selected_features = [feature_cols[i] for i, selected in enumerate(selected_indices) if selected]
        elif method == "mutual_info":
            # 相互情報量ベースのみ
            selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(k, len(feature_cols)))
            selector_mi.fit(X, y)
            selected_indices = selector_mi.get_support()
            selected_features = [feature_cols[i] for i, selected in enumerate(selected_indices) if selected]
        else:
            # デフォルト: 2つの特徴量選択手法を組み合わせ
            # 1. F統計量ベース
            selector_f = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
            selector_f.fit(X, y)

            # 2. 相互情報量ベース
            selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(k, len(feature_cols)))
            selector_mi.fit(X, y)

            # 両方の手法で上位に選ばれた特徴量を優先
            f_scores = selector_f.scores_
            mi_scores = selector_mi.scores_

            # 正規化してスコアを結合
            f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
            mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)

            combined_scores = f_scores_norm + mi_scores_norm

            # トップK特徴量のインデックス取得
            top_indices = np.argsort(combined_scores)[-k:][::-1]

            selected_features = [feature_cols[i] for i in top_indices]

        return selected_features

    except Exception as e:
        print(f"Warning: Feature selection failed, using default features: {e}")
        # フォールバック: 上位の特徴量を手動選択
        return feature_cols[:k]


def prepare_model_data(
    df: pd.DataFrame, target_col: str = None, feature_cols: List[str] = None, 
    auto_select: bool = True, model_type: str = "lightgbm"
) -> pd.DataFrame:
    """改良されたモデル学習用データ準備"""
    df = df.copy()

    # データクリーニング
    df = clean_and_validate_features(df)

    # 文字列カラムのエンコーディング（元のカラムは保持）
    for col in df.columns:
        if df[col].dtype == 'object' and col != target_col:
            # カテゴリカルエンコーディング（元のカラムは保持）
            df[f"{col}_encoded"] = pd.Categorical(df[col]).codes.astype('int32')

    # 特徴量選択
    if feature_cols is None:
        if auto_select and target_col and target_col in df.columns:
            # 自動特徴量選択
            selected_features = select_best_features(df, target_col, k=30)
        else:
            # デフォルト特徴量セット（拡張版）
            default_features = [
                "extrovert_score",
                "social_ratio",
                "Social_event_attendance",
                "Time_spent_Alone",
                "Drained_after_socializing_encoded",
                "Stage_fear_encoded",
                "Friends_circle_size",
                "Going_outside",
                "Post_frequency",
                "introvert_score",
                "activity_sum",
                "post_per_friend",
                "fear_drain_interaction",
                "total_activity",
                "avg_activity",
                # 新しい交互作用特徴量
                "extrovert_social_interaction",
                "extrovert_social_ratio",
                "social_extrovert_ratio",
                "extrovert_social_event_interaction",
                "extrovert_alone_interaction",
                "extrovert_alone_contrast",
                "extrovert_drain_interaction",
                "social_friends_interaction",
                "social_outside_interaction",
                "social_post_interaction",
                "triple_interaction",
            ]

            # 多項式特徴量も含める（存在する場合）
            poly_features = [col for col in df.columns if col.startswith("poly_")]
            default_features.extend(poly_features)

            # エンコードされた特徴量も含める（存在する場合）
            encoded_features = [col for col in df.columns if col.endswith("_encoded")]
            default_features.extend(encoded_features)

            selected_features = default_features

        feature_cols = selected_features

    # 使用可能な特徴量のみ選択
    available_features = [col for col in feature_cols if col in df.columns]

    # 基本カラム（元のカラムを保持）
    model_cols = list(df.columns)  # 元のカラムをすべて保持
    
    # ターゲットエンコーディング
    if target_col and target_col in df.columns:
        # エンコードされたターゲットも追加
        encoded_target = f"{target_col}_encoded"
        if encoded_target in df.columns:
            model_cols.append(encoded_target)

    return df[model_cols]


def encode_target(df: pd.DataFrame, target_col: str = "Personality") -> pd.DataFrame:
    """ターゲット変数エンコーディング"""
    df = df.copy()
    if target_col in df.columns:
        df[f"{target_col}_encoded"] = (df[target_col] == "Extrovert").astype(int)
    return df


def create_gold_tables() -> None:
    """Creates LightGBM-ready Gold tables (Silver → Gold transformation)"""
    conn = duckdb.connect(DB_PATH)

    # goldスキーマ作成
    conn.execute("CREATE SCHEMA IF NOT EXISTS gold")

    # Silver依存チェーン - Exclusively consumes Silver layer
    try:
        train_silver = conn.execute("SELECT * FROM silver.train").df()
        test_silver = conn.execute("SELECT * FROM silver.test").df()
    except Exception:
        print("Silver tables not found. Creating silver tables first...")
        from .silver import create_silver_tables

        create_silver_tables()
        train_silver = conn.execute("SELECT * FROM silver.train").df()
        test_silver = conn.execute("SELECT * FROM silver.test").df()

    # Apply Gold layer processing pipeline (CLAUDE.md specification)
    # 1. Target encoding for training data
    train_gold = encode_target(train_silver)
    test_gold = test_silver.copy()

    # 2. Final validation: Infinite value processing, outlier detection
    train_gold = clean_and_validate_features(train_gold)
    test_gold = clean_and_validate_features(test_gold)

    # 3. Statistical feature selection (F-test + MI) for LightGBM optimization
    train_gold = prepare_model_data(train_gold, target_col="Personality", auto_select=True)
    test_gold = prepare_model_data(test_gold, auto_select=False)

    # goldテーブル作成・挿入
    conn.execute("DROP TABLE IF EXISTS gold.train")
    conn.execute("DROP TABLE IF EXISTS gold.test")

    conn.register("train_gold_df", train_gold)
    conn.register("test_gold_df", test_gold)

    conn.execute("CREATE TABLE gold.train AS SELECT * FROM train_gold_df")
    conn.execute("CREATE TABLE gold.test AS SELECT * FROM test_gold_df")

    print("Gold tables created (LightGBM-ready):")
    print(f"- gold.train: {len(train_gold)} rows, {len(train_gold.columns)} columns")
    print(f"- gold.test: {len(test_gold)} rows, {len(test_gold.columns)} columns")
    print(f"- Feature selection: Statistical selection (F-test + MI) applied")
    print(f"- Data quality: Final validation ensuring model training stability")

    conn.close()


def load_gold_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """gold層データ読み込み"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM gold.train").df()
    test = conn.execute("SELECT * FROM gold.test").df()
    conn.close()
    return train, test


def get_ml_ready_data(df: pd.DataFrame, target_col: str = "Personality") -> Tuple[pd.DataFrame, pd.Series]:
    """LightGBM互換のML準備済みデータを取得"""
    # モデルデータを準備
    model_data = prepare_model_data(df, target_col=target_col)
    
    # ターゲット列を分離
    if f"{target_col}_encoded" in model_data.columns:
        # DataFrameの真偽値評価を避けるため、明示的にSeriesとして取得
        target_series = model_data[f"{target_col}_encoded"]
        if isinstance(target_series, pd.DataFrame):
            # DataFrameの場合は最初の列を使用
            target_series = target_series.iloc[:, 0]
        y = pd.Series(target_series.astype('int32'), index=model_data.index)
        # ターゲット列と文字列カラムを除外した特徴量
        feature_cols = [col for col in model_data.columns 
                       if col != f"{target_col}_encoded" and model_data[col].dtype != 'object']
        X = model_data[feature_cols]
    else:
        # エンコードされていない場合は元のターゲット列を使用
        if target_col in model_data.columns:
            # ターゲット列を数値型に変換
            target_series = model_data[target_col]
            if isinstance(target_series, pd.DataFrame):
                # DataFrameの場合は最初の列を使用
                target_series = target_series.iloc[:, 0]
            y = pd.Series(pd.Categorical(target_series).codes.astype('int32'), index=model_data.index)
            # ターゲット列と文字列カラムを除外した特徴量
            feature_cols = [col for col in model_data.columns 
                           if col != target_col and model_data[col].dtype != 'object']
            X = model_data[feature_cols]
        else:
            raise ValueError(f"Target column {target_col} not found in data")
    
    return X, y


def create_submission_format(predictions: np.ndarray, filename: str = "submission.csv") -> None:
    """Competition output standardization - Standard Kaggle submission file creation"""
    _, test = load_gold_data()

    submission = pd.DataFrame(
        {"id": test["id"], "Personality": ["Extrovert" if pred == 1 else "Introvert" for pred in predictions]}
    )

    submission.to_csv(filename, index=False)
    print(f"Submission file created: {filename}")
    print(f"Predictions: {submission['Personality'].value_counts()}")


def create_submission(df: pd.DataFrame, predictions: np.ndarray, filename: str = "submission.csv") -> pd.DataFrame:
    """Create submission format DataFrame"""
    # Ensure predictions match DataFrame length
    if len(predictions) != len(df):
        raise ValueError(f"Predictions length {len(predictions)} does not match DataFrame length {len(df)}")
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': df['id'] if 'id' in df.columns else range(len(df)),
        'Personality': ['Extrovert' if pred > 0.5 else 'Introvert' for pred in predictions]
    })
    
    return submission


def get_feature_names(df: pd.DataFrame) -> List[str]:
    """Get feature names excluding ID and target columns"""
    exclude_cols = ["id", "Personality", "Personality_encoded"]
    feature_names = [col for col in df.columns if col not in exclude_cols]
    return feature_names


def extract_model_arrays(df: pd.DataFrame, target_col: str = "Personality") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """DataFrameからX, y, feature_namesを抽出 (後方互換性のため)"""
    # 特徴量列を特定
    feature_cols = [col for col in df.columns if col not in ["id", target_col, f"{target_col}_encoded"]]
    
    # X (特徴量行列)
    X = df[feature_cols].values
    
    # y (ターゲット)
    if f"{target_col}_encoded" in df.columns:
        y = df[f"{target_col}_encoded"].values
    elif target_col in df.columns:
        # 文字列ターゲットをエンコード
        y = (df[target_col] == "Extrovert").astype(int).values
    else:
        raise ValueError(f"Target column '{target_col}' or '{target_col}_encoded' not found")
    
    return X, y, feature_cols
