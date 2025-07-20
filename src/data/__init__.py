"""
Medallion Architecture Data Management
Bronze/Silver/Gold Data Pipeline
"""

from typing import Tuple

import pandas as pd

from .bronze import create_bronze_tables, load_bronze_data
from .bronze import load_data as bronze_load_data
from .bronze import quick_preprocess
from .gold import create_gold_tables, create_submission, get_feature_names, get_ml_ready_data, load_gold_data
from .silver import advanced_features, create_silver_tables, load_silver_data, scaling_features


def create_all_tables() -> None:
    """全てのmedallionテーブルを作成"""
    print("Creating medallion architecture tables...")
    create_bronze_tables()
    create_silver_tables()
    create_gold_tables()
    print("All medallion tables created successfully!")


def quick_start(level: str = "bronze") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """レベル別クイックスタート

    Args:
        level: "bronze", "silver", "gold" のいずれか

    Returns:
        (train_df, test_df): 各レベルのデータフレーム
    """
    if level == "bronze":
        return load_bronze_data()
    elif level == "silver":
        return load_silver_data()
    elif level == "gold":
        return load_gold_data()
    else:
        raise ValueError(f"Unsupported level: {level}. Choose from 'bronze', 'silver', 'gold'")


__all__ = [
    "create_all_tables",
    "quick_start",
    "create_bronze_tables",
    "create_silver_tables",
    "create_gold_tables",
    "load_bronze_data",
    "load_silver_data",
    "load_gold_data",
    "get_ml_ready_data",
    "create_submission",
    "get_feature_names",
    "bronze_load_data",
    "quick_preprocess",
    "advanced_features",
    "scaling_features",
]
