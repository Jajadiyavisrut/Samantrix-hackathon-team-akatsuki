"""
Module: src.data.split_data
Purpose: Strict, stratified, leak-free train/validation/test splitting for ResumeForge 2026.
Splits: 70% Train, 15% Validation, 15% Test.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(
    input_path: str = "data/processed/clean_resumes.csv",
    output_dir: str = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Performs stratified 70/15/15 train/val/test splitting on cleaned resume data.
    Enforces zero ID and zero text overlap across partitions.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    df = pd.read_csv(input_path)
    print(f"[Data Split] Input shape: {df.shape}")

    # First split: Separate Train (70%) from Temporary Holdout (30%)
    temp_ratio = val_ratio + test_ratio  # 0.30
    df_train, df_temp = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df["Category"],
    )

    # Second split: Divide Temporary Holdout (30%) equally into Val (15%) and Test (15%)
    relative_test_ratio = test_ratio / temp_ratio  # 0.50
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=df_temp["Category"],
    )

    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Verification: Check strict disjointness
    train_ids = set(df_train["ID"])
    val_ids = set(df_val["ID"])
    test_ids = set(df_test["ID"])

    assert len(train_ids.intersection(val_ids)) == 0, "Train-Val ID leakage detected!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train-Test ID leakage detected!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val-Test ID leakage detected!"

    train_texts = set(df_train["Resume_str"])
    val_texts = set(df_val["Resume_str"])
    test_texts = set(df_test["Resume_str"])

    assert len(train_texts.intersection(val_texts)) == 0, "Train-Val Text leakage detected!"
    assert len(train_texts.intersection(test_texts)) == 0, "Train-Test Text leakage detected!"
    assert len(val_texts.intersection(test_texts)) == 0, "Val-Test Text leakage detected!"

    print(f"[Data Split] Split sizes: Train={len(df_train)} ({len(df_train)/len(df):.1%}), "
          f"Val={len(df_val)} ({len(df_val)/len(df):.1%}), "
          f"Test={len(df_test)} ({len(df_test)/len(df):.1%})")

    # Persist
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"[Data Split] Successfully persisted splits to {output_dir}")

    return {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }


if __name__ == "__main__":
    split_dataset()
