"""
Data loading module for Resume Classification.
Loads raw datasets (CSV, XLSX, PDF directories) with validation and metadata checks.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    from src.config import RAW_CSV_PATH, RAW_XLSX_PATH
except ImportError:
    from config import RAW_CSV_PATH, RAW_XLSX_PATH


def load_raw_csv(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw Resume.csv dataset.
    
    Returns:
        pd.DataFrame containing columns: ['ID', 'Resume_str', 'Resume_html', 'Category']
    """
    path = csv_path or RAW_CSV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw CSV file not found at: {path}")
    
    df = pd.read_csv(path, encoding='utf-8')
    return df


def load_raw_xlsx(xlsx_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw Resume.xlsx dataset if available.
    """
    path = xlsx_path or RAW_XLSX_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw XLSX file not found at: {path}")
    
    df = pd.read_excel(path)
    return df


def inspect_dataset_sources() -> Dict[str, Any]:
    """
    Inspect both CSV and XLSX raw data sources to verify equivalence and integrity.
    
    Returns:
        Dict summarizing findings from both sources.
    """
    info = {}
    csv_df = load_raw_csv()
    info["csv_shape"] = csv_df.shape
    info["csv_columns"] = list(csv_df.columns)
    info["csv_num_classes"] = csv_df["Category"].nunique()
    info["csv_categories"] = sorted(csv_df["Category"].unique().tolist())
    
    if os.path.exists(RAW_XLSX_PATH):
        try:
            xlsx_df = load_raw_xlsx()
            info["xlsx_shape"] = xlsx_df.shape
            info["xlsx_columns"] = list(xlsx_df.columns)
            info["xlsx_num_classes"] = xlsx_df["Category"].nunique()
            info["sources_match"] = (csv_df.shape == xlsx_df.shape)
        except Exception as e:
            info["xlsx_error"] = str(e)
            info["sources_match"] = False
    else:
        info["xlsx_present"] = False
        info["sources_match"] = True
        
    return info


if __name__ == "__main__":
    summary = inspect_dataset_sources()
    print("Dataset Inspection Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
