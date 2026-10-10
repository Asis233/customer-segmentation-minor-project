"""
Data Processing Module for Customer Segmentation Dashboard.
Handles loading datasets, column detection, data cleaning, and dataset summary statistics.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional


def load_dataset(file_or_path) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads a dataset from a CSV file path or file-like object.
    
    Returns:
        Tuple[Optional[pd.DataFrame], str]: (DataFrame or None, error_message or "")
    """
    try:
        if hasattr(file_or_path, 'name'):
            if not file_or_path.name.lower().endswith('.csv'):
                return None, "Invalid file format. Please upload a CSV file."
            df = pd.read_csv(file_or_path)
        else:
            df = pd.read_csv(file_or_path)
            
        if df.empty:
            return None, "The uploaded dataset is empty."
            
        return df, ""
    except Exception as e:
        return None, f"Failed to load dataset: {str(e)}"


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Automatically detects common customer segmentation columns in a DataFrame.
    
    Returns:
        Dict mapping standard role names to detected column names in df.
    """
    cols = df.columns.tolist()
    mapping = {
        "id": None,
        "income": None,
        "spending": None,
        "age": None,
        "gender": None
    }
    
    for c in cols:
        clow = c.lower().strip()
        
        # ID column
        if mapping["id"] is None and any(k in clow for k in ["customerid", "customer_id", "cust_id", "id"]):
            mapping["id"] = c
            
        # Annual Income column
        if mapping["income"] is None and any(k in clow for k in ["annual income", "income", "annual_income", "salary", "earnings"]):
            mapping["income"] = c
            
        # Spending Score column
        if mapping["spending"] is None and any(k in clow for k in ["spending score", "spending_score", "spending", "score"]):
            mapping["spending"] = c
            
        # Age column
        if mapping["age"] is None and any(k in clow for k in ["age", "customer_age"]):
            mapping["age"] = c
            
        # Gender column
        if mapping["gender"] is None and any(k in clow for k in ["gender", "sex"]):
            mapping["gender"] = c
            
    # Fallback for numerical features if exact names were not found
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if mapping["id"] in num_cols:
        num_cols.remove(mapping["id"])
        
    if mapping["income"] is None and len(num_cols) >= 1:
        mapping["income"] = num_cols[0]
    if mapping["spending"] is None and len(num_cols) >= 2:
        mapping["spending"] = num_cols[1]
    if mapping["age"] is None and len(num_cols) >= 3:
        mapping["age"] = num_cols[2]
        
    return mapping


def validate_and_clean_data(
    df: pd.DataFrame,
    selected_features: List[str],
    id_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Validates, cleans, and handles missing/invalid numeric values in selected features.
    
    Returns:
        Tuple[pd.DataFrame, Dict]: (Cleaned DataFrame, Processing Summary Dict)
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # Ensure ID column exists or generate default customer IDs
    if id_col and id_col in df_clean.columns:
        df_clean["__display_id__"] = df_clean[id_col].astype(str)
    else:
        df_clean["__display_id__"] = [f"CUST-{1001 + i}" for i in range(len(df_clean))]
        
    # Coerce numeric types for selected features
    for col in selected_features:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
    # Count missing values
    missing_counts = df_clean[selected_features].isna().sum().to_dict()
    total_missing_cells = sum(missing_counts.values())
    
    # Drop rows with NaN in selected features
    df_clean = df_clean.dropna(subset=selected_features)
    
    # Drop exact full duplicate rows or duplicate ID rows if present
    initial_dups = len(df_clean)
    if id_col and id_col in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=[id_col])
    else:
        df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_dups - len(df_clean)
    
    cleaned_rows = len(df_clean)
    rows_dropped = initial_rows - cleaned_rows
    
    summary = {
        "initial_rows": initial_rows,
        "cleaned_rows": cleaned_rows,
        "rows_dropped": rows_dropped,
        "total_missing_cells": total_missing_cells,
        "missing_per_column": missing_counts,
        "duplicates_removed": duplicates_removed
    }
    
    return df_clean, summary


def calculate_dataset_stats(df: pd.DataFrame, income_col: Optional[str], spending_col: Optional[str]) -> Dict:
    """
    Calculates summary metrics for the dataset.
    """
    stats = {
        "total_customers": len(df),
        "avg_income": None,
        "avg_spending": None,
        "income_min": None,
        "income_max": None,
        "spending_min": None,
        "spending_max": None
    }
    
    if income_col and income_col in df.columns:
        valid_inc = pd.to_numeric(df[income_col], errors='coerce').dropna()
        if not valid_inc.empty:
            stats["avg_income"] = float(valid_inc.mean())
            stats["income_min"] = float(valid_inc.min())
            stats["income_max"] = float(valid_inc.max())
            
    if spending_col and spending_col in df.columns:
        valid_spend = pd.to_numeric(df[spending_col], errors='coerce').dropna()
        if not valid_spend.empty:
            stats["avg_spending"] = float(valid_spend.mean())
            stats["spending_min"] = float(valid_spend.min())
            stats["spending_max"] = float(valid_spend.max())
            
    return stats
