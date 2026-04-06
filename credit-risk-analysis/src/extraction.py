# extract_module.py

import pandas as pd
import difflib
from typing import List, Dict


def get_column_mapping(file_path: str, desired_cols: List[str]) -> Dict[str, str]:
    """
    Match desired column names to actual dataset columns using fuzzy matching.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV file
    desired_cols : List[str]
        Desired column names

    Returns
    -------
    Dict[str, str]
        Mapping from desired column -> actual column name (or None if not found)
    """

    cols = pd.read_csv(file_path, nrows=0).columns.tolist()

    mapping = {}
    for d in desired_cols:
        match = difflib.get_close_matches(d, cols, n=1)
        mapping[d] = match[0] if match else None

    return mapping


def extract_selected_columns(
    file_path: str,
    chunksize: int = 200000
) -> pd.DataFrame:


    """
    Extract selected columns from a large CSV file using chunk processing.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV file
    desired_cols : List[str]
        Columns to extract (fuzzy matched)
    chunksize : int
        Number of rows per chunk

    Returns
    -------
    pd.DataFrame
        Extracted dataset
    """
    desired_cols = [
        "loan_status", "annual_inc", "emp_length", "home_ownership",
        "dti", "revol_util", "open_acc", "total_acc",
        "delinq_2yrs", "loan_amnt", "term", "int_rate",
        "purpose", "issue_d", "id"
    ]
    mapping = get_column_mapping(file_path, desired_cols)

    # Show mapping result (optional but useful)
    print("\n=== Column Mapping ===")
    for k, v in mapping.items():
        print(f"{k}  -->  {v}")

    real_cols = [c for c in mapping.values() if c is not None]

    if len(real_cols) == 0:
        raise ValueError("No matching columns found.")

    chunks = pd.read_csv(
        file_path,
        usecols=real_cols,
        chunksize=chunksize,
        low_memory=False
    )

    df = pd.concat(chunks, ignore_index=True)

    print(f"\nExtraction finished. Shape: {df.shape}")

    return df