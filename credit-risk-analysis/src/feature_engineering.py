import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def feature_engineering(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Feature engineering pipeline for credit risk modeling.

    Steps:
    1. Numerical feature transformation
    2. Categorical encoding
    3. Time feature extraction
    4. Feature selection
    5. Train-test split

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from cleaning step
    test_size : float
        Proportion of test dataset
    random_state : int
        Random seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    df = df.copy()

    # -------------------------------------------------
    #  Numerical Transformations
    # -------------------------------------------------

    # log income (income usually right-skewed)
    if "annual_inc" in df.columns:
        df["log_annual_inc"] = np.log1p(df["annual_inc"])

    # normalize DTI (optional clipping for extreme values)
    if "dti" in df.columns:
        df["dti"] = df["dti"].clip(upper=60)

    # interest rate already numeric after cleaning

    # loan amount scaling (optional)
    if "loan_amnt" in df.columns:
        df["loan_amnt_log"] = np.log1p(df["loan_amnt"])

    # -------------------------------------------------
    #  Term conversion
    # -------------------------------------------------

    if "term" in df.columns:
        df["term"] = (
            df["term"]
            .astype(str)
            .str.extract(r"(\d+)")
        )
        df["term"] = pd.to_numeric(df["term"], errors="coerce")

    # -------------------------------------------------
    #  Time feature extraction
    # -------------------------------------------------

    if "issue_d" in df.columns:

        df["issue_year"] = df["issue_d"].dt.year
        df["issue_month"] = df["issue_d"].dt.month

    # -------------------------------------------------
    #  Categorical Encoding
    # -------------------------------------------------

    categorical_cols = [
        "home_ownership",
        "purpose"
    ]

    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)

    # -------------------------------------------------
    #  Feature Selection
    # -------------------------------------------------

    selected_features = [
        "log_annual_inc",
        "dti",
        "revol_util",
        "open_acc",
        "total_acc",
        "delinq_2yrs",
        "loan_amnt_log",
        "int_rate",
        "emp_length",
        "term",
        "issue_year",
        "issue_month"
    ]

    # add encoded categorical columns
    for col in df.columns:
        if col.startswith("home_ownership_") or col.startswith("purpose_"):
            selected_features.append(col)

    # keep only existing columns
    selected_features = [col for col in selected_features if col in df.columns]

    X = df[selected_features].copy()

    y = df["default"]

    # -------------------------------------------------
    #  Missing Value Handling
    # -------------------------------------------------

    num_cols = X.select_dtypes(include=["float64", "int64"]).columns

    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    X = X.fillna(0)

    # -------------------------------------------------
    #  Train Test Split
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test