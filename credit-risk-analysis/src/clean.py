import pandas as pd


def clean_lendingclub(df: pd.DataFrame) -> pd.DataFrame:


    df = df.copy()

    # -----------------------------
    #  Structure cleaning
    # -----------------------------
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    percent_cols = ["revol_util", "int_rate"]

    for col in percent_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
            )

    # -----------------------------
    #  Type conversion
    # -----------------------------

    numeric_cols = [
        "annual_inc", "dti", "revol_util",
        "open_acc", "total_acc", "delinq_2yrs",
        "loan_amnt", "int_rate"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"],format="%b-%Y",errors="coerce")

    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"]
            .astype(str)
            .str.extract(r"(\d+)")
        )
        df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")

    # Convert remaining object columns to category
    categorical_cols = ["home_ownership", "purpose", "term"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # -----------------------------
    #  Target construction
    # -----------------------------
    valid_status = ["Fully Paid", "Charged Off","Default"]

    df = df[df["loan_status"].isin(valid_status)]

    if "loan_status" in df.columns:
        df["default"] = df["loan_status"].isin(
            ["Charged Off", "Default"]
        ).astype(int)

    # -----------------------------
    #  Basic missing handling
    # -----------------------------

    # Drop rows without target
    if "default" in df.columns:
        df = df.dropna(subset=["default"])

    return df

"""
Basic cleaning for LendingClub credit risk modeling.

Responsibilities:
- Structure cleaning
- Type conversion
- Target construction
- Basic missing handling

NOT included:
- Outlier strategy
- Feature transformation
"""