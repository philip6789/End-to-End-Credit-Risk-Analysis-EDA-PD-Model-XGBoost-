import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

def train_pd_model(X_train, X_test, y_train, y_test):
    """
    Train Probability of Default (PD) model using Logistic Regression.

    Returns:
    model, metrics dictionary
    """
    # -------------------------------------------------
    #  Scaling (VERY IMPORTANT)
    # -------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------
    #  Train model
    # -------------------------------------------------
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_scaled, y_train)



    # -------------------------------------------------
    #  Predict probabilities
    # -------------------------------------------------
    y_train_pred = model.predict_proba(X_train_scaled)[:, 1]
    y_test_pred = model.predict_proba(X_test_scaled)[:, 1]

    # -------------------------------------------------
    #  AUC
    # -------------------------------------------------
    auc_train = roc_auc_score(y_train, y_train_pred)
    auc_test = roc_auc_score(y_test, y_test_pred)

    print(f"AUC Train: {auc_train:.4f}")
    print(f"AUC Test: {auc_test:.4f}")

    # -------------------------------------------------
    #  ROC Curve
    # -------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_test_pred)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    # -------------------------------------------------
    #  KS Curve
    # -------------------------------------------------
    ks_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_test_pred
    })

    ks_df = ks_df.sort_values(by="y_pred", ascending=False).reset_index(drop=True)

    # 累积计算
    ks_df["cum_good"] = (1 - ks_df["y_true"]).cumsum() / (1 - ks_df["y_true"]).sum()
    ks_df["cum_bad"] = ks_df["y_true"].cumsum() / ks_df["y_true"].sum()

    ks_df["ks"] = ks_df["cum_bad"] - ks_df["cum_good"]

    ks = ks_df["ks"].max()
    print(f"KS (from curve): {ks:.4f}")

    # ks curve
    plt.figure()
    plt.plot(ks_df["cum_bad"], label="Cumulative Bad")
    plt.plot(ks_df["cum_good"], label="Cumulative Good")
    plt.title("KS Curve")
    plt.legend()
    plt.show()

    # -------------------------------------------------
    #  Feature Importance (Logistic Coefficients)
    # -------------------------------------------------
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": model.coef_[0]
    })

    coef_df = coef_df.sort_values(by="coefficient", ascending=False)

    print("\nTop Positive Risk Factors:")
    print(coef_df.head(10))

    print("\nTop Negative Risk Factors:")
    print(coef_df.tail(10))


    # -------------------------------------------------
    #  Cutoff Analysis
    # -------------------------------------------------

    cutoffs = np.linspace(0.1, 0.9, 9)

    results = []

    for c in cutoffs:
        y_pred_label = (y_test_pred >= c).astype(int)

        approval_rate = (y_pred_label == 0).mean()

        # 被批准的人中违约率
        approved = y_test[y_pred_label == 0]
        if len(approved) > 0:
            default_rate = approved.mean()
        else:
            default_rate = np.nan

        results.append([c, approval_rate, default_rate])

    cutoff_df = pd.DataFrame(results, columns=[
        "cutoff", "approval_rate", "default_rate"
    ])

    print("\nCutoff Analysis:")
    print(cutoff_df)

    # -------------------------------------------------
    #  Return
    # -------------------------------------------------
    metrics = {
        "auc_train": auc_train,
        "auc_test": auc_test,
        "ks": ks
    }
    print(y_train.mean())
    return model, scaler, metrics
