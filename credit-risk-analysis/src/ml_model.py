import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def train_ml_model(X_train, X_test, y_train, y_test):
    """
    Train ML model (XGBoost) for PD prediction
    """

    # -------------------------------------------------
    #  Model
    # -------------------------------------------------
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # -------------------------------------------------
    #  Prediction
    # -------------------------------------------------
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------
    #  AUC
    # -------------------------------------------------
    auc_train = roc_auc_score(y_train, y_train_pred)
    auc_test = roc_auc_score(y_test, y_test_pred)

    print(f"[XGBoost] AUC Train: {auc_train:.4f}")
    print(f"[XGBoost] AUC Test: {auc_test:.4f}")

    # -------------------------------------------------
    #  ROC Curve
    # -------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_test_pred)

    plt.figure()
    plt.plot(fpr, tpr, label="XGBoost")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve (XGBoost)")
    plt.legend()
    plt.show()

    # -------------------------------------------------
    #  KS
    # -------------------------------------------------
    ks = np.max(tpr - fpr)
    print(f"[XGBoost] KS: {ks:.4f}")

    # -------------------------------------------------
    #  Feature Importance
    # -------------------------------------------------
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n[XGBoost] Top Features:")
    print(importance.head(10))

    metrics = {
        "auc_train": auc_train,
        "auc_test": auc_test,
        "ks": ks
    }

    return model, metrics