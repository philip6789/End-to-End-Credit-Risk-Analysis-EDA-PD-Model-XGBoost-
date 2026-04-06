import numpy as np
import pandas as pd


def generate_scorecard(model, X, base_score=600, pdo=50):
    """
    Generate credit score from PD model
    """

    pd_pred = model.predict_proba(X)[:, 1]
    pd_pred = np.clip(pd_pred, 1e-6, 1 - 1e-6)

    B = pdo / np.log(2)
    A = base_score

    score = A - B * np.log(pd_pred / (1 - pd_pred))

    result = pd.DataFrame({
        "PD": pd_pred,
        "Score": score
    })

    return result


def assign_rating(score):
    if score >= 700:
        return "A"
    elif score >= 650:
        return "B"
    elif score >= 600:
        return "C"
    else:
        return "D"   # 一定要兜底

def credit_decision(score):
    if score >= 700:
        return "Approve"
    elif score >= 650:
        return "Review"
    else:
        return "Reject"
