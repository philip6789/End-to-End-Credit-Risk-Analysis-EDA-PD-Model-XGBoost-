import os
import sys
os.makedirs("outputs", exist_ok=True)
sys.path.append(os.path.abspath(".."))
from src.extraction import extract_selected_columns
from src.clean import clean_lendingclub
from src.feature_engineering import feature_engineering
from src.pd_model import train_pd_model
from src.ml_model import train_ml_model
from src.scorecard import generate_scorecard, assign_rating, credit_decision




# ----------------------------
# 1. File path
# ----------------------------
path = "data/appl_accepted_20072019Q3.csv"

# ----------------------------
# 2. Extraction
# ----------------------------
print("Starting extraction...")
df_raw = extract_selected_columns(path)
print("Extraction done.")

# ----------------------------
# 3️. Cleaning
# ----------------------------
print("Starting cleaning...")
df_clean = clean_lendingclub(df_raw)
print("Cleaning done.")

# ----------------------------
# 4️. Save cleaned dataset
# ----------------------------
clean_path = "outputs/lendingclub_clean.parquet"
df_clean.to_parquet(clean_path, index=False)
print(f"Clean dataset saved to {clean_path}")

# ----------------------------
# 5️. Feature Engineering
# ----------------------------
print("Starting feature engineering...")

X_train, X_test, y_train, y_test = feature_engineering(df_clean)

print("Feature engineering done.")

# ----------------------------
# 6️. Save modeling dataset
# ----------------------------
X_train.to_parquet("outputs/X_train.parquet", index=False)
X_test.to_parquet("outputs/X_test.parquet", index=False)

y_train.to_frame("default").to_parquet("outputs/y_train.parquet", index=False)
y_test.to_frame("default").to_parquet("outputs/y_test.parquet", index=False)

print("Model datasets saved.")



# ----------------------------
# 7️. Train PD model
# ----------------------------
print("Starting PD model training...")

model, scaler, metrics = train_pd_model(X_train, X_test, y_train, y_test)

print("PD model training done.")

# ----------------------------
# 8️. Train ML model
# ----------------------------
print("Starting ML model training...")

ml_model, ml_metrics = train_ml_model(X_train, X_test, y_train, y_test)

print("ML model training done.")


# ----------------------------
# 9️. Model Comparison
# ----------------------------
print("\n===== Model Comparison =====")

print(f"Logistic AUC: {metrics['auc_test']:.4f}")
print(f"XGBoost  AUC: {ml_metrics['auc_test']:.4f}")

print(f"Logistic KS: {metrics['ks']:.4f}")
print(f"XGBoost  KS: {ml_metrics['ks']:.4f}")


# ----------------------------
# 10. Generate Scorecard
# ----------------------------
print("Generating scorecard...")

X_test_scaled = scaler.transform(X_test)

score_df = generate_scorecard(model, X_test_scaled)

# 👉 加 rating
score_df["Rating"] = score_df["Score"].apply(assign_rating)

print(score_df.head())

score_df.to_parquet("outputs/scorecard.parquet", index=False)

print("Scorecard generated.")

# ----------------------------
# 11. Real Default Rate Calculation & Decision Making
# ----------------------------
score_df["Decision"] = score_df["Score"].apply(credit_decision)
print(score_df["Decision"].value_counts(normalize=True))

result = score_df.copy()
result["default"] = y_test.values

print(result.groupby("Decision")["default"].mean())