# End-to-End-Credit-Risk-Analysis-EDA-PD-Model-XGBoost-


A full credit risk modeling project based on LendingClub data, covering data preprocessing, exploratory data analysis (EDA), PD modeling, machine learning, and scorecard development.

---

## 1. Overview

This project builds a complete credit risk modeling pipeline using LendingClub loan data.

It includes:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Probability of Default (PD) modeling (Logistic Regression)
* Machine Learning model (XGBoost)
* Model evaluation (AUC and KS)
* Credit scorecard generation
* Risk-based decision strategy

The project follows a typical bank-level credit risk modeling workflow.

---

## 2. Methodology

### 2.1 Data Extraction and Cleaning

Files: `src/extraction.py`, `src/clean.py`

* Select relevant variables
* Handle missing values
* Define target variable (`default`)
* Save cleaned dataset

---

### 2.2 Exploratory Data Analysis (EDA)

File: `notebooks/eda.ipynb`

* Based on cleaned dataset (`lendingclub_clean.parquet`)
* Default rate: 19.77%
* Missing value analysis
* Distribution analysis (income, DTI, interest rate)
* Default vs non-default comparison

Key findings:

* Default borrowers have higher DTI

* Default borrowers have higher interest rates

* Default borrowers have lower income

* Higher risk groups:

  * RENT
  * SMALL BUSINESS
  * 60 months term

* No strong multicollinearity (max correlation ≈ 0.3)

---

### 2.3 Feature Engineering

File: `src/feature_engineering.py`

* Train-test split
* Numerical scaling (StandardScaler)
* Categorical encoding
* Model-ready dataset construction

---

### 2.4 PD Model (Logistic Regression)

File: `src/pd_model.py`

* Baseline interpretable model
* Outputs:

  * Probability of Default (PD)
  * AUC
  * KS statistic

---

### 2.5 Machine Learning Model

File: `src/ml_model.py`

* XGBoost classifier
* Captures nonlinear relationships

---

### 2.6 Model Evaluation
```bash
Logistic AUC: 0.7069
XGBoost AUC: 0.7202

Logistic KS: 0.3015
XGBoost KS: 0.3209

* Both models show solid predictive performance
* XGBoost slightly outperforms Logistic Regression
```
---

### 2.7 Scorecard Development

File: `src/scorecard.py`

* Convert PD to score
* Assign rating (A–E)

Example:

| PD   | Score | Rating |
| ---- | ----- | ------ |
| 0.28 | 667   | B      |
| 0.08 | 774   | A      |
| 0.43 | 618   | C      |

---

### 2.8 Credit Decision Strategy

* Approve / Review / Reject based on score
* Evaluate real default rate by decision group

---

## 3. Project Structure
```bash
CREDIT-RISK-ANALYSIS
|   requirements.txt
|
+---data
|       appl_accepted_20072019Q3.csv
|
+---notebooks
|       eda.ipynb
|       main.ipynb
|
+---outputs
|       lendingclub_clean.parquet
|       scorecard.parquet
|       X_test.parquet
|       X_train.parquet
|       y_test.parquet
|       y_train.parquet
|
\---src
        clean.py
        extraction.py
        feature_engineering.py
        ml_model.py
        pd_model.py
        pipeline.py
        scorecard.py
```
---

## 4. Data Note

The dataset used in this project is from LendingClub.

Due to GitHub size limitations, the raw data is not included in this repository.

You can download the dataset from:
https://www.kaggle.com/datasets/denychaen/lending-club-loans-rejects-data

After downloading, place the file under:

credit_risk_analysis/data

---

## 5. How to Run
```bash
pip install -r requirements.txt

Run:

notebooks/main.ipynb 
notebooks/eda.ipynb
```
---

## 6. Key Features

* End-to-end credit risk modeling pipeline
* Logistic regression and XGBoost comparison
* Scorecard system for interpretability
* Risk-based decision framework

---

## 7. Future Improvements

* LGD and EAD modeling
* WOE / IV feature engineering
* Model explainability (SHAP)
* Time-based validation

---

## 8. Environment
```bash
* Python 3.12.10
type requirements.txt
```
