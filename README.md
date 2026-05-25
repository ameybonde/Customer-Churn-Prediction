# 📉 Customer Churn Prediction

> A binary classification system that predicts whether a telecom customer will churn — built with a tuned Random Forest pipeline and served via an interactive Streamlit web app.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**GitHub Repo:** [https://github.com/ameybonde/Customer-Churn-Prediction]

---

## 📌 Overview

This project predicts customer churn for a telecom company using structured customer data — contract type, tenure, charges, internet service, and more. The model is a **Random Forest Classifier** tuned with `RandomizedSearchCV` and optimized for **Recall**, since missing a churner (false negative) is far more costly than a false alarm. A complete preprocessing pipeline handles encoding and scaling, and the final model is deployed via a Streamlit app where users can input customer details and get a live churn probability.

---

## ✨ Features

- 🎯 **Recall-Optimized Model** — Tuned to catch as many actual churners as possible, not just maximize accuracy.
- 🔧 **Full Preprocessing Pipeline** — Handles categorical encoding and feature scaling in a single `ColumnTransformer`, serialized separately from the model.
- 🌲 **Tuned Random Forest** — Hyperparameters selected via 5-fold cross-validated `RandomizedSearchCV` over 25 iterations.
- ⚖️ **Class Imbalance Handling** — `class_weight='balanced'` used to compensate for the minority churn class.
- 📊 **Churn Probability Output** — Returns not just a binary prediction but a calibrated probability score.
- 🖥️ **Interactive Streamlit App** — 19 input fields covering all customer attributes, with real-time prediction on click.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.8+ |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | Scikit-learn `ColumnTransformer`, `OneHotEncoder`, `StandardScaler` |
| Model | Scikit-learn `RandomForestClassifier` |
| Tuning | Scikit-learn `RandomizedSearchCV` |
| Model Persistence | Pickle |
| Frontend | Streamlit |
| Dataset | IBM Sample Data Sets (Kaggle) |

---

## 📂 Project Structure

```
customer-churn-prediction/
│
├── app.py                          # Streamlit app — entry point
├── Customer_Churn_Pred.ipynb       # Full ML pipeline notebook
│
├── model.pkl                       # Serialized trained Random Forest model
├── preprocessor.pkl                # Serialized ColumnTransformer pipeline
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Kaggle Dataset Link: [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The IBM Sample Data Sets (Kaggle) Telco Customer Churn dataset contains records for **7,043 customers** with **21 columns** covering demographics, account information, and subscribed services. The target variable is `Churn` — whether the customer left within the last month.

| Feature Category | Columns |
|---|---|
| Demographics | Gender, SeniorCitizen, Partner, Dependents |
| Account Info | Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

---

## 🔬 Workflow & Methodology

### 1. Data Loading
Dataset is loaded from CSV into a Pandas DataFrame. `customerID` is dropped immediately — it carries no predictive signal.

### 2. Data Cleaning
`TotalCharges` is stored as a string type in the raw data due to whitespace entries. It is coerced to numeric, and the ~11 affected rows with missing values are dropped.

### 3. Target Encoding
The `Churn` column (`Yes`/`No`) is label-encoded to binary `1`/`0`.

### 4. EDA
Churn distribution, correlation heatmaps, and feature-wise churn rate plots are generated using Matplotlib and Seaborn to understand which features drive churn.

### 5. Preprocessing Pipeline
A `ColumnTransformer` is built with two branches:
- **Categorical columns** → `OneHotEncoder` (drop first to avoid multicollinearity)
- **Numerical columns** (`tenure`, `MonthlyCharges`, `TotalCharges`) → `StandardScaler`

The pipeline is fit on training data only to prevent data leakage, then applied to both train and test sets.

### 6. Train/Test Split
Data is split 80/20 with `stratify=y` to preserve the churn ratio in both sets (~26% churn).

### 7. Baseline Model
A default `RandomForestClassifier` is trained first to establish a performance baseline before tuning.

### 8. Hyperparameter Tuning with RandomizedSearchCV
`RandomizedSearchCV` is run with:
- **25 iterations**, **5-fold cross-validation**
- **Scoring metric: Recall** — because minimizing false negatives (missed churners) is the business priority
- **Search space:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`

Best parameters found:
```
n_estimators=500, max_depth=5, min_samples_split=5,
min_samples_leaf=8, max_features='sqrt', class_weight='balanced'
```

### 9. Evaluation
The final model is evaluated on both train and test sets across five metrics:

| Metric | Train | Test |
|---|---|---|
| Accuracy | 75.66% | 72.85% |
| Precision | 52.72% | 49.34% |
| Recall | **81.54%** | **79.95%** |
| F1 Score | 64.04% | 61.02% |
| ROC-AUC | 77.54% | 75.11% |

The model deliberately trades precision for recall — a deliberate design choice for a churn use case where missing a churner costs more than a false alarm.

### 10. Serialization with Pickle
The best model and the fitted preprocessor are saved separately as `.pkl` files. Keeping them separate is intentional — the `app.py` loads both independently and applies the preprocessor before passing data to the model.

---

## 🖥️ How It Works

When a user fills in the 19 customer attribute fields and clicks **Predict**, the app constructs a single-row DataFrame matching the original training schema, passes it through the loaded `preprocessor`, feeds the result into the loaded `model`, and displays both the binary prediction (`Churn` / `No Churn`) and the raw churn probability score.

---

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.8+
- pip

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Add Dataset File
Download the dataset from the Kaggle link above and place the CSV in the project root or update the file path in the notebook.

### Step 4 — Generate Pickle Artifacts
Run `Customer_Churn_Pred.ipynb` from top to bottom. This creates `model.pkl` and `preprocessor.pkl` in the project root.

### Step 5 — Launch the App
```bash
streamlit run app.py
```

---

## 📸 Screenshots

> Add a screenshot or demo GIF of the running app here.

```
<img width="727" height="1605" alt="Screenshot 2026-05-25 164602" src="https://github.com/user-attachments/assets/ce1c329b-3c97-44a1-89f3-6bd9e59498ff" />
<img width="690" height="1613" alt="Screenshot 2026-05-25 164445" src="https://github.com/user-attachments/assets/87aca42c-91d8-4a0b-808d-10b11fddf126" />


```

---

## 🚀 Future Improvements

- [ ] **SMOTE / Oversampling** — Handle class imbalance at the data level rather than relying solely on `class_weight`.
- [ ] **XGBoost / LightGBM** — Compare gradient boosting approaches against the current Random Forest baseline.
- [ ] **Feature Importance Plot** — Visualize which features drive churn most, directly in the Streamlit app.
- [ ] **Threshold Tuning** — Adjust the classification threshold beyond the default 0.5 to further optimize recall vs. precision tradeoff.
- [ ] **Calibrated Probabilities** — Apply `CalibratedClassifierCV` to make the probability scores more reliable.
- [ ] **Cloud Deployment** — Deploy on Streamlit Cloud or Hugging Face Spaces for public access.

---

## 👤 Author

**Amey Bonde**

- GitHub: [@ameybonde](https://github.com/ameybonde)
- LinkedIn: [Amey Bonde](https://www.linkedin.com/in/amey-bonde-72a8b23b2/)

---

> ⭐ If this helped you, consider starring the repository!
