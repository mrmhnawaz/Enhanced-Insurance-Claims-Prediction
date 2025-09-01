# Enhanced Insurance Claims Prediction (Accuracy: 32%)

This project focuses on predicting insurance claims using machine learning techniques.  
It was developed in **Google Colab** as part of a data analytics and machine learning practice project.

---

## 📌 Project Overview
The goal of this project is to build and evaluate a predictive model for insurance claim prediction.  
We apply data preprocessing, exploratory data analysis (EDA), and machine learning algorithms to predict whether a claim will be made.

---

## ⚙️ Features
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Accuracy comparison across models
- Achieved **32% prediction accuracy**

---

## 🛠️ Tech Stack
- **Python**
- **Google Colab / Jupyter Notebook**
- **Libraries:**
  - NumPy
  - Pandas
  - Matplotlib / Seaborn
  - Scikit-learn

---

## 📊 Workflow
1. Import dataset  
2. Clean and preprocess data  
3. Perform EDA (visualizations, distributions, correlations)  
4. Apply feature engineering  
5. Train ML models (Logistic Regression, Decision Trees, etc.)  
6. Evaluate model performance  
7. Report accuracy and insights  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/mrmhnawaz/insurance-claims-prediction.git
   cd insurance-claims-prediction


Enhanced Insurance Claims Prediction — Project Walkthrough

> **Goal:** Predict the **ClaimStatus** (e.g., Approved / Pending / Denied) for health insurance claims using classical machine learning models.

This repository contains a Google Colab notebook: **`Ehanced_Claims_Prediction_with_32_accuracy_project_1.ipynb`**.  
It walks through **EDA → preprocessing → model training → evaluation** on a synthetic claims dataset.

---

## 📌 Project Objective

- Build a baseline ML pipeline to predict **`ClaimStatus`** from claim-, patient-, and provider-level attributes.
- Establish initial benchmarks with multiple models (Logistic Regression, Decision Tree, Random Forest, SVM).
- Document the pipeline **step by step** so anyone can reproduce and extend it.

You can install via:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```


# 1. Step-by-Step Walkthrough (with Code & Rationale)
from google.colab import files
uploaded = files.upload()

# 2. Load data
import pandas as pd

# Replace with your file name
df = pd.read_csv("enhanced_health_insurance_claims.csv")

# Peek at the data
df.head()

# 3. Basic checks (shape, info, missing values)
# Shape of dataset
print("Rows & Columns:", df.shape)

# Columns + dtypes
print(df.info())

# Missing values per column
print(df.isnull().sum())

# 4. EDA: Distributions & Counts
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Claim Amount Distribution
sns.histplot(df['ClaimAmount'], bins=30, kde=True)
plt.show()

# Example: Claim Status Count
sns.countplot(x='ClaimStatus', data=df)
plt.show()

# 5. Pre processing Imports

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# 6. Handle Missing Values (defensive)
# Numeric: fill NA with mean
for col in df.select_dtypes(include=['float64','int64']).columns:
    df[col] = df[col].fillna(df[col].mean())

# Categorical: fill NA with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 7. Encode Categorical Variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# 8. Train / Test Split
X = df.drop('ClaimStatus', axis=1)   # Features
y = df['ClaimStatus']                # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Baseline Model — Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_scaled, y_train)

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\\nClassification Report:\\n", classification_report(y_pred))



# 📈 Results So Far

Baseline model (Logistic Regression): 32% accuracy
Visualizations highlight imbalanced data distribution
Project demonstrates the workflow, but prediction accuracy is low

# 🔮 Future Improvements

Apply advanced models: Random Forest, XGBoost, LightGBM
Perform Hyperparameter Tuning for optimization
Handle Class Imbalance with SMOTE or Class Weights
Feature scaling & selection for improved performance
Increase dataset size for better generalization

👤 Author

Mohammed Hussain Nawaz
B.Tech in Electronics & Communication Engineering
Aspiring Data Analyst / Data Scientist
