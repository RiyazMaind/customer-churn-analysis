# Customer Churn Analysis

This repository contains a Jupyter Notebook (`Customer Churn Analysis.ipynb`) that performs an in-depth analysis of customer churn for a telecommunications company. The project aims to identify key factors contributing to churn and develop a predictive model to accurately forecast customer attrition.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Machine Learning Models](#machine-learning-models)
6. [Best Model Summary](#best-model-summary)
7. [How to Run the Notebook](#how-to-run-the-notebook)
8. [Technologies Used](#technologies-used)

---

## Project Overview

Customer churn is a critical concern for businesses, especially in competitive sectors like telecommunications. This project provides a comprehensive analysis of customer behavior to understand why customers leave and to predict future churners. The insights gained can help companies develop targeted retention strategies.

---

## Dataset

The analysis utilizes a Telco customer dataset, which includes various attributes such as:

- **Customer demographics**: gender, age, partner, dependents
- **Services subscribed**: phone, internet, online security, streaming, etc.
- **Account information**: tenure, contract type, monthly charges, total charges
- **Churn status**: binary (Yes/No)

---

## Exploratory Data Analysis (EDA)

Key findings from EDA:

- **Demographics**: Senior citizens churn at 41.6%, compared to 23.6% for non-senior citizens. Gender is not a strong predictor.
- **Relationship Status**: Customers without a partner (32.9%) or dependents (31%) are more likely to churn.
- **Services**:
  - Fiber Optic internet users churn at 41.8%.
  - Customers without online security (41.7%), backup (39.9%), or tech support (41.6%) churn more often.
- **Contract & Billing**:
  - Month-to-month contracts: churn rate 42.7%.
  - Electronic check payments: churn rate 45.2%.
- **Charges**: Lower total charges and higher monthly charges are linked to higher churn.
- **Tenure**: Newer customers churn more frequently.

---

## Data Preprocessing

Steps applied:

1. **Feature Dropping**: Removed `customerID`.
2. **Missing Values**: Filled missing `TotalCharges` with median.
3. **Categorical Encoding**:
   - Binary features mapped to 0/1.
   - `Contract` encoded numerically (0 = Month-to-month, 1 = One year, 2 = Two year).
   - Other categorical variables one-hot encoded.
4. **Feature Scaling**: Standardized numerical features with `StandardScaler`.

---

## Machine Learning Models

The following models were trained and evaluated:

- **Logistic Regression**

  - Accuracy: 0.807
  - ROC-AUC: 0.73
  - Recall (Churn): 0.57

- **Random Forest Classifier (Base)**

  - ROC-AUC: 0.826

- **XGBoost Classifier**

  - Accuracy: 0.791
  - ROC-AUC: 0.828

- **Support Vector Classifier (SVM)**

  - Accuracy: 0.790
  - ROC-AUC: 0.792
  - Recall (Churn): 0.49

- **Tuned Random Forest Classifier (Best Model)**
  - Parameters: `max_depth=10`, `min_samples_split=10`, `n_estimators=300`, `class_weight="balanced"`
  - Accuracy: 0.770
  - ROC-AUC: 0.843
  - Recall (Churn): 0.76

---

## Best Model Summary

The **Tuned Random Forest Classifier** is the best-performing model. Although its accuracy is slightly lower than Logistic Regression, it achieves the highest ROC-AUC (0.843) and a strong recall (0.76) for the churn class.  
In a business context, identifying churners correctly (high recall) is more valuable than maximizing accuracy.

---

## How to Run the Notebook

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/customer-churn-analysis.git
   cd customer-churn-analysis
   ```

````

2. **Install dependencies**:

   ```bash
   pip install pandas seaborn matplotlib scikit-learn xgboost imbalanced-learn
   ```

3. **Open the notebook**:

   ```bash
   jupyter notebook
   ```

   or

   ```bash
   jupyter lab
   ```

4. **Run cells**: Execute cells sequentially to reproduce the analysis and train models.

---

## Technologies Used

* **Python**
* **Pandas**: Data manipulation
* **NumPy**: Numerical computations
* **Matplotlib & Seaborn**: Data visualization
* **Scikit-learn**: ML algorithms and preprocessing
* **XGBoost**: Gradient boosting model
* **Imbalanced-learn (SMOTE)**: Class imbalance handling
````
