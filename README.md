# 22 ML Projects

> **Status:** 🚧 In Progress — more projects coming as I work through the series.

A growing collection of machine learning projects built from scratch — covering classification, regression, and model comparison pipelines. Each project is a self-contained Jupyter notebook with data, EDA, model training, and evaluation.

## Completed Projects

### 1. Titanic Survival Prediction 🛳️
Classic binary classification problem: predict passenger survival using logistic regression, decision trees, KNN, random forest, SVC, and a voting classifier. Includes feature engineering from `Cabin`, `Ticket`, and `Name` columns. Best accuracy ~83.6%.

**Stack:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

### 2. Housing Price Prediction 🏡
Predict California district median house value from census-block features. Uses stratified train/test splits, feature engineering (`rooms_per_house`, `bedrooms_ratio`, `people_per_house`), `ColumnTransformer` preprocessing, and compares Decision Tree, Random Forest (tuned via `GridSearchCV`), and SGDRegressor. Saved pipeline artifact.

**Stack:** Pandas, NumPy, Matplotlib, Scikit-learn  
**Dataset:** California Housing

---

### 3. Iris Flower Classification 🌸
Classify iris flowers into three species (*Setosa*, *Versicolor*, *Virginica*) from sepal/petal measurements. Trains and compares four classifiers: Decision Tree, KNN, SVM, and Logistic Regression with full EDA, pairplots, correlation heatmaps, and decision boundary visualization.

**Stack:** Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn  
**Dataset:** Fisher's Iris

---

### 4. SONAR Rock vs. Mine 💣
Binary classification to distinguish rocks from mines using SONAR frequency-energy readings. Uses Logistic Regression with train/test split and accuracy evaluation.

**Stack:** Pandas, NumPy, Scikit-learn  
**Dataset:** Sonar Mines/Rocks (60-band frequency data)

---

### 5. Build Your Own Linear Regression 〰️
A from-scratch implementation of simple linear regression using the closed-form OLS normal equations — no `LinearRegression()` from scikit-learn. Implemented as a `MyLR` class with `.fit()` and `.predict()` methods, validated against scikit-learn's output. Uses CGPA → placement salary data.

**Stack:** NumPy, Pandas, Matplotlib, Scikit-learn (for splitting only)  
**Dataset:** Placement (CGPA vs. package)

---

### 6. Heart Disease Prediction 🫀

Binary classification problem: predict presence of heart disease from clinical attributes (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, max heart rate, exercise-induced angina, and more). Suggested Improvements: Include EDA, correlation analysis, and comparison across multiple classifiers (e.g. Logistic Regression, KNN, Random Forest, SVM).

**Stack:** NumPy, Pandas, Scikit-learn

**Dataset:** UCI Heart Disease Prediction (Cleveland)

---

### 7. Credit Card Fraud Detection 💳

Binary classification problem: detect fraudulent transactions from anonymized transaction data (features V1–V28, Time, and Amount). Built a Logistic Regression model and evaluated performance using scikit-learn's accuracy_score.

**Stack:** NumPy, Pandas, Scikit-learn 

**Dataset:** Credit Card Fraud Detection (Kaggle - ULB)

---

## Repository Structure

```
22 ML Projects/
├── README.md                          ← this file
├── build-your-own-linear-regression/  # OLS from scratch
├── Housing Price Prediction/          # California housing
├── Iris Flower Classification/        # Species classification
├── SONAR-rock-or-mine/                # Rock vs mine detection
└── Titanic Survival Prediction/       # Passenger survival
```


