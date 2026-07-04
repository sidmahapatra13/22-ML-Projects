# Iris Flower Classification

A machine learning project that classifies iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on sepal and petal measurements.

## Objective

The Iris dataset contains 150 samples (50 per class), each described by four features: sepal length, sepal width, petal length, and petal width. The goal is to build a model that generalizes well to new, unseen flower measurements rather than just fitting the training data.

## Dataset

- **File:** `iris_data.csv`
- **Features:** `SepalLength(cm)`, `SepalWidth(cm)`, `PetalLength(cm)`, `PetalWidth(cm)`
- **Target:** `Species` (3 classes)
- No missing values in the dataset.

## Project Workflow

1. **Exploratory Data Analysis (EDA)**
   - Distribution histograms for each feature, split by species
   - 2D scatter plot (sepal length vs. sepal width)
   - Pairwise scatter plots across all features
   - Correlation heatmap

2. **Data Preprocessing**
   - Label encoding of the `Species` column
   - Feature selection (petal length & petal width used for modeling)
   - Train/test split (70/30)
   - Feature scaling with `StandardScaler`

3. **Model Development**
   Four classifiers are trained and evaluated:
   - Decision Tree Classifier (entropy criterion)
   - K-Nearest Neighbors (k=3)
   - Support Vector Machine (RBF kernel)
   - Logistic Regression

4. **Evaluation**
   - Accuracy score
   - RMSE
   - R² score
   - Confusion matrix (Decision Tree)
   - Decision boundary visualizations for training and test sets

## Requirements

```
numpy
pandas
seaborn
matplotlib
scikit-learn
```

## Usage

1. Place `iris_data.csv` in the same directory as the notebook.
2. Open and run `iris.ipynb` cell by cell in Jupyter.

## Notes

- This is part of a broader personal project working through a curated list of 22 ML projects, building hands-on classification and modeling skills from scratch.
- Feel free to extend this with additional models (e.g., Random Forest, Naive Bayes) or hyperparameter tuning for comparison.
