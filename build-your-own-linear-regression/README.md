# Linear Regression from Scratch

A from-scratch implementation of **Simple Linear Regression** in Python, built without using `scikit-learn`'s modeling tools. The goal is to understand the mathematics behind linear regression — specifically the closed-form (Ordinary Least Squares) solution — by implementing it manually with NumPy.

## Objective

Most tutorials use `LinearRegression()` from scikit-learn as a black box. This project instead derives and implements the slope and intercept directly from the OLS normal equations, to build intuition for:

- How the least-squares line is actually calculated
- The relationship between covariance/variance and the regression slope
- How a `fit` / `predict` API (like scikit-learn's) is structured under the hood

## How It Works

For simple linear regression with one feature, the line `y = mx + b` that minimizes squared error has a closed-form solution:

```
m = Σ (xᵢ - x̄)(yᵢ - ȳ) / Σ (xᵢ - x̄)²
b = ȳ - m·x̄
```

This is implemented in a `MyLR` class with `fit()` and `predict()` methods, mirroring scikit-learn's API. No gradient descent is used — the solution is computed analytically in one step, since there's a single feature.

## Dataset

- **File:** `placement.csv`
- **Features:** CGPA (input) → Placement package/salary (target)
- A single numeric feature is used to predict a single continuous target, making it a good fit for simple (as opposed to multiple) linear regression.

## Workflow

1. Load `placement.csv` with pandas
2. Split into `X` (feature) and `y` (target)
3. Train/test split (80/20, `random_state=2`) using `sklearn.model_selection.train_test_split` (used only for splitting, not modeling)
4. Fit `MyLR` on the training data to learn `m` (slope) and `b` (intercept)
5. Predict on the full test set and evaluate with MAE, MSE, RMSE, and R²
6. Visualize the fitted line against the train/test scatter
7. Cross-check `m` and `b` against scikit-learn's `LinearRegression` fit on the same split

## Requirements

- numpy
- pandas
- matplotlib
- scikit-learn (used for `train_test_split` and as a sanity-check baseline)

## Evaluation

The notebook reports MAE, MSE, RMSE, and R² on the test set, and separately compares `MyLR`'s learned slope/intercept against scikit-learn's `LinearRegression` fit on the same data to confirm the closed-form implementation is correct. Fill in the actual numbers here once you've run it:

- MAE: —
- MSE: —
- RMSE: —
- R²: —
- Slope/intercept difference vs. scikit-learn: —

## Possible Extensions

- Extend to multiple linear regression (more than one feature) using matrix form: `β = (XᵀX)⁻¹Xᵀy`
- Add k-fold cross-validation instead of a single train/test split
- Add gradient descent as an alternative fitting method and compare convergence/accuracy against the closed-form solution

## Key Takeaway

Linear regression's "learning" here isn't iterative — it's a direct algebraic solution. This project is meant to build intuition for *why* the least-squares line has the form it does, before moving on to gradient-descent-based methods used in more complex models.
