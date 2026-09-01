# Heart Disease Prediction

Predict whether a patient has heart disease from 13 clinical attributes, using logistic regression
on the Cleveland subset of the UCI Heart Disease dataset.

## Objective

Given age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar,
resting ECG results, maximum heart rate, exercise-induced angina, ST depression, and a few related
measurements, classify each patient as having heart disease (`target = 1`) or not (`target = 0`).

## Dataset

- **File:** `heart_disease_data.csv`
- **Rows:** 303 patients
- **Features:** 13 clinical attributes
- **Target:** `target` — 1 = defective heart, 0 = healthy heart
- No missing values

## Workflow

1. Load the data and inspect shape, dtypes, and summary statistics
2. Confirm there are no nulls and check the class balance
3. Split features from the target, then a stratified 80/20 train/test split
4. Fit logistic regression
5. Score on both the training and test halves
6. Run a single held-out patient through the model end to end

## Results

| Split | Accuracy |
|-------|----------|
| Train | 0.847 |
| Test  | 0.803 |

## Caveats

303 rows is small, and a single 80/20 split leaves a 61-patient test set — one prediction moves
accuracy by more than a point and a half. Treat the number above as a rough indication rather than a
reliable estimate.

## Next steps

- Cross-validate (`StratifiedKFold`) and report mean ± std instead of a single split
- Scale the features — logistic regression converges poorly on the raw scales here
- Report precision, recall, and ROC-AUC: in a clinical screening context a false negative is far
  more costly than a false positive, and accuracy hides that entirely
- Add EDA — correlation analysis and per-feature distributions by class
- Compare against KNN, random forest, and SVM

## Files

- `HDiseasePrediction.ipynb` — the notebook
- `heart_disease_data.csv` — the dataset
