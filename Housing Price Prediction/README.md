# Housing Price Prediction

Predict California district **median house value** from census-block features using a cleaned sklearn pipeline and model comparison.

## What’s inside
- Stratified train/test split by `median_income` category
- Feature engineering: `rooms_per_house`, `bedrooms_ratio`, `people_per_house`
- Numeric + categorical preprocessing with `ColumnTransformer`
- Candidate models:
  - Decision Tree
  - Random Forest (`GridSearchCV`)
  - Ridge-style regressor via `SGDRegressor` (l2)
- Final hold-out evaluation on the held-out test set
- Saved fitted pipeline artifact + metrics

## Repo structure
```
Housing Price Prediction/
├── data/
│   └── housing.csv
├── 01_notebook.ipynb
└── README.md
```

## Notebook cells
1. Quick look at data
2. Create stratified test set
3. Explore + visualize geographic and correlation patterns
4. Attribute combinations
5. Prepare data for ML
6. Train models + cross-validate
7. Tune Random Forest via `GridSearchCV`
8. Finalize best model and evaluate on test set
9. Small model comparison: Ridge vs Random Forest
10. Save artifact to `artifacts/` and predict on new input

## Dataset
- Source: California Housing dataset
- Target: `median_house_value`
- Key signals: `median_income`, geographic coordinates, `ocean_proximity`

## Notes
- This notebook is intended as a portfolio-ready reference.
- Final model selection uses test-set RMSE from cross-validated candidates.
