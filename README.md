# 22 ML Projects

A working-through of machine learning, one project at a time — classification, regression,
recommendation, and a few things rebuilt from scratch with nothing but NumPy so the maths
stops being a black box.

> **Status:** 11 of 22 done. Each project is a self-contained folder with its data, a notebook
> carrying the full EDA → training → evaluation path, and notes on what worked and what didn't.

---

## The projects

| # | Project | Task | Result | Stack |
|---|---------|------|--------|-------|
| 1 | [Titanic Survival Prediction](./Titanic%20Survival%20Prediction) 🛳️ | Binary classification | ~83.6% CV accuracy | scikit-learn |
| 2 | [Housing Price Prediction](./Housing%20Price%20Prediction) 🏡 | Regression | RMSE on stratified hold-out | scikit-learn |
| 3 | [Iris Flower Classification](./Iris%20Flower%20Classification) 🌸 | Multiclass classification | 0.978 (SVM, logistic regression) | scikit-learn |
| 4 | [SONAR Rock vs. Mine](./SONAR-rock-or-mine) 💣 | Binary classification | 0.857 test accuracy | scikit-learn |
| 5 | [Linear Regression from Scratch](./build-your-own-linear-regression) 〰️ | Regression | R² 0.781, matches scikit-learn | NumPy only |
| 6 | [Heart Disease Prediction](./Heart%20Disease%20Prediction) 🫀 | Binary classification | 0.803 test accuracy | scikit-learn |
| 7 | [Credit Card Fraud Detection](./Credit%20Card%20Fraud%20Detection) 💳 | Imbalanced classification | Precision/recall/PR-AUC | scikit-learn |
| 8 | [Neural Network from Scratch](./building-neural-net-from-scratch) 🧠 | MNIST digits | ~66% after 100 GD steps | NumPy only |
| 9 | [Dog vs Cat Classifier](./Simple%20Image%20Classifier) 🐶🐱 | Image classification | 81.7% best val accuracy | TensorFlow/Keras |
| 10 | [Stock Price Prediction](./Stock%20Market%20Prediction) 📈 | Time-series forecasting | Test RMSE 595.94 | PyTorch |
| 11 | [Movie Recommender System](./Movie%20Recommender%20System) 🎬 | Content-based recommendation | Streamlit app included | scikit-learn, NLTK |

Two notebooks (Titanic, Housing) are committed without cell outputs — clone and run them to see
the numbers. Everything else renders with its outputs on GitHub.

---

## Running any of this

Each project reads its data with a relative path, so run the notebook from inside its own folder:

```bash
git clone https://github.com/sidmahapatra13/22-ML-Projects.git
cd 22-ML-Projects

python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

cd "Iris Flower Classification" && jupyter lab
```

Three projects need extras: `Movie Recommender System` and `Stock Market Prediction` each ship a
`requirements.txt`, and `Simple Image Classifier` needs TensorFlow (it was written for Colab on a
T4 GPU).

### The Streamlit app

The movie recommender has a frontend:

```bash
cd "Movie Recommender System"
pip install -r requirements.txt
streamlit run app.py
```

![Movie Recommender Streamlit preview](./Movie%20Recommender%20System/assets/streamlit-preview.png)

### Tests

The recommender's logic is covered by pytest:

```bash
pip install pytest pandas scikit-learn nltk streamlit
pytest tests/
```

### Datasets

Most data is committed alongside its notebook. Two are too large and need a download first:

| Project | Dataset | Where |
|---|---|---|
| Credit Card Fraud Detection | `creditcard.csv` | [Kaggle — ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Neural Network from Scratch | `train.csv` | [Kaggle — Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data) |

`Simple Image Classifier` pulls `salader/dogsvscats` through the Kaggle API from inside the notebook.

---

## Notes on each project

### 1. Titanic Survival Prediction 🛳️
Predict passenger survival, with most of the work in feature engineering rather than modelling —
cabin letter and cabin count from `Cabin`, numeric-vs-lettered `Ticket`, and title (Mr., Master.,
Dr.) pulled out of `Name`. Preprocessing runs through a `ColumnTransformer` with median/mode
imputation, scaling, and one-hot encoding; four classifiers (logistic regression, KNN, random
forest, SVC) are compared by 5-fold stratified CV, then tuned with `GridSearchCV` and re-scored.

**Dataset:** Kaggle Titanic

---

### 2. Housing Price Prediction 🏡
Predict California district median house value from census-block features. Stratified split on
income category so the train and test sets share the same income distribution, ratio features
(`rooms_per_house`, `bedrooms_ratio`, `people_per_house`) added inside the pipeline rather than
before it, and a decision tree, a `GridSearchCV`-tuned random forest, and an SGD-based ridge
compared by cross-validated RMSE. The winner is picked on CV score and only then scored once on
the hold-out, and the fitted pipeline is saved to `artifacts/`.

**Dataset:** California Housing

---

### 3. Iris Flower Classification 🌸
Classify *Setosa*, *Versicolor*, and *Virginica*. Full EDA first — per-feature histograms split by
species, a pairplot, and a correlation heatmap — which shows the petal measurements separate the
classes almost completely, so the models train on petal length and width alone. Decision tree, KNN,
SVM, and logistic regression are compared, with decision boundaries plotted for the training and
test sets.

**Dataset:** Fisher's Iris

---

### 4. SONAR Rock vs. Mine 💣
Distinguish rocks from mines using 60 bands of SONAR frequency-energy readings, with logistic
regression. A small dataset — 208 samples — so the 21-row test set makes the accuracy figure
noisier than it looks.

**Dataset:** UCI Connectionist Bench (Sonar, Mines vs. Rocks)

---

### 5. Linear Regression from Scratch 〰️
Simple linear regression implemented from the closed-form OLS normal equations, no
`LinearRegression()` involved:

```
m = Σ (xᵢ - x̄)(yᵢ - ȳ) / Σ (xᵢ - x̄)²
b = ȳ - m·x̄
```

Wrapped in a `MyLR` class with the usual `.fit()` / `.predict()` API, then checked against
scikit-learn — slope and intercept agree to floating-point noise.

**Dataset:** Placement (CGPA → package)

---

### 6. Heart Disease Prediction 🫀
Predict presence of heart disease from 13 clinical attributes — age, sex, chest pain type, resting
blood pressure, cholesterol, max heart rate, exercise-induced angina, and others — with logistic
regression on a stratified split.

**Dataset:** UCI Heart Disease (Cleveland)

---

### 7. Credit Card Fraud Detection 💳
284,807 transactions, 492 of them fraudulent — 0.172%. The interesting part isn't the model, it's
the evaluation: the data is split *before* under-sampling so the test set keeps the real fraud rate,
and results are reported as precision, recall, F1 and PR-AUC rather than accuracy. A
"predict legitimate for everything" baseline scores ~99.8% accuracy and catches zero fraud, which
is the whole point.

**Dataset:** Credit Card Fraud Detection (Kaggle, ULB) — download required

---

### 8. Neural Network from Scratch 🧠
A three-layer network (784 → 10 → 10) for MNIST digits in pure NumPy. Forward propagation with ReLU
and a numerically stable softmax, backpropagation derived by hand through both layers, and vanilla
gradient descent — no TensorFlow, no PyTorch, no autograd.

**Dataset:** MNIST via Kaggle Digit Recognizer — download required

---

### 9. Dog vs Cat Image Classification 🐶🐱
A CNN built from three `Conv2D` → `BatchNormalization` → `MaxPooling2D` blocks, then dense layers
with dropout. Trained 10 epochs on a Colab T4. Validation accuracy peaks at 81.7% while training
accuracy climbs to 96% and validation loss starts rising — textbook overfitting, and the plots show
it clearly. The fix is `EarlyStopping` on `val_loss` plus heavier dropout (0.1 is far too light) and
data augmentation.

**Dataset:** Kaggle Dogs vs. Cats (`salader/dogsvscats`)

---

### 10. Stock Price Prediction (LSTM) 📈
Next-day close for `^NSEI`, windowed into 30-day sequences and run through stacked LSTM layers in
PyTorch, with LayerNorm and Xavier initialisation.

The model tracks the trend but lags sharp reversals and understates sudden spikes — which is the
honest result, and a known trap with price-level forecasting: a model that simply echoes yesterday's
close looks good on RMSE. The next iteration is to predict *returns* instead of levels and to score
against that naive baseline.

**Dataset:** Yahoo Finance via `yfinance`

---

### 11. Movie Recommender System 🎬
Content-based recommendations over the TMDB 5000 dataset. Overview, genres, keywords, top-3 cast,
and director are merged into a single "tags" string per film, stemmed, vectorised with a
bag-of-words `CountVectorizer` (5000 features), and ranked by cosine similarity.

Ships with a Streamlit frontend and a pytest suite over the recommendation logic. The movie and
credits tables are joined on TMDB id rather than title — three titles are duplicated in this dataset
(*Batman*, *The Host*, *Out of the Blue*) and joining on title cross-joins them into each other's
cast lists.

**Dataset:** TMDB 5000 Movie Dataset (Kaggle)

---

## Repo layout

```
22-ML-Projects/
├── <project folder>/
│   ├── *.ipynb          # the notebook
│   ├── *.csv            # data, where it fits in git
│   └── README.md        # project-specific notes
├── Movie Recommender System/
│   ├── app.py           # Streamlit frontend
│   ├── recommender.py   # recommendation logic
│   └── data/            # zipped TMDB CSVs
└── tests/               # pytest suite
```

## What's next

Eleven more projects, and a few revisits: returns-based forecasting for the LSTM, `EarlyStopping`
and augmentation for the CNN, and cross-validation for the small-dataset projects (Heart Disease and
SONAR) where a single split doesn't say much.
