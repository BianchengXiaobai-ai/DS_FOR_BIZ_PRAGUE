# DS for Business — Prague Airbnb Case Study

Group 13 · Focus Area 1: **Price Prediction**

## Project structure

```
├── notebooks/
│   ├── 01_eda_data_preparation.ipynb   # Part 1: EDA, cleaning, feature selection
│   └── 02_price_prediction.ipynb       # Part 2: Pipeline, splits, models, tuning
├── src/
│   ├── data_prepro_func.py             # Data loader (GitHub Release helper)
│   └── price_modeling.py               # Modeling utilities
├── outputs/                            # Result tables and figures for presentation
├── data/                               # Local data (not in git — see below)
├── run_price_modeling.py               # One-click reproduce script
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
```

Place the feature file in `data/`:

- `data/airbnb_listings_features.csv.gz` (from Part 1 / teammate Release)

Then either:

```bash
python run_price_modeling.py
```

or open `notebooks/02_price_prediction.ipynb`.

## Data

Large files are **not** stored in this repo. Options:

1. Copy `airbnb_listings_features.csv.gz` into `data/` (shared via WhatsApp / Release)
2. Upload to **GitHub Release** tag `data` and use `get_release_df()` in `data_prepro_func.py`

## Part 2 key notes

- **Target:** `price_log = log1p(price)` (CZK)
- **Primary features:** 10 variables (`DEFAULT_FEATURES`), excluding `estimated_revenue_l365d_log` (target leakage)
- **Splits:** random / neighbourhood / host (`GroupShuffleSplit`)
- **Models:** Ridge, Random Forest, Gradient Boosting, KNN
- **Tuning:** `GridSearchCV` vs `RandomizedSearchCV`

## Team分工

| Part | Topic | Owner |
|------|-------|-------|
| 1 | EDA + feature engineering | Teammate (BO) |
| 2 | Price prediction modeling | — |
| 3 | Reviews + time patterns | TBD |
| 4 | Geo clustering + presentation | TBD |
