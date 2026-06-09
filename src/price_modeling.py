"""Price prediction utilities for Prague Airbnb case study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_prepro_func import get_release_df
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "price_log"
ID_COLS = ["id", "name"]
FEATURE_FILE = "airbnb_listings_features.csv.gz"
LISTINGS_FILE = "listings.csv.gz"

# Features selected in teammate EDA notebook (Random Forest, 95% importance).
SELECTED_FEATURES = [
    "bedrooms_log",
    "estimated_revenue_l365d_log",
    "property_type_clean_Private room in rental unit",
    "bathrooms_log",
    "room_type_Shared room",
    "host_response_rate",
    "availability_365",
    "estimated_occupancy_l365d",
    "reviews_per_month_log",
    "calculated_host_listings_count_entire_homes_log",
    "availability_60",
]

# Same set without potential target leakage from estimated revenue.
SELECTED_FEATURES_NO_LEAK = [f for f in SELECTED_FEATURES if f != "estimated_revenue_l365d_log"]

# Primary feature set for reporting (avoids target leakage).
DEFAULT_FEATURES = SELECTED_FEATURES_NO_LEAK


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return project_root() / "data"


def outputs_dir() -> Path:
    path = project_root() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv_from_path(path: Path, index_col: int | str | None = None) -> pd.DataFrame:
    """Read CSV or GZ with correct compression settings."""
    kwargs: dict[str, Any] = {"low_memory": False}
    if index_col is not None:
        kwargs["index_col"] = index_col
    if str(path).endswith(".gz"):
        kwargs["compression"] = "gzip"
    return pd.read_csv(path, **kwargs)


def _read_csv_smart(filename: str) -> pd.DataFrame | None:
    """Load CSV/GZ from local data/ first, then teammate GitHub Release helper."""
    local_path = data_dir() / filename
    if local_path.exists():
        print(f"Loaded local file: {local_path}")
        return _read_csv_from_path(local_path)

    # Also check project root (e.g. file shared in repo root).
    root_path = project_root() / filename
    if root_path.exists():
        print(f"Loaded local file: {root_path}")
        return _read_csv_from_path(root_path)

    print(f"Local file not found, trying GitHub Release via data_prepro_func: {filename}")
    df = get_release_df(filename)
    if df is not None:
        return df

    return None


def load_feature_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load preprocessed feature file produced by EDA notebook."""
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        return _read_csv_from_path(path, index_col=0)

    df = _read_csv_smart(FEATURE_FILE)
    if df is None:
        raise FileNotFoundError(
            f"Feature file '{FEATURE_FILE}' not found locally or on GitHub Release."
        )
    return df


def _clean_price(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def _clean_rate(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce") / 100.0
    )


def build_feature_dataset_from_listings(listings_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild teammate feature set from raw Inside Airbnb listings."""
    df = listings_df.copy()

    df["price"] = _clean_price(df["price"])
    df["host_response_rate"] = _clean_rate(df["host_response_rate"])

    for col in [
        "bedrooms",
        "bathrooms",
        "reviews_per_month",
        "calculated_host_listings_count_entire_homes",
        "estimated_revenue_l365d",
    ]:
        df[f"{col}_log"] = np.log1p(df[col].fillna(0))

    top_property = df["property_type"].value_counts().head(5).index.tolist()
    df["property_type_clean"] = df["property_type"].apply(
        lambda x: x if pd.notna(x) and x in top_property else "Other"
    )

    property_dummies = pd.get_dummies(
        df["property_type_clean"], prefix="property_type_clean", dtype=int
    )
    room_dummies = pd.get_dummies(df["room_type"], prefix="room_type", dtype=int)

    feature_cols = [
        "bedrooms_log",
        "estimated_revenue_l365d_log",
        "bathrooms_log",
        "host_response_rate",
        "availability_365",
        "estimated_occupancy_l365d",
        "reviews_per_month_log",
        "calculated_host_listings_count_entire_homes_log",
        "availability_60",
    ]

    dataset = pd.concat(
        [
            df[["id", "name", "price", "neighbourhood_cleansed", "host_id"] + feature_cols],
            property_dummies,
            room_dummies,
        ],
        axis=1,
    )
    dataset[TARGET] = np.log1p(dataset["price"])
    dataset = dataset.dropna(subset=[TARGET])
    return dataset


def _attach_group_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add split-group columns from listings when feature file omits them."""
    needs_neighbourhood = "neighbourhood_cleansed" not in dataset.columns
    needs_host = "host_id" not in dataset.columns
    if not needs_neighbourhood and not needs_host:
        return dataset
    if "id" not in dataset.columns:
        return dataset

    listings_df = _read_csv_smart(LISTINGS_FILE)
    if listings_df is None or "id" not in listings_df.columns:
        print("Warning: group columns unavailable; neighbourhood/host splits will be skipped.")
        return dataset

    merge_cols = ["id"]
    if needs_neighbourhood and "neighbourhood_cleansed" in listings_df.columns:
        merge_cols.append("neighbourhood_cleansed")
    if needs_host and "host_id" in listings_df.columns:
        merge_cols.append("host_id")

    if len(merge_cols) == 1:
        return dataset

    group_df = listings_df[merge_cols].drop_duplicates(subset=["id"])
    return dataset.merge(group_df, on="id", how="left")


def load_or_build_dataset() -> pd.DataFrame:
    """Load modeling dataset with teammate data loader integration.

    Priority:
    1. airbnb_listings_features.csv.gz (local data/ or GitHub Release)
    2. listings.csv.gz (local data/ or GitHub Release) + rebuild features
    """
    feature_df = _read_csv_smart(FEATURE_FILE)
    if feature_df is not None:
        if "price_log" not in feature_df.columns and "price" in feature_df.columns:
            feature_df["price_log"] = np.log1p(feature_df["price"])
        return _attach_group_columns(feature_df)

    listings_df = _read_csv_smart(LISTINGS_FILE)
    if listings_df is not None:
        print("Feature file unavailable; rebuilding features from listings.")
        return build_feature_dataset_from_listings(listings_df)

    raise FileNotFoundError(
        "Could not load modeling data. Provide one of:\n"
        f"  - data/{FEATURE_FILE}\n"
        f"  - data/{LISTINGS_FILE}\n"
        "Or configure git remote and upload files to GitHub Release tag 'data'."
    )


def get_xy(
    dataset: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, pd.Series | None]:
    features = features or DEFAULT_FEATURES
    missing = [f for f in features if f not in dataset.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    model_df = dataset.dropna(subset=[TARGET] + features).copy()
    X = model_df[features]
    y = model_df[TARGET]

    groups_neighbourhood = model_df["neighbourhood_cleansed"] if "neighbourhood_cleansed" in model_df else None
    groups_host = model_df["host_id"] if "host_id" in model_df else None
    return X, y, groups_neighbourhood, groups_host


def make_split(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str = "random",
    groups: pd.Series | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if strategy == "random":
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    if groups is None:
        raise ValueError(f"strategy='{strategy}' requires group labels")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def build_pipeline(model_name: str) -> Pipeline:
    models: dict[str, Any] = {
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "knn": KNeighborsRegressor(),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    # Scale for linear / distance-based models; tree models are mostly unaffected.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", models[model_name]),
        ]
    )


def evaluate_predictions(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse_czk": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_czk": float(mean_absolute_error(y_true, y_pred)),
        "r2_czk": float(r2_score(y_true, y_pred)),
    }


def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    model_names = model_names or ["ridge", "random_forest", "gradient_boosting", "knn"]
    rows = []

    for name in model_names:
        pipe = build_pipeline(name)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate_predictions(y_test, preds)
        rows.append({"model": name, **metrics})

    return pd.DataFrame(rows).sort_values("rmse_log")


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest",
    cv: int = 5,
) -> pd.DataFrame:
    pipe = build_pipeline(model_name)
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "metric": ["rmse_log", "mae_log", "r2_log"],
            "mean": [
                -scores["test_neg_root_mean_squared_error"].mean(),
                -scores["test_neg_mean_absolute_error"].mean(),
                scores["test_r2"].mean(),
            ],
            "std": [
                scores["test_neg_root_mean_squared_error"].std(),
                scores["test_neg_mean_absolute_error"].std(),
                scores["test_r2"].std(),
            ],
        }
    )


def hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "random_forest",
    method: str = "grid",
    cv: int = 3,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame]:
    param_grids: dict[str, list[dict[str, Any]]] = {
        "ridge": [{"model__alpha": [0.1, 1.0, 10.0, 100.0]}],
        "random_forest": [
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 3, 5],
            }
        ],
        "gradient_boosting": [
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3, 4],
            }
        ],
        "knn": [{"model__n_neighbors": [3, 5, 11, 21], "model__weights": ["uniform", "distance"]}],
    }

    pipe = build_pipeline(model_name)
    grid = param_grids[model_name]

    if method == "grid":
        search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    elif method == "random":
        search = RandomizedSearchCV(
            pipe,
            grid,
            n_iter=12,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError("method must be 'grid' or 'random'")

    search.fit(X_train, y_train)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return search.best_estimator_, results
