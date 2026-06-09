"""Run end-to-end price modeling and save outputs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from price_modeling import (  # noqa: E402
    DEFAULT_FEATURES,
    SELECTED_FEATURES,
    SELECTED_FEATURES_NO_LEAK,
    compare_models,
    cross_validate_model,
    evaluate_predictions,
    get_xy,
    hyperparameter_search,
    load_or_build_dataset,
    make_split,
    outputs_dir,
)


def main() -> None:
    dataset = load_or_build_dataset()
    print(f"Dataset shape: {dataset.shape}")

    split_results = []
    for strategy, groups_col in [
        ("random", None),
        ("neighbourhood", "neighbourhood_cleansed"),
        ("host", "host_id"),
    ]:
        X, y, groups_neighbourhood, groups_host = get_xy(dataset, DEFAULT_FEATURES)
        groups = None
        if groups_col == "neighbourhood_cleansed":
            groups = groups_neighbourhood
        elif groups_col == "host_id":
            groups = groups_host

        if strategy != "random" and groups is None:
            print(f"Skipping split '{strategy}' because group column is unavailable.")
            continue

        X_train, X_test, y_train, y_test = make_split(X, y, strategy=strategy, groups=groups)
        comparison = compare_models(X_train, X_test, y_train, y_test)
        comparison.insert(0, "split", strategy)
        split_results.append(comparison)
        print(f"\n=== Split: {strategy} ===")
        print(comparison.to_string(index=False))

    all_splits = pd.concat(split_results, ignore_index=True)
    all_splits.to_csv(outputs_dir() / "model_comparison_by_split.csv", index=False)

    # Leakage sensitivity: with vs without estimated revenue feature.
    leakage_rows = []
    for feature_set, label in [
        (SELECTED_FEATURES, "with_revenue_feature"),
        (SELECTED_FEATURES_NO_LEAK, "without_revenue_feature"),
    ]:
        X, y, _, _ = get_xy(dataset, feature_set)
        X_train, X_test, y_train, y_test = make_split(X, y, strategy="random")
        comparison = compare_models(X_train, X_test, y_train, y_test)
        comparison.insert(0, "feature_set", label)
        leakage_rows.append(comparison)
    leakage_df = pd.concat(leakage_rows, ignore_index=True)
    leakage_df.to_csv(outputs_dir() / "leakage_sensitivity.csv", index=False)
    print("\n=== Leakage sensitivity (random split) ===")
    print(leakage_df.to_string(index=False))

    # Pick best model on random split for tuning (no leakage features).
    X, y, _, _ = get_xy(dataset, DEFAULT_FEATURES)
    X_train, X_test, y_train, y_test = make_split(X, y, strategy="random")
    baseline = compare_models(X_train, X_test, y_train, y_test)
    best_model = baseline.iloc[0]["model"]
    print(f"\nBest baseline model: {best_model}")

    cv_df = cross_validate_model(X_train, y_train, model_name=best_model)
    cv_df.to_csv(outputs_dir() / f"cv_{best_model}.csv", index=False)
    print("\n=== Cross-validation ===")
    print(cv_df.to_string(index=False))

    tuned_grid, grid_results = hyperparameter_search(
        X_train, y_train, model_name=best_model, method="grid", cv=3
    )
    tuned_random, random_results = hyperparameter_search(
        X_train, y_train, model_name=best_model, method="random", cv=3
    )

    grid_results.head(10).to_csv(outputs_dir() / "grid_search_top10.csv", index=False)
    random_results.head(10).to_csv(outputs_dir() / "random_search_top10.csv", index=False)

    grid_test = evaluate_predictions(y_test, tuned_grid.predict(X_test))
    random_test = evaluate_predictions(y_test, tuned_random.predict(X_test))
    tuning_comparison = pd.DataFrame(
        [
            {"method": "grid_search", **grid_test},
            {"method": "random_search", **random_test},
        ]
    )
    tuning_comparison.to_csv(outputs_dir() / "tuning_comparison.csv", index=False)

    print("\n=== Tuned model test metrics (no-leak features) ===")
    print(tuning_comparison.to_string(index=False))

    print(f"\nOutputs saved to: {outputs_dir()}")


if __name__ == "__main__":
    import pandas as pd

    main()
