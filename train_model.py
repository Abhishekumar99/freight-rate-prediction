
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RANDOM_SEED = 42


def make_features(df, medians=None):
    """Build features available in both validation and December inputs."""
    X = df[
        ["pickup", "delivery", "distance", "equipment", "weight", "date"]
    ].copy()

    dates = pd.to_datetime(X.pop("date"), errors="coerce")

    X["month"] = dates.dt.month.fillna(0).astype(int).astype(str)
    X["day_of_week"] = dates.dt.dayofweek.fillna(0).astype(int).astype(str)
    X["day_of_month"] = dates.dt.day
    X["day_of_year"] = dates.dt.dayofyear

    X["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    X["weekday_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)

    categorical = [
        "pickup", "delivery", "equipment", "month", "day_of_week"
    ]

    for col in categorical:
        X[col] = X[col].fillna("Unknown").astype(str)

    numeric = [
        "distance", "weight", "day_of_month", "day_of_year",
        "month_sin", "month_cos", "weekday_sin", "weekday_cos"
    ]

    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Learn imputation values from training data only.
    if medians is None:
        medians = X[numeric].median().fillna(0)

    X[numeric] = X[numeric].fillna(medians).fillna(0)

    return X, categorical, medians


def main():
    train = pd.read_csv(DATA / "train_test.csv")
    validation = pd.read_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    december = pd.read_csv(DATA / "december_chart_inputs.csv")

    required = {
        "pickup", "delivery", "distance", "equipment",
        "weight", "date", "posted_rate"
    }
    if not required.issubset(train.columns):
        raise ValueError(f"Missing training columns: {required - set(train.columns)}")

    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    validation["date"] = pd.to_datetime(validation["date"], errors="coerce")
    december["date"] = pd.to_datetime(december["date"], errors="coerce")

    if train["date"].isna().any() or validation["date"].isna().any():
        raise ValueError("Invalid dates found in training or validation data.")

    if train["posted_rate"].isna().any():
        raise ValueError("The training target contains missing values.")

    print("Training shape:", train.shape)
    print("Validation shape:", validation.shape)
    print("\nMissing training values:\n", train.isna().sum())
    print("\nTarget summary:\n", train["posted_rate"].describe())

    # Temporal holdout: train through September, validate on October.
    holdout_start = pd.Timestamp("2025-10-01")
    fit_mask = train["date"] < holdout_start
    holdout_mask = train["date"] >= holdout_start

    if not fit_mask.any() or not holdout_mask.any():
        raise ValueError("The expected chronological split is empty.")

    X_train, categorical, medians = make_features(
        train.loc[fit_mask]
    )
    X_holdout, _, _ = make_features(
        train.loc[holdout_mask], medians
    )

    y_train = train.loc[fit_mask, "posted_rate"]
    y_holdout = train.loc[holdout_mask, "posted_rate"]

    # Baseline: always predict the training-set median.
    baseline = np.full(len(y_holdout), y_train.median())
    print("\nBaseline MAE:",
          round(mean_absolute_error(y_holdout, baseline), 2))

    model = CatBoostRegressor(
        iterations=1200,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=5,
        random_seed=RANDOM_SEED,
        verbose=200,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical,
        eval_set=(X_holdout, y_holdout),
        early_stopping_rounds=120,
        verbose=200
    )

    holdout_predictions = np.maximum(
        model.predict(X_holdout), 1.0
    )

    mae = mean_absolute_error(y_holdout, holdout_predictions)
    rmse = np.sqrt(
        mean_squared_error(y_holdout, holdout_predictions)
    )
    r2 = r2_score(y_holdout, holdout_predictions)

    print("\nChronological validation results")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2:    {r2:.4f}")

    # Refit on all labeled data using the selected iteration count.
    X_all, categorical, medians = make_features(train)
    iterations = max(100, int(model.best_iteration_) + 1)

    final_model = CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        l2_leaf_reg=5,
        random_seed=RANDOM_SEED,
        verbose=200,
        allow_writing_files=False
    )

    final_model.fit(
        X_all,
        train["posted_rate"],
        cat_features=categorical
    )

    # Predict every validation load and preserve template IDs.
    X_validation, _, _ = make_features(validation, medians)

    if len(template) != len(validation):
        raise ValueError("Template and validation row counts differ.")

    if template["load_id"].tolist() != validation["load_id"].tolist():
        raise ValueError("Template IDs/order do not match validation.csv.")

    predictions = np.maximum(
        final_model.predict(X_validation), 1.0
    )

    output = template[["load_id"]].copy()
    output["predicted_rate"] = np.round(predictions, 2)
    output.to_csv(ROOT / "validation_predictions.csv", index=False)

    # Generate all 31 December scenario predictions.
    X_december, _, _ = make_features(december, medians)
    december["predicted_rate"] = np.round(
        np.maximum(final_model.predict(X_december), 1.0), 2
    )
    december.to_csv(DATA / "december_chart_inputs.csv", index=False)

    # Save the trained model for reproducibility.
    final_model.save_model(ROOT / "freight_rate_model.cbm")

    print("\nSaved validation_predictions.csv")
    print("Updated data/december_chart_inputs.csv")
    print("Saved freight_rate_model.cbm")
    print("Validation predictions:", len(output))
    print("Positive predictions:",
          bool((output["predicted_rate"] > 0).all()))


if __name__ == "__main__":
    main()
