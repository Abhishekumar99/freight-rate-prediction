
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_ROWS = 12_000
EXPECTED_IDS = {
    f"TE-{i:06d}" for i in range(1, EXPECTED_ROWS + 1)
}

DECEMBER_DATES = pd.date_range(
    "2025-12-01", "2025-12-31", freq="D"
)


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def read_csv(path, label):
    path = Path(path)
    if not path.is_file():
        fail(f"{label} file not found: {path}")

    try:
        return pd.read_csv(path)
    except Exception as exc:
        fail(f"Could not read {label}: {exc}")


def validate_predictions(df):
    if list(df.columns) != ["load_id", "predicted_rate"]:
        fail("Required columns: load_id,predicted_rate")

    if len(df) != EXPECTED_ROWS:
        fail(f"Expected {EXPECTED_ROWS} prediction rows.")

    if df["load_id"].isna().any():
        fail("Missing load_id values.")

    if df["load_id"].duplicated().any():
        fail("Duplicate load_id values.")

    ids = set(df["load_id"].astype(str))

    if ids != EXPECTED_IDS:
        fail(
            f"ID mismatch: {len(EXPECTED_IDS - ids)} missing, "
            f"{len(ids - EXPECTED_IDS)} extra."
        )

    rates = pd.to_numeric(df["predicted_rate"], errors="coerce")

    if rates.isna().any() or not np.isfinite(rates).all():
        fail("Invalid predicted_rate values.")

    if (rates <= 0).any():
        fail("Predicted rates must be positive.")

    print("Validated 12,000 final predictions.")


def validate_december(df):
    columns = [
        "pickup", "delivery", "distance", "equipment",
        "weight", "date", "predicted_rate"
    ]

    if list(df.columns) != columns:
        fail(
            "December CSV must contain these columns in order: "
            + ",".join(columns)
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        fail("Invalid December dates.")

    if len(df) != 31 or set(df["date"]) != set(DECEMBER_DATES):
        fail("Need one row for every date in December 2025.")

    if df["date"].duplicated().any():
        fail("Duplicate December dates.")

    if not df["pickup"].eq("Lexington").all():
        fail("December pickup must be Lexington.")

    if not df["delivery"].eq("Fort Wayne").all():
        fail("December delivery must be Fort Wayne.")

    if not np.isclose(
        pd.to_numeric(df["distance"], errors="coerce"), 360
    ).all():
        fail("December distance must be 360.")

    if not df["equipment"].eq("Dry Van").all():
        fail("December equipment must be Dry Van.")

    if not np.isclose(
        pd.to_numeric(df["weight"], errors="coerce"), 32000
    ).all():
        fail("December weight must be 32000.")

    rates = pd.to_numeric(df["predicted_rate"], errors="coerce")

    if rates.isna().any() or not np.isfinite(rates).all():
        fail("Invalid December predictions.")

    if (rates <= 0).any():
        fail("December predictions must be positive.")

    print("Validated 31 fixed December predictions.")
    return df.sort_values("date")


def save_chart(df, output):
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)

    ax.plot(
        df["date"],
        df["predicted_rate"],
        color="#064A56",
        linewidth=2.6,
        marker="o",
        markersize=3.2
    )

    floor = float(df["predicted_rate"].min())

    ax.fill_between(
        df["date"],
        df["predicted_rate"],
        floor - max(10, floor * 0.02),
        color="#064A56",
        alpha=0.08
    )

    ax.set_title(
        "Candidate: December 2025 Predicted Load Rate",
        loc="left",
        fontsize=15,
        fontweight="bold"
    )
    ax.set_ylabel("Predicted rate ($)")
    ax.grid(axis="y", color="#D9E2E4")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=35)

    ax.text(
        0, -0.30,
        "Lexington to Fort Wayne | 360 miles | Dry Van | "
        "32,000 lb | only date changes",
        transform=ax.transAxes,
        fontsize=9
    )

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Validate freight rate predictions and create chart."
    )

    parser.add_argument("--predictions", required=True)
    parser.add_argument("--december-predictions", required=True)
    parser.add_argument(
        "--output-dir", default="scorer_results"
    )

    args = parser.parse_args()

    predictions = read_csv(args.predictions, "Predictions")
    validate_predictions(predictions)

    december = read_csv(
        args.december_predictions, "December predictions"
    )
    december = validate_december(december)

    chart_path = (
        Path(args.output_dir) / "candidate_december.png"
    )

    save_chart(december, chart_path)

    print(f"Created chart: {chart_path}")
    print(
        "Final validation metrics are calculated by "
        "Spotter after submission."
    )


if __name__ == "__main__":
    main()
