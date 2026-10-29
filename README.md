# Freight Rate Prediction Using Machine Learning

## Project Overview

This project predicts freight load rates using historical shipment data and machine learning. It uses CatBoost Regression to learn patterns from route information, distance, equipment type, weight, and date-related features.

The objective is to generate predictions for 12,000 validation loads and produce daily rate predictions for a fixed December 2025 scenario.

## Technologies Used

- Python
- Pandas and NumPy
- CatBoost
- Scikit-learn
- Matplotlib
- Git and GitHub

## Project Structure

```text
freight-rate-prediction/
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
├── scorer_results/
│   └── candidate_december.png
├── train_model.py
├── score.py
├── requirements.txt
├── README.md
├── validation_predictions.csv
└── freight_rate_model.cbm
```

## Machine Learning Workflow

1. **Data exploration:** Inspect dataset dimensions, missing values, dates, and target distribution.
2. **Feature engineering:** Use pickup, delivery, distance, equipment, weight, and date-derived features.
3. **Data validation:** Train on earlier dates and evaluate on a later chronological holdout.
4. **Model training:** Train a CatBoost Regressor using MAE as the optimization objective.
5. **Model evaluation:** Calculate Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².
6. **Prediction generation:** Generate predictions for all validation loads and the 31 December scenarios.
7. **Output validation:** Run the provided scoring script to check the CSV files and create the December prediction chart.

## Setup Instructions

Create and activate a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Project

From the project root, train the model and generate predictions:

```powershell
python train_model.py
```

Validate the output files and generate the December chart:

```powershell
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer should confirm 12,000 final predictions and 31 December predictions and create:

```text
scorer_results/candidate_december.png
```

## Validation Results

The local chronological holdout produced the following results:

| Metric | Result |
|---|---:|
| MAE | $135.38 |
| RMSE | $654.93 |
| R² | 0.8164 |

These results are from the local October holdout and do not represent the assessment provider's hidden final evaluation score.

## Output Files

- `validation_predictions.csv` — predictions for the 12,000 validation loads.
- `data/december_chart_inputs.csv` — completed December scenario predictions.
- `freight_rate_model.cbm` — saved CatBoost model.
- `scorer_results/candidate_december.png` — December prediction chart.

## Limitations

Model performance on future dates or previously unseen routes may differ from the local holdout results. Further evaluation, feature analysis, and model tuning may improve generalization.

## Reproducibility

The training script, dependencies, and execution instructions are included in this repository. Run the commands above to reproduce the predictions and scoring checks.