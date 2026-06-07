# CatBoost House Price Regression

This folder trains a `CatBoostRegressor` to predict house price (`Tran_Pri`) using:

- training data: `../data/TRAIN.csv`
- test data: `../data/TEST.csv`

## Files

- `catboost.ipynb`: notebook version with step-by-step markdown explanations
- `outputs/`: saved metrics, predictions, charts, and model artifacts

## Run

You can run all the notebook.

### Notebook

Open `catboost.ipynb` in Jupyter or VS Code and run the cells from top to bottom.

- configuration and file paths
- helper functions
- feature engineering
- time-based holdout splitting
- CatBoost training
- evaluation on train, holdout, and test
- feature importance and SHAP explanations
- saving outputs

## Validation Setup

The script uses a time-based holdout split instead of a random split:

- earliest 80% of `TRAIN.csv` by `Tran_Dt` for training
- latest 20% of `TRAIN.csv` by `Tran_Dt` for holdout validation
- all of `TEST.csv` for the future-period test

## Feature Engineering

The model adds extra engineered features, including:

- calendar and cyclic month features
- cleaned build year and property age features
- clipped and log floor area features
- area-per-room and area-per-floor ratios
- parsed room counts from `Layout`
- parsed elevator/apartment counts from `EA_Ratio`
- floor-position and building-height flags
- composite location, layout, building, and renovation categorical keys

## Outputs

After running, the script or notebook writes these files into `outputs/`:

- `metrics.json`
- `metrics_comparison.csv`
- `metrics_comparison.png`
- `actual_vs_predicted.csv`
- `actual_vs_predicted.png`
- `holdout_actual_vs_predicted.png`
- `split_predictions.csv`
- `test_full_with_predictions.csv`
- `feature_importance.csv`
- `feature_importance.png`
- `shap_feature_importance.csv`
- `shap_feature_importance.png`
- `catboost_house_price_model.cbm`

`metrics_comparison.csv` compares `rmse`, `mae`, and `r2` across:

- `train`
- `holdout`
- `test`
