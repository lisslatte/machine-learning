# Decision Tree Regression for Property Price Prediction

## Overview
This project implements a **Decision Tree Regressor** to predict property transaction prices (`Tran_Pri`). The workflow includes data preprocessing, feature engineering, model tuning, evaluation, and extensive visualization. The goal is to build an interpretable model that can estimate property values based on location, structural attributes, and time features.

## Data
- **Training set**: `../data/TRAIN.csv` (165,808 records)
- **Test set**: `../data/TEST.csv` (62,733 records)
- **Time range**: Train covers 2012-11-22 to 2021-06-30; test covers 2021-07-01 to 2023-02-13.

## Workflow
1. **Load & clean data** – Parse dates, convert target to numeric, handle missing values.
2. **Feature engineering**:
   - Extract `Tran_Year`, `Tran_Month`, `Tran_Quarter` from transaction date.
   - Compute `Building_Age` = `Tran_Year - Build_Yr_Num`.
   - Parse elevator ratio (`EA_Ratio_Num`) from `EA_Ratio` (e.g., `1E2A` → 0.5).
   - Create `Prop_Flo_Level` (Lower/Middle/Upper → 1/2/3) and `Floor_Ratio`.
   - Log-transform floor area → `Log_Flo_Area`.
3. **Feature selection**:
   - **Numeric** (11): `Tot_Flo`, `Flo_Area`, `Build_Yr_Num`, `Tran_Year`, `Tran_Month`, `Tran_Quarter`, `Build_Age`, `EA_Ratio_Num`, `Prop_Flo_Level`, `Floor_Ratio`, `Log_Flo_Area`.
   - **Categorical** (13): `City`, `Reg`, `Dist`, `Neigh`, `Layout`, `Prop_Flo`, `Build_Type`, `Orient`, `ReNov`, `Build_Str`, `Lift`, `Prop_Type`, `Build_Pur`.
4. **Preprocessing**:
   - Impute missing: median for numeric, `"Unknown"` for categorical.
   - One‑hot encode categoricals (drop first), resulting in 13,246 features after encoding.
5. **Train/Validation split**: 80% train (132,646), 20% validation (33,162).
6. **Model tuning**:
   - `DecisionTreeRegressor` with `GridSearchCV` (3‑fold CV, scoring=`r2`).
   - Parameter grid: `max_depth`=[10,15], `min_samples_split`=[200,500], `min_samples_leaf`=[50,100].
   - Best parameters: `max_depth=15`, `min_samples_split=200`, `min_samples_leaf=50`.
   - Best CV R²: **0.6911**.
7. **Evaluation** (validation set):
   - **MAE**: 507.64
   - **RMSE**: 853.68
   - **R²**: 0.7246
8. **Visualizations**:
   - Top 15 feature importances.
   - Actual vs. predicted scatter plot.
   - Residual distribution histogram.
   - Grid search heatmaps (by `min_samples_leaf`).
   - Partial dependence plots for `Log_Flo_Area` and `EA_Ratio_Num`.
   - Price‑band confusion matrix (binned actual vs. predicted).

## Key Findings
- **Floor area** (`Flo_Area`) is the most important predictor (47.3% importance).
- Location features (`Reg_Xihu`, `Reg_Shangcheng`) strongly influence price.
- Transaction year (`Tran_Year`) captures market trend.
- Elevator ratio and total floors also contribute notably.

## Output Files
- `dt_X_train_ready.csv`, `dt_X_test_ready.csv`, `dt_y_train_ready.csv`
- `dt_grid_search_results.csv`
- `dt_feature_importance_best.csv`
- `dt_test_predictions.csv`
- `dt_best_model_summary.csv`
- Multiple figures (`fig_*.png`) and a figure log (`dt_figure_log.csv`).

## Usage
Run the Jupyter notebook `decision_tree_model.ipynb` sequentially. All outputs are saved in the working directory. The best model is not persisted but can be recreated using the reported parameters.