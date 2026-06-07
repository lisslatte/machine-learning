# SVM Regression for Property Price Prediction

## Overview
This project builds a **Support Vector Machine (SVM) regressor** to predict property transaction prices (`Tran_Pri`). A time‑based validation split ensures realistic evaluation, and a `TransformedTargetRegressor` with `log1p` transformation stabilizes the target. The pipeline includes custom feature engineering, robust preprocessing, hyperparameter tuning, and comprehensive visual diagnostics.

## Data
- **Training set**: `../data/TRAIN.csv` (165,808 records, 2012‑11‑22 to 2021‑06‑30)
- **Test set**: `../data/TEST.csv` (62,733 records, 2021‑07‑01 to 2023‑02‑13)

## Workflow
1. **Time‑based split**: Sort train by date, use latest 20% as validation (time‑ordered).
2. **Feature engineering**:
   - Time features: `Tran_Year`, `Tran_Month`, `Tran_Quarter`.
   - `Building_Age` = `Tran_Year - Build_Yr_Filled`.
   - Binary `Has_Lift` and interaction `HighFloor_NoLift`.
   - `Area_per_Room`, `Central_District` (based on `Reg`).
3. **Feature selection** (after ablation):
   - **Numeric** (14): `Tran_Day`, `Tot_Flo`, `Flo_Area`, `Build_Yr`, `Total_Rooms`, `Build_Yr_Filled`, `Tran_Year`, `Tran_Month`, `Tran_Quarter`, `Building_Age`, `Has_Lift`, `HighFloor_NoLift`, `Area_per_Room`, `Central_District`.
   - **Categorical** (9): `Reg`, `Dist`, `Prop_Flo`, `Build_Type`, `ReNov`, `Build_Str`, `Lift`, `Prop_Type`, `Build_Pur`.
4. **Preprocessing pipeline**:
   - Numeric: median imputation → `StandardScaler`.
   - Categorical: most‑frequent imputation → `OneHotEncoder(handle_unknown='ignore')`.
5. **Model**:
   - `SVR` with RBF kernel.
   - Wrapped in `TransformedTargetRegressor` with `func=np.log1p`, `inverse_func=np.expm1`.
6. **Tuning strategy**:
   - Tune on capped recent window (15k rows) with `TimeSeriesSplit` (3 splits).
   - Parameter grid:
     - `C`: [5, 10, 20, 30]
     - `epsilon`: [0.03, 0.05, 0.1]
     - `gamma`: [0.001, 0.003, 0.01]
   - Best parameters: `C=5`, `epsilon=0.1`, `gamma=0.01`.
   - Best CV MAE: 495.25.
   - Refit final model on larger recent window (40k rows).
7. **Evaluation** (validation & holdout test):

| Dataset                     | MAE     | RMSE    | R²     | MAPE (%) | SMAPE (%) |
|-----------------------------|---------|---------|--------|----------|-----------|
| Validation (last 20% train) | 384.43  | 753.22  | 0.826  | 16.77    | 15.81     |
| Holdout Test (latest years) | 503.03  | 923.34  | 0.773  | 25.87    | 21.55     |

8. **Visualizations**:
   - Price trend over time (train vs. test).
   - Actual vs. predicted scatter (test set).
   - Residual distribution and city‑level MAE.
   - Error by price segment (low/mid/high).
   - Monthly MAE on test set.
   - **Permutation importance** (model‑agnostic) and feature effect plots for top predictors.

## Key Findings
- **Central_District**, **Reg**, **Dist**, **Total_Rooms**, and **Flo_Area** are the most influential features.
- The model performs well on typical properties but shows larger errors for high‑value or unusual properties (P90 APE ≈ 74%).
- Time‑based validation confirms that market trends are partially captured, but prediction error increases in the most recent periods.

## Business Recommendation
- Use the model as a **decision‑support tool** for price estimation and negotiation.
- Best suited for screening fair‑value properties; extra caution needed for high‑end or unique properties.
- Regular retraining with new data is recommended to maintain accuracy.

## Output Files
- `best_svm_model.joblib` – serialized tuned pipeline.
- Prediction preview table displayed in notebook.
- Various figures illustrating model performance and feature effects.

## Usage
Execute `SVM.ipynb` sequentially. The final model is saved and can be loaded via `joblib.load("best_svm_model.joblib")`. All plots are generated inline.s