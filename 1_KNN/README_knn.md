# KNN House Price Model

This folder contains the notebook-based KNN regression workflow for predicting `Tran_Pri`.

## Main Files

- `knn.ipynb`: full notebook with data loading, cleaning, feature engineering, model tuning, evaluation, and plots
- `TRAIN.csv`: training dataset
- `TEST.csv`: external test dataset
- `knn_price_model.joblib`: saved trained KNN pipeline after the notebook is run
- `new_test_with_predictions.csv`: test file with appended predictions after the notebook is run

## How to Run the Model

1. Open a terminal in `d:\MACHINE LEARNING\knn`.
2. Install the required packages if needed:

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

3. Open the notebook:

```powershell
jupyter lab
```

You can also open `knn.ipynb` directly in VS Code and use `Run All`.

4. Run the notebook from top to bottom in order.

The cells depend on variables created earlier, so later cells should not be run first.

## What the Notebook Does

- Loads `TRAIN.csv`
- Explores structure, summary statistics, missing values, and target distribution
- Parses `Tran_Dt` into datetime format
- Sorts the data chronologically
- Creates a time-based split:
  earliest 85% for model training and tuning
  latest 15% for holdout evaluation
- Applies feature engineering
- Builds a preprocessing + KNN pipeline
- Tunes hyperparameters with `GridSearchCV` and `TimeSeriesSplit`
- Evaluates on the holdout set
- Loads `TEST.csv`, predicts prices, and computes final metrics if `Tran_Pri` is present
- Saves the trained model and test predictions

## Additional Cleaning and Preprocessing Undertaken

- Converted `Tran_Dt` from text to datetime so the split follows the true transaction order.
- Sorted records by date before splitting to reduce time leakage from future observations.
- Created `Build_Age` from transaction year and `Build_Yr_Filled`.
- Converted `Prop_Flo` from text labels into an ordinal numeric feature.
- Extracted room, living room, kitchen, and bathroom counts from the `Layout` text.
- Dropped several raw columns after engineering features to keep the KNN feature set smaller and more focused.
- Imputed missing numeric values with the median using `SimpleImputer`.
- Imputed missing categorical values with a constant `"missing"` label.
- Standardized numeric features with `StandardScaler` because KNN is distance-based and sensitive to scale.
- One-hot encoded categorical variables so they could be used by the KNN regressor.

## Outputs After Running

- `knn_price_model.joblib`
- `new_test_with_predictions.csv`
- printed holdout and test metrics
- comparison and diagnostic plots shown inside the notebook
