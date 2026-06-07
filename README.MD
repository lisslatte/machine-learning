## Project Structure

## Models Overview

| Model               | Type            | File / Folder                           |
|---------------------|-----------------|-----------------------------------------|
| K‑Nearest Neighbors | Non‑linear      | `Models/1_KNN`                          |
| CatBoost            | Tree‑based      | `Models/2_Catboost`                     |
| Random Forest       | Tree‑based      | `Models/3_Random_Forest`                |
| Decision Tree       | Tree‑based      | `Models/4_Decision_Tree`                |
| SVM                 | Margin‑based    | `Models/5_SVM`                          |
| Neural Network      | Deep learning   | `Models/6_Neural_Network`               |
| Ridge regression    | Linear (L2)     | `EXTRA MODELS/ridge_linear.ipynb`       |
| Lasso regression    | Linear (L1)     | `EXTRA MODELS/lasso_linear.ipynb`       |
| Elastic Net         | Linear (L1+L2)  | `EXTRA MODELS/elastic_net_linear.ipynb` |
| Simple NN v1/v2     | Lightweight NN  | `EXTRA MODELS/simple_NN_1`, `_2`        |
|---------------------|-----------------|-----------------------------------------|

DATA USAGE
| ALL EXCEPT `Models/6_Neural_Network`  | `data/TRAIN.csv & TEST.csv`           | 
| `Models/6_Neural_Network`             | `Models/6_Neural_Network/data_train_cleaned.csv + data_test_cleaned.csv`           | 