import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

print("Запуск обучения")

np.random.seed(51)

train = pd.read_csv(
    "https://raw.githubusercontent.com/PersDep/data-mining-intro-2021/main/hw08-boosting-clustering-data/flight_delays_train.csv"  # noqa: Е501
)
test = pd.read_csv(
    "https://raw.githubusercontent.com/PersDep/data-mining-intro-2021/main/hw08-boosting-clustering-data/flight_delays_test.csv"  # noqa: Е501
)
print("Данные получены")


X_train = train[["Distance", "DepTime"]].values
y_train = train["dep_delayed_15min"].map({"Y": 1, "N": 0}).values
X_test = test[["Distance", "DepTime"]].values
X_train_part, X_valid, y_train_part, y_valid = train_test_split(
    X_train, y_train, test_size=0.3
)

print("Обучение модели")

xgb_model = XGBClassifier()
xgb_model.fit(X_train_part, y_train_part)

preds = xgb_model.predict(X_valid)
preds_probas = xgb_model.predict_proba(X_valid)[:, 1]
print("Результат обучения:")


print("Accuracy:", accuracy_score(y_valid, preds))
print("Precision:", precision_score(y_valid, preds))
print("Recall:", recall_score(y_valid, preds))
print("MSE:", mean_squared_error(y_valid, preds))
print("ROC-AUC:", roc_auc_score(y_valid, preds_probas))
