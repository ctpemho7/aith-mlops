import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier


def train_and_validate(train, test):
    print("Запуск обучения")

    np.random.seed(51)
    print(train)
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
    return xgb_model

def save_model(model, save_path):
    model.save_model(save_path + "xgb_model.json")

def main():
    DATA_PATH = "./flights_data/"
    SAVE_PATH = "./flights_data/"

    train = pd.read_csv(DATA_PATH + "flight_delays_train.csv")
    test = pd.read_csv(DATA_PATH + "flight_delays_test.csv")
    model = train_and_validate(train, test)
    save_model(model, SAVE_PATH)


if __name__ == "__main__":
    main()
