from clearml import Task
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

from utils import get_data, TRAIN_FILENAME, TEST_FILENAME


def train_and_validate(train, test, task):
    print("Запуск обучения")

    np.random.seed(51)
    X_train = train[["Distance", "DepTime"]].values
    y_train = train["dep_delayed_15min"].map({"Y": 1, "N": 0}).values
    X_test = test[["Distance", "DepTime"]].values
    X_train_part, X_valid, y_train_part, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=51
    )

    print("Обучение модели")

    catboost_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        random_seed=51,
        verbose=10  
    )
    
    catboost_model.fit(
        X_train_part, y_train_part,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    preds = catboost_model.predict(X_valid)
    preds_probas = catboost_model.predict_proba(X_valid)[:, 1]
    print("Результат обучения:")

    metrics = {
        "Accuracy": accuracy_score(y_valid, preds),
        "Precision": precision_score(y_valid, preds),
        "Recall": recall_score(y_valid, preds),
        "MSE": mean_squared_error(y_valid, preds),
        "ROC-AUC": roc_auc_score(y_valid, preds_probas)
    }

    logger = task.get_logger()
    for metric_name, metric_value in metrics.items():
        logger.report_scalar(title="Metrics", series=metric_name, value=metric_value, iteration=0)
        print(f"{metric_name}: {metric_value}")

    return catboost_model, metrics

def save_model(model, save_path):
    model_path = save_path + "catboost_model.cbm"
    # Сохранение модели в формате CatBoost
    model.save_model(model_path)

def main():
    DATA_PATH = "./flights_data/"
    SAVE_PATH = "./flights_data/"
    task = Task.init(project_name='MLOps project', task_name='catboost task)

    train = pd.read_csv(DATA_PATH + TRAIN_FILENAME)
    test = pd.read_csv(DATA_PATH + TEST_FILENAME)
    
    # логирование датасетов
    task.upload_artifact(name="train_data", artifact_object=DATA_PATH + "flight_delays_train.csv")
    task.upload_artifact(name="test_data", artifact_object=DATA_PATH + "flight_delays_test.csv")
    
    model, metrics = train_and_validate(train, test, task)
    
    # логирование метрик как артефактов
    task.upload_artifact(name="metrics", artifact_object=metrics)    
    # модель не логируем, clearml сам сохраняет  
    save_model(model, SAVE_PATH)


if __name__ == "__main__":
    main()
