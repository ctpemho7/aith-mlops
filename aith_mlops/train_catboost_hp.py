from clearml import Task
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, mean_squared_error,
                            precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, cv
import optuna
from optuna.samplers import TPESampler
from functools import partial
import json

from utils import get_data, TRAIN_FILENAME, TEST_FILENAME


def optimize_hyperparameters(X, y, task, max_evals=20):
    print("Начало оптимизации гиперпараметров с Optuna")

    # train pool для CatBoost
    train_pool = Pool(X, y)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0.1, 1),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 254]),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 10),
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'early_stopping_rounds': 50,
            'verbose': False,
            'random_seed': 51
        }

        # Кросс-валидация
        cv_data = cv(
            pool=train_pool,
            params=params,
            fold_count=5,
            shuffle=True,
            partition_random_seed=51,
            stratified=True,
            verbose=False
        )

        # Лучшее значение AUC
        best_auc = np.max(cv_data['test-AUC-mean'])

        # Логирование в ClearML
        logger = task.get_logger()
        logger.report_scalar(
            title="Optuna", 
            series="AUC", 
            value=best_auc, 
            iteration=trial.number
        )

        return best_auc

    # Настройка Optuna
    sampler = TPESampler(seed=51)  # Детерминированный сэмплер
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)  # Ранняя остановка
    )
    study.optimize(objective, n_trials=max_evals)

    print("Лучшие параметры:", study.best_params)
    return study.best_params


def train_and_validate(train, test, task):
    print("Запуск обучения")

    np.random.seed(51)
    X_train = train[["Distance", "DepTime"]].values
    y_train = train["dep_delayed_15min"].map({"Y": 1, "N": 0}).values
    X_test = test[["Distance", "DepTime"]].values
    
    X_train_part, X_valid, y_train_part, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=51, stratify=y_train
    )

    # Подбор гиперпараметров
    best_params = optimize_hyperparameters(X_train_part, y_train_part, task)

    print("Обучение модели с лучшими параметрами")
    
    # Инициализация модели
    model = CatBoostClassifier(
        **best_params,
        eval_metric='AUC',
        early_stopping_rounds=50,
        random_seed=51,
        thread_count=-1,
        verbose=50
    )
    
    # Обучение с валидацией
    model.fit(
        X_train_part, y_train_part,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        plot=False
    )

    # Предсказания и метрики
    preds = model.predict(X_valid)
    preds_probas = model.predict_proba(X_valid)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y_valid, preds),
        "Precision": precision_score(y_valid, preds),
        "Recall": recall_score(y_valid, preds),
        "MSE": mean_squared_error(y_valid, preds),
        "ROC-AUC": roc_auc_score(y_valid, preds_probas)
    }

    # Логирование метрик в ClearML
    logger = task.get_logger()
    for metric_name, metric_value in metrics.items():
        logger.report_scalar(title="Metrics", series=metric_name, value=metric_value, iteration=0)
        print(f"{metric_name}: {metric_value}")
    
    # Сохранение параметров
    task.connect(best_params, name="Best Parameters")
    return model, metrics


def save_model(model, save_path):
    model_path = save_path + "catboost_model_opt.cbm"
    model.save_model(model_path)


def main():
    DATA_PATH = "./flights_data/"
    SAVE_PATH = "./flights_data/"
    
    task = Task.init(
        project_name='MLOps project', 
        task_name='catboost optimized (Optuna)',
        tags=['catboost', 'optimized', 'optuna']
    )
    
    train = pd.read_csv(DATA_PATH + TRAIN_FILENAME)
    test = pd.read_csv(DATA_PATH + TEST_FILENAME)
    
    # Логирование данных
    task.upload_artifact(
        name="train_data", 
        artifact_object=DATA_PATH + "flight_delays_train.csv",
        metadata={
            "samples": len(train),
            "features": train.shape[1],
            "class_distribution": train["dep_delayed_15min"].value_counts().to_dict()
        }
    )
    task.upload_artifact(
        name="test_data", 
        artifact_object=DATA_PATH + "flight_delays_test.csv",
        metadata={
            "samples": len(test),
            "features": test.shape[1]
        }
    )
    
    model, metrics = train_and_validate(train, test, task)
    
    # Логирование метрик
    task.upload_artifact(name="metrics", artifact_object=metrics)
    
    save_model(model, SAVE_PATH)
    

if __name__ == "__main__":
    main()