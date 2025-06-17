from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from utils import TEST_FILENAME, TRAIN_FILENAME, get_data, transform_data, train_model, test_model, write_data


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='ML Pipeline: обработка данных, обучение, тестирование',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

def process_data(**kwargs):
    """Обработка данных"""
    train = get_data(TRAIN_FILENAME)
    test = get_data(TEST_FILENAME)
    X_train, X_test = transform_data(train, test)

    # поместим обработанные данные
    kwargs['ti'].xcom_push(key='train_data', value=X_train.to_json())
    kwargs['ti'].xcom_push(key='test_data', value=X_test.to_json())

    

def train_model(**kwargs):
    """Обучение модели"""
    # читаем данные из XCom
    ti = kwargs['ti']
    train_json = ti.xcom_pull(task_ids='process_data_task', key='train_data')
    test_json = ti.xcom_pull(task_ids='process_data_task', key='test_data')

    # обучим
    train = pd.read_json(train_json)
    test = pd.read_json(test_json)
    xgb_model, X_valid, y_valid = train_model(train, test)
    # достанем все веса
    model_config = xgb_model.get_booster().save_config()
    
    # поместим обработанные данные
    kwargs['ti'].xcom_push(key='model_config', value=model_config)
    kwargs['ti'].xcom_push(key='x_valid', value=X_valid.to_json())
    kwargs['ti'].xcom_push(key='y_valid', value=y_valid.to_json())

def test_model():
    """Тестирование модели"""
    # читаем данные из XCom
    ti = kwargs['ti']
    model_config = ti.xcom_pull(task_ids='train_model_task', key='model_config')
    X_valid = ti.xcom_pull(task_ids='train_model_task', key='x_valid')
    y_valid = ti.xcom_pull(task_ids='train_model_task', key='y_valid')

    test_model(xgb_model, X_valid, y_valid)
    write_data(xgb_model, 
        "xgb_model.json", 
        "model trained", 
        "json")
    
    print("Тестирование модели завершено")


# определение задач
process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

test_model_task = PythonOperator(
    task_id='test_model',
    python_callable=test_model,
    dag=dag,
)

process_data_task >> train_model_task >> test_model_task
