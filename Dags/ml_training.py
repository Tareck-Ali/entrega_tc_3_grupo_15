from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from train_baseline import train
from convert_to_onnx import convert


with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train,
    )

    convert_model = PythonOperator(
        task_id="convert_to_onnx",
        python_callable=convert,
    )

    train_model >> convert_model
