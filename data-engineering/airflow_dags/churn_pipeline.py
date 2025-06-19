
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from data_engineering.generators.customer_data import generate_all_data
from data_engineering.loaders.load_to_snowflake import load_to_snowflake

def generate_data():
    generate_all_data(n_customers=1000, save_to_csv=True)

def load_data():
    load_to_snowflake()

with DAG('churn_pipeline',
         start_date=datetime(2023, 1, 1),
         schedule_interval='@daily',
         catchup=False) as dag:

    generate = PythonOperator(task_id='generate_data', python_callable=generate_data)
    load = PythonOperator(task_id='load_to_snowflake', python_callable=load_data)

    generate >> load
