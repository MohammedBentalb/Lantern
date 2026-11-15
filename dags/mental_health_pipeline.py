from airflow.decorators import dag
from brief_croise_1.config.conf import DAG_ID, DAG_START_DATE, DAG_TAGS
from tasks.eda import eda
from tasks.evaluate import evaluate
from tasks.load_data import load_data
from tasks.preprocess import preprocess
from tasks.train import train


@dag(dag_id=DAG_ID, start_date=DAG_START_DATE, schedule=None, catchup=False, tags=DAG_TAGS)
def mental_health_pipeline():
    load = load_data()
    eda_task = eda()
    preprocess_task = preprocess()
    train_task = train()
    evaluate_task = evaluate()

    load >> [eda_task, preprocess_task]
    preprocess_task >> train_task >> evaluate_task


mental_health_pipeline()
