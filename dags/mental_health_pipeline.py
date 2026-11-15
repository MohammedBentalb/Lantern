from airflow.decorators import dag
from brief_croise_1.config.conf import DAG_ID, DAG_START_DATE
from tasks.evaluate import evaluate
from tasks.preprocess import preprocess
from tasks.load_data import load_data
from tasks.eda import eda
from tasks.train import bare_train, tune_model


@dag(dag_id=DAG_ID, start_date=DAG_START_DATE, schedule=None, catchup=False)
def mental_health_pipeline():
    load_data() >> preprocess() >> bare_train() >> tune_model() >> evaluate()


mental_health_pipeline()
