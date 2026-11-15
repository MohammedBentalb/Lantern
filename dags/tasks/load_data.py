from airflow.decorators import task
from brief_croise_1.config.conf import DATASETS
from brief_croise_1.data import download_kaggle_dataset


@task
def load_data():
    download_kaggle_dataset()
