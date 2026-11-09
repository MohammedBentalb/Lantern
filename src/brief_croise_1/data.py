import kagglehub
from brief_croise_1.config.conf import DATASETS, KAGGLE_DATASET_PATH


def download_kaggle_dataset(dataset : str = "maqdadali/mental-health-risk-prediction-dataset-students"):
    if not DATASETS['raw'].exists():
        KAGGLE_DATASET_PATH.mkdir(parents=True, exist_ok=True)
        path = kagglehub.dataset_download(dataset, output_dir=str(KAGGLE_DATASET_PATH), force_download=True)
        print("Path to dataset files:", path)
        return path
    return KAGGLE_DATASET_PATH
