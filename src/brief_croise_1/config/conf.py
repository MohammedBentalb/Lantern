from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KAGGLE_DATASET_PATH =  PROJECT_ROOT / 'data'/ 'raw'

DATASETS = {
        'raw' : PROJECT_ROOT / 'data'/ 'raw' / 'student_mental_health.csv'
    }