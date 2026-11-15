from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KAGGLE_DATASET_PATH =  PROJECT_ROOT / 'data'/ 'raw'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'

RANDOM_STATE = 42
CROSS_VALIDATION = 5

SCORING = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro',
}

TARGET_LABELS = ['Low', 'Medium', 'High']

NUMERIC_COLUMNS = ['Age', 'Year_of_Study', 'Sleep_Hours', 'Study_Hours_Per_Week', 'GPA', 'Social_Media_Hours', 'Screen_Time_Hours', 'Financial_Stress', 'Family_Support', 'Attendance_Rate']
NOMINAL_COLUMNS = ['Gender', 'Part_Time_Job', 'Previous_Mental_Health_History']
ORDICAL_COLUMNS= {
    'Physical_Activity': ['Low', 'Moderate', 'High'],
    'Diet_Quality': ['Poor', 'Average', 'Good']
}


GENDER_OPTIONS = ['Male', 'Female', 'Non-binary']
YES_NO_OPTIONS = ['No', 'Yes']

DATASETS = {
    'raw' : PROJECT_ROOT / 'data'/ 'raw' / 'student_mental_health.csv'
}

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
MODEL_EXTENSION = '.joblib'
CHAMPION_DIR = MODELS_DIR / 'champion'
CHAMPION_MODEL_PATH = CHAMPION_DIR / f'champion{MODEL_EXTENSION}'

X_TRAIN_PATH = PROCESSED_DIR / 'x_train.parquet'
X_TEST_PATH = PROCESSED_DIR / 'x_test.parquet'
Y_TRAIN_PATH = PROCESSED_DIR / 'y_train.parquet'
Y_TEST_PATH = PROCESSED_DIR / 'y_test.parquet'

METRICS_DIR = ARTIFACTS_DIR / 'metrics'
BARE_SUMMARY_PATH = METRICS_DIR / 'bare_cv_summary.csv'
GS_SUMMARY_PATH = METRICS_DIR / 'gs_summary.csv'
TEST_REPORT_PATH = METRICS_DIR / 'test_report.csv'

DAG_ID = 'mental_health_pipeline'
DAG_START_DATE = datetime(2026, 1, 1)
