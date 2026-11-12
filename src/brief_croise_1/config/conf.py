from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KAGGLE_DATASET_PATH =  PROJECT_ROOT / 'data'/ 'raw'

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


DATASETS = {
    'raw' : PROJECT_ROOT / 'data'/ 'raw' / 'student_mental_health.csv'
}