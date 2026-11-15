import joblib
import pandas as pd
from airflow.decorators import task
from brief_croise_1.config.conf import ARTIFACTS_DIR, CV_SUMMARY_PATH, MODEL_EXTENSION, MODELS_DIR, X_TRAIN_PATH, Y_TRAIN_PATH
from brief_croise_1.models import get_models_with_params
from brief_croise_1.preprocessing import read_features
from brief_croise_1.train import summarize_grids, train_grid_search


@task
def train():
    MODELS_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    x_train = read_features(X_TRAIN_PATH)
    y_train = pd.read_parquet(Y_TRAIN_PATH).squeeze('columns')
    models, params = get_models_with_params()
    grids = {name: train_grid_search(x_train, y_train, model, params[name]) for name, model in models.items()}
    summary = summarize_grids(grids)
    for name, row in summary.iterrows():
        joblib.dump(row['pipe'], MODELS_DIR / f'{name}{MODEL_EXTENSION}')
    summary.drop(columns='pipe').to_csv(CV_SUMMARY_PATH)
