import joblib
import pandas as pd
from airflow.decorators import task
from brief_croise_1.config.conf import BARE_SUMMARY_PATH, GS_SUMMARY_PATH, METRICS_DIR, MODEL_EXTENSION, MODELS_DIR, X_TRAIN_PATH, Y_TRAIN_PATH
from brief_croise_1.models import get_models_with_params
from brief_croise_1.preprocessing import read_features
from brief_croise_1.train import fit_bare_model, get_best_bare_name, summarize_grids, train_cross_val_bare_model, train_grid_search
from brief_croise_1.visualization import plot_confusion_matrix


@task
def bare_train():
    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    x_train = read_features(X_TRAIN_PATH)
    y_train = pd.read_parquet(Y_TRAIN_PATH).squeeze('columns')
    
    models, _ = get_models_with_params()
    rows = {}
    for name, model in models.items():
        row, _ = train_cross_val_bare_model(x_train, y_train, model)
        plot_confusion_matrix(row['cm'], f'{name}_bare_cv', show=False)
        rows[name] = row
    
    summary = pd.DataFrame.from_dict(rows, orient='index').drop(columns='cm')
    summary.to_csv(BARE_SUMMARY_PATH)
    best_name = get_best_bare_name()
    joblib.dump(fit_bare_model(x_train, y_train, models[best_name]), MODELS_DIR / f'{best_name}_bare{MODEL_EXTENSION}')

@task
def tune_model():
    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    best_name = get_best_bare_name()
    x_train = read_features(X_TRAIN_PATH)
    y_train = pd.read_parquet(Y_TRAIN_PATH).squeeze('columns')
    
    models, params = get_models_with_params()
    grid = train_grid_search(x_train, y_train, models[best_name], params[best_name])
    summary = summarize_grids({best_name: grid})
    joblib.dump(grid.best_estimator_, MODELS_DIR / f'{best_name}_tuned{MODEL_EXTENSION}')
    summary.drop(columns='pipe').to_csv(GS_SUMMARY_PATH)