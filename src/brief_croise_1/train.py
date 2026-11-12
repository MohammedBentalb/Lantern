from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from brief_croise_1.config.conf import CROSS_VALIDATION, RANDOM_STATE, SCORING, TARGET_LABELS
from brief_croise_1.models import get_models_with_params
from brief_croise_1.preprocessing import get_preprocessing_pipeline


def train_models(x_train, y_train, scoring=SCORING, refit='f1', cv=CROSS_VALIDATION, random_state=RANDOM_STATE, bare_models = False, cross_val = True):
    models, params = get_models_with_params(random_state)
    grides = {}
    for name, model in models.items():
        if not bare_models:
            grides[name] = train_grid_models(x_train, y_train, model, params, name, scoring, refit, cv, random_state)
        elif cross_val:
            grides[name] = train_cross_val_bare_models(x_train, y_train, model, scoring, cv, random_state)
        else:
            grides[name] = train_bare_models(x_train, y_train, model, random_state)
    return grides

def build_pipline(model, random_state = RANDOM_STATE):
    return Pipeline([('preprocessing', get_preprocessing_pipeline()), ('smote', SMOTE(random_state=random_state, k_neighbors=4)), ('model', model)])

def train_grid_models(x_train, y_train, model, params, name, scoring, refit, cv, random_state = RANDOM_STATE):
    pipeline = build_pipline(model, random_state)
    grid = {f'model__{param}': values for param, values in params[name].items()}
    grid = GridSearchCV(pipeline, param_grid=grid, scoring=scoring, refit=refit, cv=StratifiedKFold(cv, shuffle=True, random_state=random_state), n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid

def train_cross_val_bare_models(x_train, y_train, model, scoring=SCORING, cv=CROSS_VALIDATION, random_state = RANDOM_STATE):
    pipeline = build_pipline(model, random_state).set_params(model__random_state=random_state)
    folds = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    scores = cross_validate(pipeline, x_train, y_train, scoring=scoring, cv=folds, n_jobs=-1)
    y_pred = cross_val_predict(pipeline, x_train, y_train, cv=folds, n_jobs=-1)
    row = {metric: scores[f'test_{metric}'].mean() for metric in scoring}
    row['cm'] = confusion_matrix(y_train, y_pred, labels=TARGET_LABELS)
    return row, scores

def train_bare_models(x_train, y_train, model, random_state = RANDOM_STATE, test_size = 0.2):
    x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=test_size, random_state=random_state)
    pipeline = build_pipline(model, random_state).set_params(model__random_state=random_state)
    pipeline.fit(x_fit, y_fit)
    y_pred = pipeline.predict(x_val)
    row = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro'),
        'recall': recall_score(y_val, y_pred, average='macro'),
        'f1': f1_score(y_val, y_pred, average='macro'),
        'cm': confusion_matrix(y_val, y_pred, labels=TARGET_LABELS),
    }
    return row, pipeline

def summarize_grids(grids: dict):
    rows = {}
    for name, g in grids.items():
        row = {metric: g.cv_results_[f'mean_test_{metric}'][g.best_index_] for metric in SCORING}
        row['best_params'] = g.best_params_
        row['pipe'] = g.best_estimator_
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient='index')
