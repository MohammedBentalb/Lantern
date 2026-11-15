from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from brief_croise_1.config.conf import CROSS_VALIDATION, RANDOM_STATE, SCORING, TARGET_LABELS
from brief_croise_1.preprocessing import get_preprocessing_pipeline


def build_pipeline(model, random_state=RANDOM_STATE):
    return Pipeline([('preprocessing', get_preprocessing_pipeline()), ('smote', SMOTE(random_state=random_state, k_neighbors=4)), ('model', model)])

def train_grid_search(x_train, y_train, model, param_grid, scoring=SCORING, refit='f1', cv=CROSS_VALIDATION, random_state=RANDOM_STATE):
    pipeline = build_pipeline(model, random_state)
    grid = {f'model__{param}': values for param, values in param_grid.items()}
    search = GridSearchCV(pipeline, param_grid=grid, scoring=scoring, refit=refit, cv=StratifiedKFold(cv, shuffle=True, random_state=random_state), n_jobs=-1)
    search.fit(x_train, y_train)
    return search

def train_bare_model(x_train, y_train, model, random_state=RANDOM_STATE, test_size=0.2):
    x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=test_size, random_state=random_state)
    pipeline = build_pipeline(model, random_state)
    pipeline.fit(x_fit, y_fit)
    y_pred = pipeline.predict(x_val)
    row = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, average='macro'),
        'recall': recall_score(y_val, y_pred, average='macro'),
        'f1': f1_score(y_val, y_pred, average='macro'),
        'cm': confusion_matrix(y_val, y_pred, labels=TARGET_LABELS),
    }
    return {**row, "pipe" : pipeline}

def train_cross_val_bare_model(x_train, y_train, model, scoring=SCORING, cv=CROSS_VALIDATION, random_state=RANDOM_STATE):
    pipeline = build_pipeline(model, random_state)
    folds = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    scores = cross_validate(pipeline, x_train, y_train, scoring=scoring, cv=folds, n_jobs=-1)
    y_pred = cross_val_predict(pipeline, x_train, y_train, cv=folds, n_jobs=-1)
    row = {metric: scores[f'test_{metric}'].mean() for metric in scoring}
    row['cm'] = confusion_matrix(y_train, y_pred, labels=TARGET_LABELS)
    return row, scores

def summarize_grids(grids: dict):
    rows = {}
    for name, g in grids.items():
        row = {metric: g.cv_results_[f'mean_test_{metric}'][g.best_index_] for metric in SCORING}
        row['best_params'] = g.best_params_
        row['pipe'] = g.best_estimator_
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient='index')
