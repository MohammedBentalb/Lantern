from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from brief_croise_1.config.conf import RANDOM_STATE


def get_models_with_params(random_state=RANDOM_STATE):
    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForestClassifier': RandomForestClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
    }
    params = {
        'LogisticRegression': {'C': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [2000], 'class_weight': [None, 'balanced'], 'random_state': [random_state]},
        'RandomForestClassifier': {'n_estimators': [100, 300, 500], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2, 3, 5], 'max_features': ['sqrt', 0.5], 'random_state': [random_state]},
        'DecisionTreeClassifier': {'max_depth': [3, 5, 7, 10, 15, None], 'min_samples_leaf': [1, 5, 10, 20, 50], 'criterion': ['gini', 'entropy'], 'random_state': [random_state]},
    }
    return models, params
