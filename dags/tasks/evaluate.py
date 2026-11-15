import shutil
import joblib
import pandas as pd
from airflow.decorators import task
from sklearn.metrics import classification_report, confusion_matrix
from brief_croise_1.config.conf import BARE_SUMMARY_PATH, CHAMPION_DIR, CHAMPION_MODEL_PATH, GS_SUMMARY_PATH, METRICS_DIR, MODEL_EXTENSION, MODELS_DIR, TARGET_LABELS, TEST_REPORT_PATH, X_TEST_PATH, Y_TEST_PATH
from brief_croise_1.preprocessing import read_features
from brief_croise_1.train import get_best_bare_name
from brief_croise_1.visualization import plot_confusion_matrix


@task
def evaluate():
    best_name = get_best_bare_name()
    x_test = read_features(X_TEST_PATH)
    y_test = pd.read_parquet(Y_TEST_PATH).squeeze('columns')
    cv_f1 = {
        f'{best_name}_tuned': pd.read_csv(GS_SUMMARY_PATH, index_col=0).loc[best_name, 'f1'],
        f'{best_name}_bare': pd.read_csv(BARE_SUMMARY_PATH, index_col=0).loc[best_name, 'f1'],
    }
    rows = {}
    for stem in cv_f1:
        pipe = joblib.load(MODELS_DIR / f'{stem}{MODEL_EXTENSION}')
        y_pred = pipe.predict(x_test)
        plot_confusion_matrix(confusion_matrix(y_test, y_pred, labels=TARGET_LABELS), f'{stem}_test', show=False)
        report = classification_report(y_test, y_pred, labels=TARGET_LABELS, output_dict=True)
        rows[stem] = {
            'accuracy': report['accuracy'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1': report['macro avg']['f1-score'],
            **{f'recall_{label}': report[label]['recall'] for label in TARGET_LABELS},
        }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).T.to_csv(TEST_REPORT_PATH)
    champion = max(cv_f1, key=cv_f1.get)
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODELS_DIR / f'{champion}{MODEL_EXTENSION}', CHAMPION_MODEL_PATH)
