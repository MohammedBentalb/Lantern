import shutil
import joblib
import pandas as pd
from airflow.decorators import task
from sklearn.metrics import classification_report, confusion_matrix
from brief_croise_1.config.conf import CHAMPION_DIR, CHAMPION_MODEL_PATH, MODEL_EXTENSION, MODELS_DIR, TARGET_LABELS, TEST_REPORT_PATH, X_TEST_PATH, Y_TEST_PATH
from brief_croise_1.preprocessing import read_features
from brief_croise_1.visualization import plot_confusion_matrix


@task
def evaluate():
    x_test = read_features(X_TEST_PATH)
    y_test = pd.read_parquet(Y_TEST_PATH).squeeze('columns')
    reports = {}
    for path in sorted(MODELS_DIR.glob(f'*{MODEL_EXTENSION}')):
        pipe = joblib.load(path)
        y_pred = pipe.predict(x_test)
        cm = confusion_matrix(y_test, y_pred, labels=TARGET_LABELS)
        plot_confusion_matrix(cm, path.stem, show=False)
        report = classification_report(y_test, y_pred, output_dict=True)
        reports[path.stem] = {
            'accuracy': report['accuracy'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1': report['macro avg']['f1-score'],
        }
    report_df = pd.DataFrame(reports).T
    report_df.to_csv(TEST_REPORT_PATH)

    champion = report_df['f1'].idxmax()
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODELS_DIR / f'{champion}{MODEL_EXTENSION}', CHAMPION_MODEL_PATH)
