import pandas as pd
from airflow.decorators import task
from brief_croise_1.config.conf import DATASETS, PROCESSED_DIR, X_TEST_PATH, X_TRAIN_PATH, Y_TEST_PATH, Y_TRAIN_PATH
from brief_croise_1.preprocessing import get_X_Y, preprocess_data, split_data_stractified


@task
def preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASETS['raw'])
    X, Y = get_X_Y(df)
    x_train, x_test, y_train, y_test = split_data_stractified(X, Y)
    x_train, y_train, x_test = preprocess_data(x_train, y_train, x_test)
    x_train.to_parquet(X_TRAIN_PATH)
    x_test.to_parquet(X_TEST_PATH)
    y_train.to_frame().to_parquet(Y_TRAIN_PATH)
    y_test.to_frame().to_parquet(Y_TEST_PATH)
 