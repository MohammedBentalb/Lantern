from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from brief_croise_1.config.conf import NOMINAL_COLUMNS, NUMERIC_COLUMNS, ORDICAL_COLUMNS, RANDOM_STATE
from brief_croise_1.transformers import IQROutliersRemover, ZscoreOutliersRemover, SocialMediaScreenFixer


def get_X_Y(df: pd.DataFrame):
    X = df.drop(columns=['Student_ID', 'Mental_Health_Risk'])
    Y = df['Mental_Health_Risk']
    return X, Y

def split_data_stractified(X: pd.DataFrame, Y: pd.Series, random_state=RANDOM_STATE):
    return train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=random_state)

def preprocess_data(x_train, y_train, x_test, remove_outliers=False, outliers_method='iqr', handle_social_media=False, calibrate_data = False, calibration_trashold = 0.7):
    if handle_social_media:
        fixer = SocialMediaScreenFixer().fit(x_train)
        x_train, x_test = fixer.transform(x_train), fixer.transform(x_test)

    if remove_outliers:
        remover = IQROutliersRemover() if outliers_method == 'iqr' else ZscoreOutliersRemover()
        remover.fit(x_train, y_train)
        x_train, y_train = remover.transform(x_train, y_train)

    if calibrate_data:
        values = y_train.value_counts()
        ratio = values.min() / values.max()
        if ratio < calibration_trashold:
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=4)
            x_train, y_train = smote.fit_resample(x_train, y_train)

    return x_train, y_train, x_test

def get_preprocessing_pipeline():
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    ordinal = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[v  for c, v in ORDICAL_COLUMNS.items()])),
    ])
    nominal = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')),
    ])
    return ColumnTransformer([
        ('numeric', numeric, NUMERIC_COLUMNS),
        ('ordinal', ordinal, [c for c in ORDICAL_COLUMNS.keys()]),
        ('nominal', nominal, NOMINAL_COLUMNS),
    ])