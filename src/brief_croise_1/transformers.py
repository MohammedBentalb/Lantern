import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class IQROutliersRemover(BaseEstimator):
    def __init__(self, cols : None | list = None):
        self.cols = cols

    def fit(self, x: pd.DataFrame, y: pd.Series):
        cols = list(x.select_dtypes(include='number').columns) if self.cols is None else list(self.cols)

        Q1, Q3 = x[cols].quantile(0.25), x[cols].quantile(0.75)
        IQR = Q3 - Q1

        self.cols = cols
        self._min_edge = Q1  - 1.5*IQR
        self._max_edge = Q3  + 1.5*IQR
        return self

    def transform(self, x, y):
        if(not self.cols):
            raise Exception('fit must be called befor transforming')

        outliers= (x[self.cols] > self._max_edge) | (x[self.cols] < self._min_edge)
        retain = ~outliers.any(axis=1)
        return x[retain], y[retain]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x, y)

class ZscoreOutliersRemover(BaseEstimator):
    def __init__(self, cols : None | list = None, threshold : float = 3):
        self.cols = cols
        self.threshold = threshold

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        cols = list(x.select_dtypes(include='number').columns) if self.cols is None else list(self.cols)

        self._cols = cols
        self._mean = x[cols].mean()
        self._std = x[cols].std()
        return self

    def transform(self, x, y):
        if(not self.cols):
            raise Exception('fit must be called befor transforming')
        
        zscore = (x[self._cols] - self._mean) / self._std
        outliers = zscore.abs() > self.threshold
        retain = ~outliers.any(axis=1)
        return x[retain], y[retain]
    
    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform

class SocialMediaScreenFixer(BaseEstimator, TransformerMixin):
    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, x: pd.DataFrame):
        x = x.copy()
        flipped = x['Social_Media_Hours'] > x['Screen_Time_Hours']
        x.loc[flipped, ['Social_Media_Hours', 'Screen_Time_Hours']] = x.loc[flipped, ['Screen_Time_Hours', 'Social_Media_Hours']].values
        return x

    def fit_transform(self, X):
        return self.transform(X)
