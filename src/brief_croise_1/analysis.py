import numpy as np
from IPython.display import display


def detect_anomalies(df):
    print('--------------- detecting the anomalies ------------------')
    print('social media scren time is bigger than the spent time on screen') #37
    display((df['Social_Media_Hours'] > df['Screen_Time_Hours']).sum())
    print()
    print()
    print('---------- gpa that exceeds the limit --------------')
    display((df['GPA'] > 5).sum())
    print()
    print()
    print('---------------- sleep hours and screen time both exceed 24h -------------------')
    display((df['Sleep_Hours'] + df['Screen_Time_Hours'] > 24).sum())
    print()
    print()
    print('---------------- age and years of study do not match ------------------')
    min_age = 17
    res = df['Age'] < min_age + (df['Year_of_Study'] - 1).clip(lower=0)
    display(res.sum().sum())


def get_outliers_report_Zscore(df, cols = None, with_mask = False):
    result = {}
    global_mask = None

    if cols is None:
        cols = df.select_dtypes(include='number').columns

    for c in cols:
        ZSCORE = (df[c] - df[c].mean()) / df[c].std()
        mask = np.abs(ZSCORE) > 3
        result[c] = {
            "number_of_outliers" : mask.sum(),
            "percentage_of_outliers" : mask.sum() * 100 / df.shape[0],
            "mean" : df[c].mean(),
            "median" : df[c].median()
        }

        if global_mask is None:
            global_mask = mask
        else:
            global_mask |= mask
    if with_mask:
        return result, global_mask
    return result

def get_outliers_report_IQR(df, cols = None, with_mask = False):
    if cols is None:
        cols = df.select_dtypes(include='number').columns

    result = {}
    global_mask = None
    for c in cols:
        Q1, Q3 = df[c].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        MAX_EDGE = Q3 + 1.5*IQR
        MIN_EDGE = Q1 - 1.5*IQR
        mask = (df[c] < MIN_EDGE) | (df[c] > MAX_EDGE)
        result[c] = {
            "number_of_outliers": mask.sum(),
            "percentage_of_outliers": mask.sum() * 100 / df.shape[0],
             "min_edge": MIN_EDGE,
             "max_edge": MAX_EDGE,
        }
        if global_mask is None:
            global_mask = mask
        else:
            global_mask |= mask
    if with_mask:
        return result, global_mask
    return result
