import numpy as np


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
