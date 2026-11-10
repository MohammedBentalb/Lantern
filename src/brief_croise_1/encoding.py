import pandas as pd


def encode_target(df):
    return df['Mental_Health_Risk'].map({'Low': 0, 'Medium': 1, 'High': 2})


def encode_features(df):
    copy = df.copy()
    copy = pd.get_dummies(copy, columns=['Gender'], dtype=int)
    copy['Physical_Activity'] = copy['Physical_Activity'].map({'Low': 0, "Moderate": 1, "High": 2})
    copy["Part_Time_Job"] = copy['Part_Time_Job'].map({'Yes': 0, "No": 1})
    copy['Diet_Quality'] = copy['Diet_Quality'].map({'Poor': 0, "Good": 2, "Average": 1})
    copy["Previous_Mental_Health_History"]= copy['Previous_Mental_Health_History'].map({'Yes': 0, "No": 1})

    return copy.drop(columns=['Student_ID', 'Mental_Health_Risk'])
