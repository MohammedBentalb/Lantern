import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def plot_numeric_distributions(df):
    columns = df.select_dtypes(include='number').columns
    fig, axes = plt.subplots(len(columns), 2, figsize=(12, 3 * len(columns)), squeeze=False)
    for col, ax in zip(columns, axes):
        ax[0].boxplot(df[col].dropna(), vert=False)
        ax[0].set_title(f"{col} boxplot")

        ax[1].hist(df[col].dropna())
        ax[1].set_title(f"{col} histogram")
    fig.tight_layout()
    plt.show()


def plot_categorical_vs_target(df, target="Mental_Health_Risk"):
    cols = df.select_dtypes(include="str").columns.drop([target, 'Student_ID'])

    for c in cols:
        res = df.groupby([c, target]).size().unstack(fill_value=0)
        display(res)
        res.plot(kind="bar", figsize=(7, 4), title=f"{c} vs {target}")
        plt.show()


def plot_correlation_heatmap(df):
    columns = df.select_dtypes(include='number').columns
    corr = df[columns].dropna().corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    return corr


def plot_encoded_correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)