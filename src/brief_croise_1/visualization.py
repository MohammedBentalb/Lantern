import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from brief_croise_1.config.conf import ARTIFACTS_DIR, TARGET_LABELS


def save_and_show(fig, name, show=True):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_numeric_distributions(df, show=True):
    columns = df.select_dtypes(include='number').columns
    fig, axes = plt.subplots(len(columns), 2, figsize=(12, 3 * len(columns)), squeeze=False)
    for col, ax in zip(columns, axes):
        ax[0].boxplot(df[col].dropna(), vert=False)
        ax[0].set_title(f"{col} boxplot")

        ax[1].hist(df[col].dropna())
        ax[1].set_title(f"{col} histogram")
    fig.tight_layout()
    return save_and_show(fig, "numeric_distributions", show)


def plot_categorical_vs_target(df, target="Mental_Health_Risk", show=True):
    cols = df.select_dtypes(include=["object", "string"]).columns.drop([target, 'Student_ID'])
    paths = []

    for c in cols:
        res = df.groupby([c, target]).size().unstack(fill_value=0)
        ax = res.plot(kind="bar", figsize=(7, 4), title=f"{c} vs {target}")
        paths.append(save_and_show(ax.get_figure(), f"{c}_vs_{target}", show))
    return paths


def plot_correlation_heatmap(df, show=True):
    columns = df.select_dtypes(include='number').columns
    corr = df[columns].dropna().corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    save_and_show(fig, "correlation_heatmap", show)
    return corr


def plot_encoded_correlation_heatmap(df, show=True):
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
    return save_and_show(fig, "encoded_correlation_heatmap", show)


def plot_confusion_matrix(cm, name, labels=TARGET_LABELS, show=True):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name}_confusion_matrix")
    return save_and_show(fig, f"{name}_confusion_matrix", show)
