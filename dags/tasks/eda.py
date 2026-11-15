import pandas as pd
from airflow.decorators import task
from brief_croise_1.config.conf import DATASETS
from brief_croise_1.visualization import plot_categorical_vs_target, plot_correlation_heatmap, plot_numeric_distributions


@task
def eda():
    df = pd.read_csv(DATASETS['raw'])
    plot_numeric_distributions(df, show=False)
    plot_categorical_vs_target(df, show=False)
    plot_correlation_heatmap(df, show=False)
