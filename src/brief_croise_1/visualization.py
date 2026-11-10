import matplotlib.pyplot as plt


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
