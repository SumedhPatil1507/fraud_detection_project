import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PLOT_DIR

os.makedirs(PLOT_DIR, exist_ok=True)

def save_plot(name):
    plt.savefig(os.path.join(PLOT_DIR, f"{name}.png"), bbox_inches='tight')
    plt.close()

def plot_all(df):

    if 'transaction_amount' in df.columns:
        sns.histplot(df['transaction_amount'], kde=True)
        save_plot("hist_amount")

        sns.boxplot(x='label', y='transaction_amount', data=df)
        save_plot("box_amount")

        sns.violinplot(x='label', y='transaction_amount', data=df)
        save_plot("violin_amount")

    if 'timestamp' in df.columns:
        temp = df.groupby(df['timestamp'].dt.date)['label'].mean()
        plt.plot(temp)
        save_plot("line_temporal")

    if 'label' in df.columns:
        df['label'].value_counts().plot.pie(autopct='%1.1f%%')
        save_plot("pie")

    if {'transaction_amount','distance_from_home_km'}.issubset(df.columns):
        sns.scatterplot(
            x='transaction_amount',
            y='distance_from_home_km',
            hue='label',
            data=df
        )
        plt.xscale('log'); plt.yscale('log')
        save_plot("scatter")

    corr = df.select_dtypes(include=['number']).corr()
    sns.heatmap(corr)
    save_plot("heatmap")