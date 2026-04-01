import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import PLOT_DIR

os.makedirs(PLOT_DIR, exist_ok=True)

def save_plot(name):
    plt.savefig(os.path.join(PLOT_DIR, f"{name}.png"), bbox_inches='tight')
    plt.close()

def plot_all(df):

    # Histogram
    if 'transaction_amount' in df.columns:
        sns.histplot(df['transaction_amount'], kde=True)
        save_plot("hist_amount")

    # Box
    if 'transaction_amount' in df.columns:
        sns.boxplot(x='label', y='transaction_amount', data=df)
        save_plot("box_amount")

    # Violin
    if 'transaction_amount' in df.columns:
        sns.violinplot(x='label', y='transaction_amount', data=df)
        save_plot("violin_amount")

    # Temporal (SAFE)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        if df['timestamp'].notna().sum() > 0:
            temp = df.groupby(df['timestamp'].dt.date)['label'].mean()
            plt.plot(temp)
            save_plot("temporal")

    # Pie
    if 'label' in df.columns:
        df['label'].value_counts().plot.pie(autopct='%1.1f%%')
        save_plot("pie")

    # Heatmap
    corr = df.select_dtypes(include=['number']).corr()
    sns.heatmap(corr)
    save_plot("heatmap")