import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df['label'].value_counts()
    ax.pie(counts, labels=['Legit', 'Fraud'], autopct='%1.1f%%',
           colors=['#2ecc71', '#e74c3c'], startangle=90)
    ax.set_title("Class Distribution")
    return fig


def plot_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, color, name in [(0, '#2ecc71', 'Legit'), (1, '#e74c3c', 'Fraud')]:
        subset = df[df['label'] == label]['transaction_amount']
        axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=name)
        axes[1].boxplot(subset, positions=[label], patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.6))
    axes[0].set_title("Transaction Amount Distribution")
    axes[0].set_xlabel("Amount")
    axes[0].legend()
    axes[1].set_title("Amount by Class")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Legit', 'Fraud'])
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include='number')
    # Keep top correlated features with label
    if 'label' in num_df.columns:
        top_cols = num_df.corr()['label'].abs().nlargest(12).index.tolist()
        num_df = num_df[top_cols]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_test, probs):
    fpr, tpr, _ = roc_curve(y_test, probs)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#3498db', lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_precision_recall(y_test, probs):
    precision, recall, _ = precision_recall_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color='#e67e22', lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_test, probs, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in indices], importances[indices], color='#3498db')
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    return fig


def plot_threshold_analysis(y_test, probs):
    thresholds = np.linspace(0.01, 0.99, 100)
    precisions, recalls, f1s, costs = [], [], [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        fn = ((preds == 0) & (y_test == 1)).sum()
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / (p + r + 1e-10))
        costs.append((fn * 5000 + fp * 200) / 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(thresholds, precisions, label='Precision', color='#3498db')
    ax1.plot(thresholds, recalls, label='Recall', color='#e74c3c')
    ax1.plot(thresholds, f1s, label='F1', color='#2ecc71', lw=2)
    ax1.set_xlabel("Threshold")
    ax1.set_title("Precision / Recall / F1 vs Threshold")
    ax1.legend()

    ax2.plot(thresholds, costs, color='#e67e22', lw=2)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Cost ($K)")
    ax2.set_title("Business Cost vs Threshold")

    fig.tight_layout()
    return fig
