"""All interactive plots using Plotly."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import auc


def plot_class_distribution(df):
    counts = df['label'].value_counts().reset_index()
    counts.columns = ['Class', 'Count']
    counts['Class'] = counts['Class'].map({0: 'Legit', 1: 'Fraud'})
    fig = px.pie(counts, values='Count', names='Class',
                 color='Class', color_discrete_map={'Legit': '#2ecc71', 'Fraud': '#e74c3c'},
                 title='Class Distribution', hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def plot_amount_distribution(df):
    fig = go.Figure()
    for label, color, name in [(0, '#2ecc71', 'Legit'), (1, '#e74c3c', 'Fraud')]:
        subset = df[df['label'] == label]['transaction_amount']
        fig.add_trace(go.Histogram(x=subset, name=name, opacity=0.7,
                                   marker_color=color, nbinsx=60))
    fig.update_layout(barmode='overlay', title='Transaction Amount Distribution',
                      xaxis_title='Amount ($)', yaxis_title='Count',
                      legend_title='Class')
    return fig


def plot_amount_box(df):
    df2 = df.copy()
    df2['Class'] = df2['label'].map({0: 'Legit', 1: 'Fraud'})
    fig = px.box(df2, x='Class', y='transaction_amount', color='Class',
                 color_discrete_map={'Legit': '#2ecc71', 'Fraud': '#e74c3c'},
                 title='Amount Distribution by Class', points='outliers')
    return fig


def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include='number')
    if 'label' in num_df.columns:
        top_cols = num_df.corr()['label'].abs().nlargest(14).index.tolist()
        num_df = num_df[top_cols]
    corr = num_df.corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                    title='Feature Correlation Heatmap', aspect='auto',
                    zmin=-1, zmax=1)
    fig.update_layout(height=550)
    return fig


def plot_roc_curve(y_test, probs):
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.3f}',
                             line=dict(color='#3498db', width=2),
                             fill='tozeroy', fillcolor='rgba(52,152,219,0.1)'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             line=dict(color='gray', dash='dash'), name='Random'))
    fig.update_layout(title=f'ROC Curve (AUC = {roc_auc:.3f})',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate')
    return fig


def plot_precision_recall(y_test, probs):
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1[:-1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                             line=dict(color='#e67e22', width=2),
                             fill='tozeroy', fillcolor='rgba(230,126,34,0.1)',
                             name='PR Curve'))
    fig.add_trace(go.Scatter(x=[recall[best_idx]], y=[precision[best_idx]],
                             mode='markers', marker=dict(size=12, color='red'),
                             name=f'Best F1={f1[best_idx]:.3f}'))
    fig.update_layout(title='Precision-Recall Curve',
                      xaxis_title='Recall', yaxis_title='Precision')
    return fig


def plot_confusion_matrix(y_test, probs, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    labels = ['Legit', 'Fraud']
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    x=labels, y=labels,
                    title=f'Confusion Matrix (threshold={threshold:.2f})',
                    labels=dict(x='Predicted', y='Actual'))
    fig.update_layout(height=400)
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation='h',
        marker_color='#3498db',
    ))
    fig.update_layout(title=f'Top {top_n} Feature Importances',
                      xaxis_title='Importance Score', height=500)
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

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Precision / Recall / F1', 'Business Cost ($K)'])
    fig.add_trace(go.Scatter(x=thresholds, y=precisions, name='Precision',
                             line=dict(color='#3498db')), row=1, col=1)
    fig.add_trace(go.Scatter(x=thresholds, y=recalls, name='Recall',
                             line=dict(color='#e74c3c')), row=1, col=1)
    fig.add_trace(go.Scatter(x=thresholds, y=f1s, name='F1',
                             line=dict(color='#2ecc71', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=thresholds, y=costs, name='Cost ($K)',
                             line=dict(color='#e67e22', width=2),
                             fill='tozeroy', fillcolor='rgba(230,126,34,0.1)'),
                  row=1, col=2)
    fig.update_xaxes(title_text='Threshold')
    fig.update_layout(height=400, showlegend=True)
    return fig


def plot_velocity_heatmap(df):
    if 'hour' not in df.columns or 'transaction_velocity_1h' not in df.columns:
        return None
    pivot = df.groupby(['hour', 'label'])['transaction_velocity_1h'].mean().unstack(fill_value=0)
    pivot.columns = ['Legit', 'Fraud']
    fig = px.imshow(pivot.T, color_continuous_scale='YlOrRd',
                    title='Avg Transaction Velocity by Hour & Class',
                    labels=dict(x='Hour of Day', y='Class', color='Avg Velocity'),
                    text_auto='.1f', aspect='auto')
    return fig


def plot_fraud_by_hour(df):
    if 'hour' not in df.columns:
        return None
    hourly = df.groupby('hour')['label'].agg(['sum', 'count']).reset_index()
    hourly.columns = ['hour', 'fraud', 'total']
    hourly['fraud_rate'] = hourly['fraud'] / hourly['total'] * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=hourly['hour'], y=hourly['total'],
                         name='Total Txns', marker_color='#bdc3c7', opacity=0.6))
    fig.add_trace(go.Scatter(x=hourly['hour'], y=hourly['fraud_rate'],
                             name='Fraud Rate %', line=dict(color='#e74c3c', width=3),
                             mode='lines+markers'), secondary_y=True)
    fig.update_layout(title='Fraud Rate by Hour of Day',
                      xaxis_title='Hour', barmode='overlay')
    fig.update_yaxes(title_text='Transaction Count', secondary_y=False)
    fig.update_yaxes(title_text='Fraud Rate (%)', secondary_y=True)
    return fig


def plot_fraud_by_channel(df):
    if 'channel' not in df.columns:
        return None
    ch = df.groupby('channel')['label'].agg(['sum', 'count']).reset_index()
    ch.columns = ['channel', 'fraud', 'total']
    ch['fraud_rate'] = ch['fraud'] / ch['total'] * 100
    fig = px.bar(ch, x='channel', y='fraud_rate', color='fraud_rate',
                 color_continuous_scale='Reds', title='Fraud Rate by Channel',
                 labels={'fraud_rate': 'Fraud Rate (%)', 'channel': 'Channel'},
                 text_auto='.1f')
    return fig


def plot_scatter_risk(df, sample=2000):
    if 'transaction_amount' not in df.columns or 'distance_from_home_km' not in df.columns:
        return None
    sample_df = df.sample(min(sample, len(df)), random_state=42).copy()
    sample_df['Class'] = sample_df['label'].map({0: 'Legit', 1: 'Fraud'})
    fig = px.scatter(sample_df, x='distance_from_home_km', y='transaction_amount',
                     color='Class', opacity=0.6,
                     color_discrete_map={'Legit': '#2ecc71', 'Fraud': '#e74c3c'},
                     title='Amount vs Distance from Home',
                     labels={'distance_from_home_km': 'Distance (km)',
                             'transaction_amount': 'Amount ($)'},
                     hover_data=[c for c in ['hour', 'is_foreign', 'vpn_detected']
                                 if c in sample_df.columns])
    return fig


def plot_anomaly_scatter(anomaly_scores, probs, y_test):
    df_plot = pd.DataFrame({
        'Anomaly Score': anomaly_scores,
        'Fraud Probability': probs,
        'Actual': pd.Series(y_test.values).map({0: 'Legit', 1: 'Fraud'})
    })
    fig = px.scatter(df_plot, x='Anomaly Score', y='Fraud Probability',
                     color='Actual',
                     color_discrete_map={'Legit': '#2ecc71', 'Fraud': '#e74c3c'},
                     title='Anomaly Score vs Fraud Probability',
                     opacity=0.6)
    fig.add_hline(y=0.3, line_dash='dash', line_color='orange',
                  annotation_text='Default threshold')
    return fig


def plot_shap_bar_interactive(shap_values, feature_names, top_n=15):
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[-top_n:]
    fig = go.Figure(go.Bar(
        x=mean_abs[idx],
        y=[feature_names[i] for i in idx],
        orientation='h',
        marker_color='#9b59b6',
    ))
    fig.update_layout(title=f'SHAP Feature Importance (Mean |SHAP|)',
                      xaxis_title='Mean |SHAP Value|', height=500)
    return fig
