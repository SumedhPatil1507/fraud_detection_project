import pandas as pd
import numpy as np
from src.config import DATA_PATH, SAMPLE_PATH
from src.graph_intelligence import extract_graph_features

try:
    import networkx as nx
    _NX_AVAILABLE = True
except Exception:
    _NX_AVAILABLE = False


def load_data(uploaded_file=None):
    """Load data from uploaded file, local CSV, or sample fallback."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] > 2:
                return df, "uploaded"
        except Exception:
            pass

    for path, sep in [(DATA_PATH, '\t'), (DATA_PATH, ','), (SAMPLE_PATH, ',')]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 2:
                return df, "local"
        except Exception:
            continue

    raise FileNotFoundError("No valid data file found.")


def preprocess(df):
    df = df.drop_duplicates()
    df = df.drop(columns=['transaction_id', 'device_id', 'fraud_type'], errors='ignore')
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        if df[col].nunique() <= 20:
            df[col] = pd.Categorical(df[col]).codes
        else:
            df = df.drop(columns=[col])
    return df


def feature_engineering(df):
    if 'transaction_amount' in df.columns:
        df['amount_log'] = np.log1p(df['transaction_amount'])
    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    if 'avg_amount_30d' in df.columns and 'transaction_amount' in df.columns:
        df['amount_vs_avg'] = df['transaction_amount'] / (df['avg_amount_30d'] + 1)
    if 'transaction_amount' in df.columns and 'distance_from_home_km' in df.columns:
        df['amount_x_distance'] = df['transaction_amount'] * df['distance_from_home_km']
    if 'transaction_velocity_1h' in df.columns and 'transaction_velocity_24h' in df.columns:
        df['velocity_ratio'] = df['transaction_velocity_1h'] / (df['transaction_velocity_24h'] + 1)
    return df


def run_pipeline(uploaded_file=None, raw_df=None):
    if raw_df is not None:
        df_raw = raw_df
        source = "uploaded"
    else:
        df_raw, source = load_data(uploaded_file)

    # Rich graph features: degree, betweenness, fraud_rate, ring_member
    graph_feats = extract_graph_features(df_raw)

    df = preprocess(df_raw)
    df = feature_engineering(df)

    if not graph_feats.empty:
        df = pd.concat([df.reset_index(drop=True),
                        graph_feats.reset_index(drop=True)], axis=1)
        # Drop duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]

    return df, source
