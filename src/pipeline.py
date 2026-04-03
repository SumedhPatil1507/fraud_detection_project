import pandas as pd
import numpy as np
from src.config import DATA_PATH, SAMPLE_PATH


def load_data():
    for path, sep in [(DATA_PATH, '\t'), (DATA_PATH, ','), (SAMPLE_PATH, ',')]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 2:
                return df
        except Exception:
            continue
    raise FileNotFoundError("No valid data file found.")


def preprocess(df):
    df = df.drop_duplicates()
    df = df.drop(columns=['transaction_id', 'device_id', 'fraud_type'], errors='ignore')
    # Encode low-cardinality categoricals
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

    return df


def run_pipeline():
    df = load_data()
    df = preprocess(df)
    df = feature_engineering(df)
    return df
