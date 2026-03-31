import pandas as pd
import numpy as np
from src.config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH, sep='\t')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

def preprocess(df):
    df = df.drop_duplicates()
    df = df.drop(columns=['transaction_id','device_id'], errors='ignore')
    return df

def feature_engineering(df):
    if 'transaction_amount' in df.columns:
        df['amount_log'] = np.log1p(df['transaction_amount'])

    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
        df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

    return df

def run_pipeline():
    df = load_data()
    df = preprocess(df)
    df = feature_engineering(df)
    return df