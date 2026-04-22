"""PII Masking & Data Scrubbing"""
import pandas as pd
import re

PII_COLUMNS = ['customer_id', 'merchant_id', 'device_id', 'home_lat',
               'home_lon', 'txn_lat', 'txn_lon', 'merchant_lat', 'merchant_lon']

def mask_pii(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in PII_COLUMNS:
        if col in df.columns:
            if 'lat' in col or 'lon' in col:
                df[col] = df[col].apply(lambda x: round(x, 1) if pd.notnull(x) else x)
            else:
                df[col] = df[col].astype(str).apply(
                    lambda x: x[:3] + '***' + x[-2:] if len(x) > 5 else '***')
    return df

def scrub_text(text: str) -> str:
    text = re.sub(r'\b\d{16}\b', '[CARD_REDACTED]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
    text = re.sub(r'[\w.+-]+@[\w-]+\.[a-zA-Z]+', '[EMAIL_REDACTED]', text)
    return text
