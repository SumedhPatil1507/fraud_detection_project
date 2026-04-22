"""
Universal file ingestion — supports CSV, Excel, JSON, Parquet.
Also handles live data simulation mode.
"""
import pandas as pd
import io


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read any uploaded file format into a DataFrame."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith('.csv'):
            # Try comma first, then tab
            content = uploaded_file.read()
            try:
                df = pd.read_csv(io.BytesIO(content))
                if df.shape[1] > 2:
                    return df
            except Exception:
                pass
            return pd.read_csv(io.BytesIO(content), sep='\t')

        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)

        elif name.endswith('.json'):
            return pd.read_json(uploaded_file)

        elif name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)

        else:
            # Try CSV as fallback
            return pd.read_csv(uploaded_file)

    except Exception as e:
        raise ValueError(f"Could not read file '{uploaded_file.name}': {e}")


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """Basic validation — check for label column and minimum rows."""
    if df.shape[0] < 10:
        return False, "Dataset too small (< 10 rows)."
    if 'label' not in df.columns:
        return False, "Missing required 'label' column (0=legit, 1=fraud)."
    if df['label'].nunique() < 2:
        return False, "Label column must have both classes (0 and 1)."
    return True, "OK"
