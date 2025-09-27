# src/utils.py
import warnings
warnings.filterwarnings('ignore')

def suppress_warnings():
    warnings.filterwarnings('ignore')


def safe_slice(df, start_idx, end_idx):
    return df.iloc[start_idx:end_idx].copy()

def fix_timestamp_alignment(df, reference_ts):
    df['timestamp'] = reference_ts
    return df