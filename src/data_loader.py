# src/data_loader.py

import pandas as pd
import numpy as np
from src.config import DATA_PATH_1, DATA_PATH_2

def crypto_data_ingestion():
    df1 = pd.read_csv(DATA_PATH_1)
    df2 = pd.read_csv(DATA_PATH_2)
    df = pd.concat([df1, df2])
    return df

# Data Preprocessing
def perform_data_preprocessing(df):
    """ #### Performs Data Preprocessing: 
        - Column Renaming, 
        - Missing value imputation
        - Convertion of Datetime values,  """

    #Step1: Rename columns"""

    df.rename(columns={
        '1h': 'pct_change_1h',
        '24h': 'pct_change_24h',
        '7d': 'pct_change_7d',
        '24h_volume': 'volume_24h'
    }, inplace=True)

    # Check missing values before filling.
    print(f"Missing values before fill: \n {df.isna().sum()}")

    #Step2: Median imputation for percentage change columns.
    for col in ['pct_change_1h', 'pct_change_24h', 'pct_change_7d']:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    # Step3: # Median imputation for volume_24h
    # If you want to treat missing volume as "no trades", use 0 instead of median
    median_vol = df['volume_24h'].median()
    df['volume_24h'].fillna(median_vol, inplace=True)

    # Verify no missing values remain
    print(f"\n Missing values after fill: {df.isna().sum()}")

    # Step 4: Convert and Sort by Date
    #- Ensures chronological order for rolling averages, volatility, and train/test splits.
    #- Avoids data leakage when creating time-based features.

    # convert 'date' column to date time
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Check for any parsing issues
    print('Null dates after conversion: ', df['date'].isna().sum())

    # sort by date (ascending)
    df.sort_values(by='date', inplace=True)

    # Reset index after sorting
    df.reset_index(drop=True, inplace=True)

    print(df.columns)

    return df

# Perform Scaling using Standard Scalar, if necessary
def perform_scaling(df):
    """#### Scaling Numerical Features since they are in different scale. 
    - using StandardScalar().
    - numeric columns: 'price', 'pct_change_1h', 'pct_change_24h', 'pct_change_7d', 'volume_24h', 'mkt_cap'"""
    from sklearn.preprocessing import StandardScaler

    # Select numeric columns for scaling
    num_cols = ['price', 'pct_change_1h', 'pct_change_24h', 'pct_change_7d', 'volume_24h', 'mkt_cap']

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    #Quick Check
    print(df[num_cols].describe().round(2))

    return df

# Step 4:(Optional) Log Transform Skewed Features
def perform_log_transform_skewed_features(df):
    """(Optional) Perform Log Transform on Skewed Features
    - volumne_24h
    - mkt_cap"""

    df['volume_24h'] = np.log1p(df['volume_24h'])
    df['mkt_cap'] = np.log1p(df['mkt_cap'])

    return df

def load_and_prepare_data():        # Not used.
    pass




def engineer_features(df):
    # Add your feature engineering logic here
    df['log_return'] = np.log(df['price']).diff()
    df['volatility'] = df['log_return'].rolling(10).std()
    return df.dropna()
