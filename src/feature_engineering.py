# src/feature_engineering.py
import numpy as np


import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from src.config import LAG_LIST, LGB_PARAMS,TARGET_COL, ROLL_WINDOWS, CAT_COLS

# ----- Liquidity-related Feature Engineering. -----

def add_liquidity_feature(df, safe_mode=False):
    """ Liquidity-related Feature Engineering:
     -  """
    df = df.copy()

    # Rolling Mean Price (7-day window)
    df['rolling_mean_price_7d'] = df['price'].rolling(window=7, min_periods=1).mean()

    # Rolling Volatility (7-day window)
    df['rolling_volatility_7d'] = df['price'].rolling(window=7, min_periods=1).std()

    # Liquidity Ration (Volume / Market Cap)

    vol_col = '24h_volume' if '24h_volume' in df.columns else 'volume_24h'
    if safe_mode:
        if vol_col not in df.columns or 'mkt_cap' not in df.columns:
            df['liquidity_ratio'] = 0.0  # fallback value
    
    df['liquidity_ratio'] = df[vol_col] / (df['mkt_cap'] + 1e-9) # avoid div by zero


     # Price Momentum (difference from previous day)
    df['price_momentum'] = df['price'] - df['price'].shift(1)

    # Volume Change % (relative change from previous day)
    df['volume_change_pct'] = df['volume_24h'].pct_change().replace([np.inf, -np.inf], np.nan)

    # Z-score of Volume (standardized volume)
    vol_mean = df['volume_24h'].mean()
    vol_std = df['volume_24h'].std()
    df['volume_zscore'] = (df['volume_24h'] - vol_mean) / (vol_std + 1e-9)

    # Handle NaNs from rolling/pct_change
    df.fillna(0, inplace=True)

    return df

### 4. Feature Engineering - Build Features and Blending Function with LGB_PARAMS config.

def build_features(df: pd.DataFrame, regime_aware: bool = False) -> pd.DataFrame:
    """
    Prepares features from raw input DataFrame.
    Handles datetime indexing, target cleaning, lag/rolling stats,
    regime-aware indicators, and calendar features.
    """
    dfc = df.copy()

    # print("dfc.columns",dfc.columns)

    # Detect and set datetime index
    date_cols = [c for c in dfc.columns if 'date' in c.lower() or c.lower() in ('ds', 'timestamp', 'time')]
    if date_cols:
        dcol = date_cols[0]
        dfc[dcol] = pd.to_datetime(dfc[dcol])
        dfc = dfc.sort_values(dcol).set_index(dcol)
    else:
        dfc['date'] = pd.date_range(start='2000-01-01', periods=len(dfc), freq='D')
        dfc = dfc.set_index('date')

    # Clean target
    dfc[TARGET_COL] = pd.to_numeric(dfc[TARGET_COL], errors='coerce').ffill().bfill()

    # Drop non-numeric object columns
    for col in list(dfc.columns):
        if dfc[col].dtype == 'object' and col not in CAT_COLS:
            dfc = dfc.drop(columns=[col])

    # Lag features
    for lag in LAG_LIST:
        dfc[f'lag_{lag}'] = dfc[TARGET_COL].shift(lag)

    # Rolling stats
    for w in ROLL_WINDOWS:
        dfc[f'roll_mean_{w}'] = dfc[TARGET_COL].shift(1).rolling(w).mean()
        dfc[f'roll_std_{w}'] = dfc[TARGET_COL].shift(1).rolling(w).std()

    # Momentum indicators
    dfc['momentum_1d'] = dfc[TARGET_COL] - dfc[TARGET_COL].shift(1)
    dfc['momentum_7d'] = dfc[TARGET_COL] - dfc[TARGET_COL].shift(7)

    if regime_aware == True:
        # Respect Pre-Injected Regime Features - Compute only if missing.
        if 'shock_z' not in dfc.columns or 'vol_ratio_7_30' not in dfc.columns:
            delta = dfc[TARGET_COL].diff()
            vol_s = delta.rolling(7).std()
            vol_l = delta.rolling(30).std()
            dfc['vol_ratio_7_30'] = (vol_s / (vol_l + 1e-8)).shift(1)
            dfc['shock_z'] = ((delta - delta.rolling(30).mean()) / (delta.rolling(30).std() + 1e-8)).shift(1)

        dfc['regime_high_vol'] = (dfc['vol_ratio_7_30'] > 1.5).astype(int)
        dfc['regime_shock'] = (dfc['shock_z'].abs() > 2.0).astype(int)

        # Interaction terms
        dfc['lag1_shock'] = dfc['lag_1'] * dfc['shock_z']
        dfc['std7_volflag'] = dfc['roll_std_7'] * dfc['regime_high_vol']

        # Residual-sensitive features
        dfc['abs_momentum_1d'] = dfc['momentum_1d'].abs()
        dfc['shock_lag1'] = dfc['shock_z'] * dfc['lag_1']
        dfc['std7_shockflag'] = dfc['roll_std_7'] * dfc['regime_shock']

    # Calendar features
    dfc['dayofweek'] = dfc.index.dayofweek
    dfc['month'] = dfc.index.month

    # External drivers
    for col in ['volume_change_pct', 'mkt_cap']:
        if col in dfc.columns:
            dfc[col] = dfc[col].fillna(0.0)

    # print("Final feature columns:", dfc.columns.tolist())

    return dfc