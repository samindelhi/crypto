# src/config.py
import os
# Model and pipeline configuration
NUM_FOLDS = 5
BLEND_WITH_LAG = 5
TOP_N_FEATURES = 20
ROLLING_WINDOW = 30
TRAIN_COLS = {'coin', 'symbol', 'price', '1h', '24h', '7d', '24h_volume', 'mkt_cap', 'date'}

# SHAP settings
SHAP_SAMPLE_SIZE = 1000
SHAP_SEED = 42

# Data paths
DATASETS_DIR = './datasets/'
# 1DATA_PATH_1 = "..\crypto\datasets\coin_gecko_2022-03-16.csv"

DATA_PATH_1 = os.path.join(DATASETS_DIR,   'coin_gecko_2022-03-16.csv')
DATA_PATH_2 = os.path.join(DATASETS_DIR,  'coin_gecko_2022-03-17.csv')

# Output folders
PLOT_DIR = './plot_images/'
MODEL_DIR = './models/'
LOG_DIR = './logs/'
EVAL_DIR = './results/'


# Output files
EVAL_SNAPSHOT_MD = './docs/evaluation_snapshot.md'
PIPELINE_SCRIPT = './src/main.py'

# Target column
TARGET_COL = 'liquidity_ratio'

# Build Features
LAG_LIST = [1, 2, 3, 7, 14]
ROLL_WINDOWS = [3, 7, 14]
CAT_COLS = ['coin', 'symbol']  # adjust if needed

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'max_depth': 6,
    'min_data_in_leaf': 300,
    'learning_rate': 0.02,
    'num_leaves': 15,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': -1
}