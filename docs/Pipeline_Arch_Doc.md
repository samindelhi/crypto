# 🔄 Pipeline Architecture Document

## 📌 Overview
This pipeline transforms raw crypto data into regime-aware predictions using engineered features, fold-adaptive models, and a Streamlit dashboard for live interaction.

### 🧬 Data Flow: End-to-End
1. Data Ingestion
    - Input: .pkl files containing raw crypto time-series data
    - Source: CoinGecko API or pre-cleaned snapshots
    - Format: Daily OHLCV + derived metrics
2. Feature Engineering (build_features.py)
    - Rolling stats: roll_mean_7, roll_std_7
    - Momentum: momentum_1d, pct_change_24h
    - Volatility ratios: vol_ratio_7_30
    - Regime tagging: shock_z computed via z-score
    - SHAP pruning: Removes low-impact features
3. Model Training (train_models.py)
    - Split by regime_shock (calm vs shock)
    - Fold-aware validation (TimeSeriesSplit)
    - Train LightGBM models per regime
    - Save: model_calm.pkl, model_shock.pkl, feature_cols.pkl
4. Live Prediction (predict_live.py)
    - Load uploaded .pkl via Streamlit
    - Apply build_features() to live data
    - Detect regime using shock_z
    - Route to appropriate model
    - Return full df_feat with prediction
5. Dashboard Visualization (app.py)
    - Upload interface for .pkl files
    - Display predictions, regime overlays
    - SHAP summary plot for interpretability
    - Export predictions as .csv or view in table

### 🧠 Routing Logic
```
if df_feat['shock_z'].iloc[-1] > 1.5:
    model = model_shock
else:
    model = model_calm
```


### 📁 Artifacts Generated
|Artifacts|Purpose|
|----|------| 
| model_calm.pkl |Calm regime predictor  | 
| model_shock.pkl |Shock regime predictor  | 
| feature_cols.pkl |Feature list for inference  | 
| df_train_augmented.pkl |Augmented training data  | 
| live_crypto_data_clean.pkl |Cleaned live input for realtime prediction and dashboarding | 



### 🧩 Modularity Highlights
- Each module is independently testable
- Feature builder is reusable across training and inference
- SHAP logic is decoupled for diagnostics
- Dashboard is plug-and-play with .pkl inputs
