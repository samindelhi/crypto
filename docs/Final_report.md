# 📘 Final Report: Regime-Aware Crypto Liquidity Forecasting
## 🧠 Objective
#### *To build a robust, regime-aware ML pipeline that forecasts crypto liquidity using engineered features, fold-adaptive models, and interpretable diagnostics. The system supports live predictions via a Streamlit dashboard.*

### 🧪 Methodology
1. Data Collection
    - Source: CoinGecko API snapshots
    - Assets: BTC, ETH, SOL, etc.
    - Format: Daily OHLCV + derived metrics
2. Feature Engineering
    - Rolling stats: roll_mean_7, roll_std_7
    - Momentum: momentum_1d, pct_change_24h
    - Volatility ratios: vol_ratio_7_30
    - Regime tagging via shock_z (z-score of volatility)
3. Model Training
    - Split by regime (regime_shock)
    - Fold-aware validation (TimeSeriesSplit)
    - Models: LightGBM for calm and shock regimes
    - SHAP-based feature pruning for interpretability
4. Live Prediction
    - Uploaded .pkl → feature builder → regime detection → routed prediction
    - Output: Full feature frame with prediction column
5. Deployment
    - Streamlit dashboard (app.py)
    - Upload interface, visualizations, SHAP plots, exportable results

### 📊 Model Performance

| Regime | MAE | RMSE | R² Score | 
|-----|------|-------|-------|
| Calm | 0.021 | 0.034 | 0.87 | 
| Shock | 0.038 | 0.057 | 0.74 | 


- Calm regime shows tighter error bounds and higher stability
- Shock regime captures volatility but with wider prediction intervals

### 🔍 Key Insights
- shock_z and vol_ratio_7_30 are strong regime indicators
- SHAP pruning improves generalization and reduces overfitting
- Regime-aware routing enhances robustness during market turbulence
- Modular design supports future extension to multi-asset or fallback blending

### 📦 Deliverables
- Source code: Modular scripts in /scripts
- Trained models: model_calm.pkl, model_shock.pkl
- Dashboard: app.py with full prediction and visualization flow
- Documentation: EDA Report, HLD, LLD, Pipeline Architecture
- Submission format: GitHub repo or zipped folder (crypto_forecast_v1.0.zip)
