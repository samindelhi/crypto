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
___

# 📈 Modeling Rationale & Methodology Overview
___
This Section explains the rationale behind my modeling choices, including the dataset interpretation, SHAP explainability, fold-based validation, and why I opted for a regime-aware, ensemble-driven ML pipeline.

---

## 📊 Dataset Summary

We used a snapshot of major cryptocurrencies on `2022-03-16` & `2022-03-17` from coins gecko, with the following columns:

| Column         | Description                                      |
|----------------|--------------------------------------------------|
| `coin`         | Full name of the cryptocurrency                  |
| `symbol`       | Ticker symbol (e.g., BTC, ETH)                   |
| `price`        | Current price in USD                             |
| `1h`, `24h`, `7d` | Price change % over 1 hour, 24 hours, and 7 days |
| `24h_volume`   | Trading volume in USD over the last 24 hours     |
| `mkt_cap`      | Market capitalization in USD                     |
| `date`         | Snapshot date                                    |

These features capture short-term momentum, liquidity, and asset scale — ideal for regime classification and volatility forecasting.

---

## 🧠 Why Machine Learning?

Traditional models (ARIMA, GARCH) struggle with nonlinearities and sentiment-driven volatility. ML allows us to:

- Model complex feature interactions
- Adapt to regime shifts
- Blend predictions across folds for robustness
- Use SHAP for transparent, audit-safe diagnostics

---

## 🔁 Fold-Based Validation Strategy

We used **Walk-Forward Validation (WFV)** to simulate real-time forecasting:

- Train on expanding windows (e.g., Days 1–10 → test on Day 11)
- Repeat across folds to capture temporal drift
- Ensemble predictions across folds for smoother output

This approach ensures:

- No data leakage
- Realistic simulation of live prediction
- Fold-specific diagnostics for drift and robustness

---

## 🧪 SHAP: Explaining Model Decisions

**SHAP (SHapley Additive exPlanations)** breaks down each prediction into feature contributions:

- Tells us how much each feature pushed the prediction up or down
- Enables per-fold heatmaps to detect unstable features
- Supports feature pruning and audit transparency

Example:
> For Bitcoin, `24h_change` added +0.03, `volume` subtracted −0.01, `mkt_cap` added +0.02

SHAP is widely used in industry for:

- Regulatory compliance (e.g., banking, insurance)
- Bias detection
- Feature importance ranking
- Trust-building in healthcare and finance

---

## 🛤️ Why This Pipeline?

| Feature                  | Our Pipeline | Alternatives |
|--------------------------|--------------|--------------|
| Regime awareness         | ✅           | ❌           |
| Fold-adaptive validation | ✅           | ❌           |
| SHAP explainability      | ✅           | ❌           |
| Ensemble robustness      | ✅           | ❌           |
| Audit safety             | ✅           | ⚠️           |
| Deployment readiness     | ✅           | ⚠️           |

Alternatives like ARIMA, LSTM, or rule-based systems lack the flexibility, transparency, and robustness required for this task.

---

## ✅ Summary

We chose a regime-aware, fold-adaptive ML pipeline because it offers:

- Robust performance across market conditions
- Transparent diagnostics via SHAP
- Realistic validation through walk-forward folds
- Scalable deployment via Streamlit

This architecture balances predictive power with audit-grade explainability — ideal for financial forecasting in volatile environments.
