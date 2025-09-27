# Cryptocurrency Liquidity Prediction for Market Stability 
- 📁 Project Plan & Audit Checklist (Optimized & Aligned)

## Problem Statement
Build a machine learning model to forecast cryptocurrency liquidity levels using market signals such as:
- Trading volume
- Price momentum
- Volatility regimes
- Exchange activity
- Social sentiment (if available)
The goal is to detect liquidity crises early and support risk-aware decision-making for traders and platforms.

## Dataset Information
- Source: Google Drive Dataset (2016–2017)
- Contents: Historical price, volume, and market metadata
- Format: Time-series data per coin/symbol

## Data Preprocessing
### Required Steps
- Handle missing values
- Ensure time alignment and consistency
- Normalize numerical features (only if required by model)
- Apply log transform conditionally based on target skew
- Remove duplicates and outliers
### Additional Steps (Implemented)
- SHAP-based feature pruning
- Regime flag generation (regime_shock, regime_high_vol)
- Fold-aware feature drift diagnostics

## Exploratory Data Analysis (EDA)
### Summary of Work Done
- Descriptive Statistics
- Summary of price, volume, volatility, and liquidity ratios
- Regime-wise breakdown of key metrics
- Visualizations
- Fold-wise feature drift plots using plot_feature_drift()
- Calm vs shock regime comparisons using plot_regime_split()
- Distribution plots for engineered features (shock_z, momentum_1d, vol_ratio_7_30)
- Optional SHAP overlays for feature importance visualization
- Correlation Analysis
- Feature-target correlation matrix
- Regime-aware correlation shifts (optional)
- Regime Diagnostics
- Count and distribution of regime_shock flags
- Transition patterns between calm and shock regimes
- Volatility clustering and drift detection across folds
### Notes
- Scaling and log transformation were skipped unless required by model behavior
- EDA was focused on actionable insights for model routing and diagnostics
- Visuals were used to validate feature stability and regime separation

## Feature Engineering
### Categories & Examples
- Lag Features: lag_1, lag_3, lag_7, lag_14
- Rolling Stats: roll_mean_7, roll_std_7, vol_ratio_7_30
- Momentum: momentum_1d, abs_momentum_1d
- Regime Flags: regime_shock, regime_high_vol, std7_volflag
- Residual Indicators: shock_z, shock_lag1, lag1_shock
- SHAP Pruning: Top-N features selected per fold

## Model Selection
### Models Used
- LightGBM: Main predictor with monotonic constraints
- Ridge Regression: Linear fallback model
- XGBoost: Residual correction model
- Ensemble Logic: Regime-aware routing + fallback blending

## Model Training & Validation
### Strategy
- Walk-forward validation across 5 folds
- Unified fallback model for sparse regime splits
- Residual modeling applied post-prediction
- Evaluation metrics: RMSE, MAE, R² (per fold and average)

## Hyperparameter Tuning
### Parameters Tuned
- fallback_weights: Lag blend ratios
- residual_weight: Strength of residual correction
- top_n_features: SHAP feature count per fold
### Method
- Grid search across combinations
- Selection based on highest average R² and lowest RMSE

## Model Testing
### Tests Performed
- Real-time JSON payloads to Flask API
- Fold-wise feature drift visualization
- Calm vs shock regime comparisons
- SHAP overlays on feature distributions (optional)

## Local Deployment
### Flask API Endpoints
- /predict: Accepts JSON input, returns liquidity prediction
- /health: Confirms service is live
- /version: Returns model version and config summary
### Modules
- app.py: Flask server
- model_loader.py: Loads serialized models and config
- predictor.py: Handles preprocessing, routing, fallback, residual
- utils.py: Feature engineering and diagnostics

## Expected Deliverables
### 1. Machine Learning Model
- Trained ensemble with fallback and residual logic
- Evaluation metrics summary (RMSE, MAE, R²)
### 2. Data Processing & Feature Engineering
- Cleaned dataset
- Feature list with rationale and SHAP rankings
### 3. EDA Report
- Summary statistics
- Visualizations: drift, regime splits, SHAP overlays
### 4. Project Documentation
- High-Level Design (HLD): System overview
- Low-Level Design (LLD): Module breakdown
- Pipeline Architecture: Data flow from input to prediction
- Final Report: Findings, model performance, insights

## Submission Guidelines
- GitHub repo or zipped folder containing:
- Source code
- EDA report
- HLD & LLD documents
- Pipeline architecture
- Final report
- Flask API for prediction testing
- Code Documentation: Clear, well-commented scripts
- Diagrams & Visuals: For data flow, model logic, evaluation
- Deployment Interface: Flask API with test payloads
