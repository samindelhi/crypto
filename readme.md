# Crypto Regime-Aware Forecasting Pipeline
## *Summary: A Regime-aware crypto liquidity forecasting model using ensemble ML*

## 📦 Overview
This project implements a regime-aware ML pipeline for forecasting crypto liquidity using volatility-sensitive features and synthetic augmentation. It routes predictions through calm and shock models based on engineered regime flags.

---

## 🧠 Machine Learning Models Used
### This project leverages a regime-aware ensemble architecture using **LightGBM**, tailored for financial time-series forecasting:

### 🔹 Calm Regime Model
- Trained on low-volatility periods
- Focuses on subtle liquidity signals and stable market behavior
- Optimized for precision in calm conditions

### 🔹 Shock Regime Model
- Trained on high-volatility, sentiment-driven periods
- Captures sudden liquidity shifts and market reactions
- Designed for robustness under stress

### 🔹 Ensemble Routing Logic
- Regime detection via shock_z and volatility/sentiment features
- Dynamic model selection: routes each coin-day to either calm or shock model
- Ensures predictions are context-sensitive and audit-safe

### 🧪 Advanced ML Techniques
|Technique|Purpose|
|--------|--------|
|SHAP-based feature pruning|Improves interpretability and reduces noise in high-dimensional data.|
|Stacked ensemble routing|Combines predictions from regime-specific models for robust output.|
|Hybrid loss functions|Balances precision and robustness across calm and shock regimes.|
|Drift & shock diagnostics|Detects regime transitions and model degradation over time.|
|Fold-adaptive training|Ensures leak-free validation across temporal splits.|
|Modular pipeline design|Enables reproducible, audit-safe deployment and future reuse.|

___

## 🧠 Architecture
- **Feature Engineering**: `build_features.py`
- **Model Training**: `train_models.py`
- **Live Prediction**: `predict_live.py`
- **Regime Routing**: via `regime_shock` flag
- **Synthetic Augmentation**: applied to shock regime
---

## 📁 Folder Structure
```
├── app.py          # Streamlit App to run the test prediction.
├── convert_csv_to_pkl.py
├── model_meta.json
├── Handoff-Note.md     # Hand off note 
├── readme.md           # Readme.md for git.
├── requirements.txt
├── shap_ranking_fold1.csv
├── EDA
│   ├── Cryto_v2.ipynb
│   └── crypto_eda.ipynb
├── datasets
│   ├── coin_gecko_2022-03-16.csv
│   ├── coin_gecko_2022-03-17.csv
│   └── crypto_sentiment_prediction_dataset.csv
├── docs
│   ├── Crypto_HLD_Document.md
│   ├── Crypto_Pipeline.png
│   ├── DataFlowSummary.png
│   ├── Final_report.md
│   ├── HLD-LLD.md
│   ├── Pipeline_Arch_Doc.md
│   ├── crypto_assignment.md
│   ├── eda_report.md
│   └── model_evaluation_report.md
├── models
│   ├── df_train_augmented.pkl
│   ├── feature_cols.pkl
│   ├── model_calm.pkl
│   └── model_shock.pkl
├── notebooks
│   ├── crypto_timeseries_90d.csv
│   ├── crypto_training_data.pkl
│   ├── fetch_live.ipynb
│   └── live_crypto_data_clean.pkl
├── plot_images
│   ├── eda_Univariate_momentum.png
│   ├── eda_Univariate_shock_z.png
│   ├── eda_Univariate_vol_ratio.png
│   ├── eda_correlation_matrix.png
│   ├── eda_momentumDist_by_regime.png
│   ├── eda_price_volume_trends.png
│   ├── eda_shockzDist_by_regime.png
│   ├── eda_time_series_trends.png
│   ├── eda_vol_ratio_Dist_by_regime.png
│   ├── eval_ModelEvaluationDashboard.png
│   ├── eval_ResidualDrift.png
│   ├── eval_ResidualvsTimestamp.png
│   ├── eval_ShapDriftAcrossFolds.png
│   ├── eval_ShapFeatureImportanceHeatmap.png
│   ├── eval_Top20ShapFeatureForFold1.png
│   ├── eval_Top20ShapFeatureForFold2.png
│   ├── eval_Top20ShapFeatureForFold3.png
│   ├── eval_Top20ShapFeatureForFold4.png
│   ├── eval_Top20ShapFeatureForFold5.png
│   ├── eval_momentumDriftAcrossFolds.png
│   ├── eval_shock_zDriftAcrossFolds.png
│   └── eval_vol_ratioDriftAcrossFolds.png
├── results
│   ├── LivePredictionsOverTime.png
│   ├── tuning_gridsearch_RMSEvsR2.png
│   ├── wfv_best_config_20250925_0202.json
│   ├── wfv_best_config_20250926_0722.json
│   ├── wfv_best_config_20250926_0731.json
│   ├── wfv_best_config_20250926_1419.json
│   └── wfv_best_config_20250926_1547.json
├── sample
│   ├── crypto_sample_test.csv
│   ├── crypto_sample_test.pkl
│   └── crypto_test_data.pkl
└── src
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── diagnostics.py
    ├── eda.py
    ├── evaluation.py
    ├── feature_engineering.py
    ├── main.py
    ├── model_runner.py
    ├── predict_live.py
    ├── shap_utils.py
    ├── tuning.py
    ├── tuning_cli.py
    └── utils.py
```

---
## 📊 Model Summary - Training dataset
| Model       | Samples | Regime | Target           | Prediction Range |
|-------------|---------|--------|------------------|------------------|
| Calm Model  | 892     | Calm   | `liquidity_ratio`| 0.175 – 0.340    |
| Shock Model | 385     | Shock  | `liquidity_ratio`| 0.047 – 0.121    |

---

## 🧪 Diagnostics
- Regime split: 892 calm, 385 shock
- Feature importances logged
- SHAP values optional
- Live predictions show regime sensitivity

---

## 🚀 How to Run
1. Train models:
   ```bash  
    python train_models.py```

2. Run live predictions:
    ```bash
    python predict_live.py```

📤 Handoff Notes & Artefacts

>### Documentation Index
- See [Readme.md](readme.md) — Project homepage and setup guide.
- See [Problem Statement](docs/crypto_assignment.md) — Assignment brief and scope.
- See [Handoff-Note.md](docs/Handoff-Note.md) — Final handoff checklist and packaging instructions.
- See [folder_structure.txt](folder_structure.txt) — Full directory layout and artifact map.
- See [Crypto_HLD_Document.md](docs/Crypto_HLD_Document.md) — High-level design overview of the crypto pipeline.
- See [HLD-LLD.md](docs/HLD-LLD.md) — Combined high-level and low-level design notes.
- See [ML Pipeline Architecture](docs/Pipeline_Arch_Doc.md) — Architecture documentation for the ML pipeline.
- See [EDA Report](docs/eda_report.md) — Exploratory data analysis findings.
- See [Mode Evaluation Report](docs/model_evaluation_report.md) — Evaluation metrics and model performance summary.
- See [Final Report](docs/Final_report.md) — Final project report and summary.



>### Final packaging includes:
> - See [Handoff-Note.md](docs/Handoff-Note.md) for deployment instructions and final packaging notes.
> - Zipped folder with models, scripts, sample files
> - README with usage instructions
> - Preprocessed `.pkl` and raw `.csv` for live testing




| Component                          | Status        | Notes                                                                 |
|-----------------------------------|---------------|-----------------------------------------------------------------------|
| **Source Code**                   | ✅ Complete    | Modular scripts in `/scripts`, models in `/models`, data in `/data`  |
| **Code Documentation**            | ✅ Done        | Well-commented, assignment-ready scripts                             |
| **Streamlit Deployment**          | ✅ Done        | `app.py` with full pipeline, plots, export, and regime routing  with screenshots     |
| **EDA Report**                    | ✅ Done        | Basic trends, correlations, distributions. See [EDA Report](./eda_report.md) for full visual and statistical analysis. |
| **High-Level Design (HLD)**       | ✅ Done        | System overview, architecture diagram. See [HLD](./hld.md)           |
| **Low-Level Design (LLD)**        | ✅ Done        | Component breakdown, function-level logic. See [LLD](./lld.md)       |
| **Pipeline Architecture Doc**     | ✅ Done        | Data flow from ingestion to prediction. See [Pipeline Doc](./pipeline_architecture.md) |
| **Final Report**                  | ✅ Done        | See below for full summary                                           |


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


### Usage Instructions
- 1. Run the Streamlit App (app.py)
    - Launch the live prediction interface:
    ``` streamlit run app.py ```
    - Upload either a .csv or preprocessed .pkl file from the sample/ folder
    - The app will auto-detect regime, route predictions, and display results

- 2. Convert Raw CSV to Preprocessed .pkl as long as its the same format of the dataset.
    - Use the utility script to convert a raw .csv into a model-ready .pkl:
    ``` python convert_csv_to_pkl.py ```
    - Default input: sample/crypto_sample_test.csv
    - Output: sample/crypto_sample_test.pkl


