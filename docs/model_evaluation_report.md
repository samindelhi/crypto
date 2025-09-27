# 📘 Model Evaluation Snapshot — Documentation

## 1. Pipeline Configuration
- **Model Type**: Regime-aware ensemble (fold-routed)
- **Feature Selection**: Top 20 via SHAP pruning
- **Blending Strategy**: `blend_with_lag=5`
- **Warm-up Logic**: Enabled for fold-safe blending
- **Residual Correction**: Applied post-prediction
- **Timestamp Alignment**: Fixed via positional slicing

---

## 2. Key Fixes & Patches

| Component      | Issue                            | Fix Applied                          |
|----------------|----------------------------------|--------------------------------------|
| `test_feat`     | Misaligned timestamps            | Index-safe slicing with `.iloc`      |
| `fold_outputs`  | Missing timestamp for dashboard  | Stored `test_feat['timestamp']`      |
| `df_all`        | Length mismatch in concat        | Used stored timestamps per fold      |
| SHAP Drift      | Fold-wise instability            | Diagnosed via std dev bar chart      |
| Residuals       | Potential drift                  | Validated with rolling mean/std plot |

---

## 3. Evaluation Dashboard Components
- 📈 Predicted vs Actual (time series)

- 📊 Residual Distribution & Residuals vs Predicted
- 📉 Rolling RMSE & MAE (window=30)
- 📋 Fold-wise Metrics (RMSE, MAE, R²)
- 🔥 SHAP Heatmap Across Folds
- 📐 SHAP Drift Bar Chart
- 🌊 Residual Drift Plot

---
## 3.1 Model Evaluation Metrics.
``` 
    === Single Split Summary ===

        RMSE: 0.1193
        MAE:  0.0336
        R²:   0.7879

        Fold 1: Shock Regime Count = 37, Calm Count = 432
        Fold 1: Sparse regime split — using unified model
        Fold 1: RMSE=0.0431, MAE=0.0199, R²=0.7675

        Fold 2: Shock Regime Count = 42, Calm Count = 527
        Fold 2: Sparse regime split — using unified model
        Fold 2: RMSE=0.0274, MAE=0.0178, R²=0.8882

        Fold 3: Shock Regime Count = 55, Calm Count = 614
        Fold 3: RMSE=0.4204, MAE=0.1650, R²=0.7028

        Fold 4: Shock Regime Count = 65, Calm Count = 704
        Fold 4: RMSE=0.0797, MAE=0.0428, R²=0.8764

        Fold 5: Shock Regime Count = 60, Calm Count = 809
        Fold 5: RMSE=0.2987, MAE=0.1247, R²=0.7598

    === Walk Forward Validation (WFV) Summary ===

        Avg RMSE: 0.1739
        Avg MAE:  0.0740
        Avg R²:   0.7989
```
## 4. Optional Enhancements
- Regime overlays on time series
- Volatility zone annotations
- SHAP deltas across regimes
- Residual autocorrelation plots

---

## 5. Reproducibility Notes
- All folds use consistent feature sets and warm-up logic
- SHAP values computed post-prediction, per fold
- Dashboard plots use `df_all` with aligned timestamps
- Evaluation metrics stored in `fold_outputs` for audit