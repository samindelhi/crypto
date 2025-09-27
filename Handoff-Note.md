# Handoff Summary: Crypto Forecasting Pipeline - Streamlit App.

This bundle contains a regime-aware ML pipeline for forecasting crypto liquidity. It includes:

- 📊 Trained models for calm and shock regimes
- 🧠 Feature engineering and synthetic augmentation logic
- 🚀 Live prediction script and Streamlit dashboard (`app.py`)
- 📁 Versioned data snapshots and model metadata
- 📦 `requirements.txt` for environment reproducibility

### The Streamlit dashboard supports:

![streamlit Dashboard](./plot_images/image.png)
- Live data upload (.pkl)

![alt text](./plot_images/image-1.png)

- Regime-aware routing

![Shock / Calm Regime count](./plot_images/image-2.png)

- Prediction visualization

![Predictions](./plot_images/image-3.png)

![alt text](./plot_images/image-4.png)

- SHAP interpretability

![alt text](./plot_images/image-5.png)

- Exportable results

![alt text](./plot_images/image-6.png)


All components are modular, audit-ready, and extensible for multi-asset support or fallback blending.

Version: `v1.0`  
Author: Shyam Subramani  
Date: 26 Sep 2025