import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from src.model_runner import build_features, run_live_prediction
from src.feature_engineering import add_liquidity_feature

from src.data_loader import perform_data_preprocessing
from src.feature_engineering import build_features, add_liquidity_feature

import warnings
warnings.filterwarnings('ignore')


def preprocess_for_prediction(df_raw):
    df_clean = perform_data_preprocessing(df_raw)
    df_clean = add_liquidity_feature(df_clean, safe_mode=True)
    df_feat = build_features(df_clean)

    # Ensure shock_z is present
    if 'shock_z' not in df_feat.columns:
        df_feat['shock_z'] = 0.0

    # Tag regime
    df_feat['regime'] = df_feat['shock_z'].apply(lambda z: 'shock' if z > 1.5 else 'calm')
    return df_feat


# Load models and feature columns
with open("models/model_calm.pkl", "rb") as f:
    model_calm = pickle.load(f)
with open("models/model_shock.pkl", "rb") as f:
    model_shock = pickle.load(f)
with open("models/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Title
st.title("Crypto Liquidity Forecast")

# Upload live data
uploaded_file = st.file_uploader("Upload live crypto data (.csv or .pkl)", type=["csv", "pkl"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
        df_live = preprocess_for_prediction(df_raw)
    else:
        df_live = pickle.load(uploaded_file)

    st.success("✅ File uploaded. Running full pipeline…")

    st.write("📊 Live Raw Data Preview", df_live.head())

     # Add symbol column
    if 'symbol' not in df_live.columns and 'coin' in df_live.columns:
        df_live['symbol'] = df_live['coin']


    # Add liquidity features
    df_live = add_liquidity_feature(df_live, safe_mode=True)

    # Simulate regime flags for Bitcoin
    # mask = df_live['coin'] == 'bitcoin'
    # n = mask.sum()
    # half = n // 2
    # shock_idx = df_live[mask].index[:half]
    # calm_idx = df_live[mask].index[half:]

    # df_live.loc[mask, 'volume_24h'] *= np.random.normal(1.5, 0.3, size=n)
    # df_live.loc[shock_idx, 'shock_z'] = np.random.normal(2.5, 0.5, size=len(shock_idx))
    # df_live.loc[shock_idx, 'vol_ratio_7_30'] = np.random.normal(1.8, 0.4, size=len(shock_idx))
    # df_live.loc[calm_idx, 'shock_z'] = np.random.normal(0.5, 0.2, size=len(calm_idx))
    # df_live.loc[calm_idx, 'vol_ratio_7_30'] = np.random.normal(0.8, 0.2, size=len(calm_idx))
    # df_live.loc[mask, 'price_scaled'] = df_live.loc[mask, 'price'] / df_live['price'].max()

    # Build features
    df_feat = build_features(df_live, regime_aware=True).dropna().reset_index(drop=False)
    st.write("📈 Feature Summary", df_feat[['price', 'volume_24h', 'vol_ratio_7_30', 'shock_z']].describe())
    st.write(f"Shock regime count: {df_feat['regime_shock'].sum()}")
    st.write(f"Calm regime count: {(df_feat['regime_shock'] == 0).sum()}")

    # Log Columns Present in df_feat
    st.write("🧪 Columns in df_feat:", df_feat.columns.tolist())


    if df_feat.empty or df_feat[feature_cols].dropna().empty:
        st.error("🚫 Feature set is empty or incomplete. Check preprocessing steps.")
        st.stop()


    # 2. Check for Missing Features
    missing = [col for col in feature_cols if col not in df_feat.columns]
    if missing:
        st.error(f"🚫 Missing required features: {missing}")
        st.stop()

    # Run predictions
    df_pred = run_live_prediction(df_feat)
    st.write("✅ Predictions", df_pred[['date', 'coin', 'prediction']].head(10))
    st.line_chart(df_pred['prediction'])

    # 1. Prediction Distribution by Regime

    

    st.subheader("📊 Prediction Distribution by Regime")

    fig, ax = plt.subplots()
    df_pred.boxplot(column='prediction', by='regime_shock', ax=ax)
    ax.set_title("Prediction Spread: Calm vs Shock")
    ax.set_xlabel("Regime (0 = Calm, 1 = Shock)")
    ax.set_ylabel("Predicted Liquidity Ratio")
    st.pyplot(fig)

    st.subheader("📈 Live Predictions Over Time")

    #2. Time Series of Prediction

    fig2, ax2 = plt.subplots()
    df_pred_sorted = df_pred.sort_values("date")
    ax2.plot(df_pred_sorted["date"], df_pred_sorted["prediction"], label="Prediction")
    ax2.set_title("Live Predictions Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Liquidity Ratio")
    ax2.legend()
    st.pyplot(fig2)

    #3. Feature Importance (Static or Precomputed)
    st.subheader("🧠 Top Features by Importance")

    top_feats = {
        "roll_std_7": 0.18,
        "momentum_1d": 0.14,
        "pct_change_24h": 0.12,
        "shock_z": 0.11,
        "vol_ratio_7_30": 0.10
    }

    feat_df = pd.DataFrame.from_dict(top_feats, orient='index', columns=['importance']).sort_values(by='importance')
    st.bar_chart(feat_df)

    # 4. SHAP Summary Plot (Optional)
    import shap
    import matplotlib.pyplot as plt

    st.subheader("🔬 SHAP Summary Plot")

    # Create SHAP explainer and values
    explainer = shap.TreeExplainer(model_calm)
    shap_values = explainer.shap_values(df_feat[feature_cols])

    # Generate SHAP summary plot into a matplotlib figure
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df_feat[feature_cols], plot_type="violin", show=False)
    st.pyplot(fig)

    #1. Regime Overlay on Time Series Plot
    st.subheader("📈 Regime Overlay on Predictions")

    fig, ax = plt.subplots()
    df_pred_sorted = df_pred.sort_values("date")
    colors = df_pred_sorted['regime_shock'].map({0: 'blue', 1: 'red'})
    ax.scatter(df_pred_sorted["date"], df_pred_sorted["prediction"], c=colors, label="Prediction", alpha=0.6)
    ax.set_title("Live Predictions with Regime Overlay")
    ax.set_xlabel("Date")
    ax.legend()
    ax.set_ylabel("Liquidity Ratio")
    st.pyplot(fig)    

    # 2. Download Button for Prediction
    st.subheader("📤 Export Predictions")

    csv = df_pred.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='live_predictions.csv',
        mime='text/csv'
    )

    # 3. Toggle Between Calm and Shock Model

    st.subheader("🔘 Manual Model Toggle")

    selected_model = st.radio("Choose model for preview:", ["Calm", "Shock"])
    if selected_model == "Calm":
        df_preview = df_feat[df_feat['regime_shock'] == 0]
        preds = model_calm.predict(df_preview[feature_cols])
    else:
        df_preview = df_feat[df_feat['regime_shock'] == 1]
        preds = model_shock.predict(df_preview[feature_cols])

    df_preview = df_preview.copy()
    df_preview['manual_prediction'] = preds
    st.write(df_preview[['coin', 'manual_prediction']].head(10))

    # 4. Model Routing Summary
    st.subheader("🧠 Model Routing Summary")

    routing_counts = df_feat['regime_shock'].value_counts().rename({0: 'Calm Model', 1: 'Shock Model'})
    st.bar_chart(routing_counts)