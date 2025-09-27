# src/shap_utils.py
import shap
import pandas as pd
import numpy as np

def get_top_shap_features(model, X, top_n=20):
    """
    Returns top N features ranked by mean absolute SHAP value.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranked = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
    return ranked.head(top_n).index.tolist()

def export_shap_ranking(model, X, save_path="shap_ranking_fold1.csv", top_n=25):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranked = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
    ranked.to_csv(save_path, index=True)
    return ranked.head(top_n).index.tolist()