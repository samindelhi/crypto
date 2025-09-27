# src/tuning.py

from src.model_runner import run_wfv_regime_routed_ensemble
from src.config import EVAL_DIR
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def plot_grid_results(df_results, save_path=None):
    """Visualizes RMSE vs R² for grid search results."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_results, x='avg_rmse', y='avg_r2', hue='top_n_features', style='fallback_weight', palette='viridis', s=100)
    plt.title("Grid Search: RMSE vs R²")
    plt.xlabel("Average RMSE")
    plt.ylabel("Average R²")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def run_wfv_hyperparameter_tuning(df, save_dir=EVAL_DIR, verbose=True):
    """
    Runs grid search over WFV ensemble parameters and saves best config with timestamp.
    Also logs top 3 configs and plots RMSE vs R².
    """
    param_grid = {
        'fallback_weights': [0.4, 0.5, 0.6, 0.7],
        'residual_weight': [0.4, 0.6, 0.8, 1.0],
        'top_n_features': [15, 20, 25]
    }

    if verbose:
        print("🔍 Starting WFV hyperparameter tuning...")

    df_results = tune_wfv_grid(df, param_grid)

    # Timestamped filename
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = f"{save_dir}wfv_best_config_{date_str}.json"

    # Save best config
    best_config = df_results.iloc[0]
    best_config.to_json(save_path, orient="records")

    if verbose:
        print(f"\n✅ Best Config Saved to: {save_path}")
        print("\n📊 Top 3 Configs:")
        print(df_results.head(3))

        plot_grid_results(df_results, save_path=os.path.join(EVAL_DIR,"tuning_gridsearch_RMSEvsR2.png"))

    return df_results, best_config

def tune_wfv_grid(df, param_grid, regime_aware=True):
    results = []
    for fw in param_grid['fallback_weights']:
        for rw in param_grid['residual_weight']:
            for top_n in param_grid['top_n_features']:
                # print(f"\n🔍 Testing: fallback={fw}, residual={rw}, top_n={top_n}")
                fold_metrics, df_feat, primary_models, top_feats, fold_outputs  = run_wfv_regime_routed_ensemble(
                    df,
                    fallback_weights={1: fw, 2: fw, 4: fw},
                    top_n_features=top_n,
                    regime_aware=regime_aware,
                    shap_lock=True,
                    blend_with_lag=5,
                    residual_weight=rw,
                    verbose=False
                )
                avg_rmse = np.mean([m[1] for m in fold_metrics])
                avg_r2 = np.mean([m[3] for m in fold_metrics])
                results.append({
                    'fallback_weight': fw,
                    'residual_weight': rw,
                    'top_n_features': top_n,
                    'avg_rmse': avg_rmse,
                    'avg_r2': avg_r2
                })
    return pd.DataFrame(results).sort_values(by='avg_r2', ascending=False)