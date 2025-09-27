# src/evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import ROLLING_WINDOW, PLOT_DIR


def plot_residual_drift(df_all):
    df_all['residual'] = df_all['actual'] - df_all['predicted']
    df_all['rolling_mean'] = df_all['residual'].rolling(ROLLING_WINDOW).mean()
    df_all['rolling_std'] = df_all['residual'].rolling(ROLLING_WINDOW).std()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='timestamp', y='rolling_mean', data=df_all, label='Rolling Mean Residual')
    sns.lineplot(x='timestamp', y='rolling_std', data=df_all, label='Rolling Std Residual')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Residual Drift (window={ROLLING_WINDOW})')
    plt.xlabel('Timestamp')
    plt.ylabel('Residual')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_images/eval_residual_drift.png", dpi=300)
    plt.close()

def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': round(rmse, 4), 'mae': round(mae, 4), 'r2': round(r2, 4)}

def summarize_print_fold_metrics(metrics):
    avg_rmse = np.mean([m[1] for m in metrics])
    avg_mae = np.mean([m[2] for m in metrics])
    avg_r2 = np.mean([m[3] for m in metrics])
    print("\n=== WFV Summary ===")
    print(f"Avg RMSE: {avg_rmse:.4f}")
    print(f"Avg MAE:  {avg_mae:.4f}")
    print(f"Avg R²:   {avg_r2:.4f}")
    
# Setting up metrics per fold for Evaluation dashboard.
def setup_for_eval_dashboard(fold_outputs):
    """ Setting up metrics per fold for Evaluation dashboard.
        Returns: df_all with parameters for dashboard."""
    # for fold_id, result in fold_outputs.items():
    #     print(f"Fold {fold_id}: len(y_true)={len(result['y_true'])}, len(y_pred)={len(result['y_pred'])}, len(timestamp)={len(result.get('timestamp', []))}")
    #     print("result:",result)

    fold_metrics = {
        fold_id: {
            'rmse': result['rmse'],
            'mae': result['mae'],
            'r2': result['r2']
        }
        for fold_id, result in fold_outputs.items()
    }

    df_all = pd.concat([
        pd.DataFrame({
            'timestamp': result['timestamp'],
            'actual': result['y_true'],
            'predicted': result['y_pred'],
            'fold_id': fold_id
        })
        for fold_id, result in fold_outputs.items()
    ], ignore_index=True)
    return df_all, fold_metrics


# Full Model Evaluation Dashboard 

def plot_evaluation_dashboard(df, fold_metrics, shap_matrix=None, rolling_window=30, save_path = None):
    """
    df: DataFrame with columns ['timestamp', 'actual', 'predicted']
    fold_metrics: dict with fold_id → {'rmse': ..., 'mae': ..., 'mape': ...}
    shap_matrix: optional DataFrame (folds x features) of mean |SHAP| values
    rolling_window: int, window size for rolling error metrics
    """

    residuals = df['actual'] - df['predicted']
    df['residual'] = residuals

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Model Evaluation Dashboard", fontsize=16)

    # 1. Prediction vs Actual
    axs[0, 0].plot(df['timestamp'], df['actual'], label='Actual', alpha=0.7)
    axs[0, 0].plot(df['timestamp'], df['predicted'], label='Predicted', alpha=0.7)
    axs[0, 0].set_title("Predicted vs Actual")
    axs[0, 0].legend()

    # 2. Residual Distribution
    sns.histplot(residuals, bins=50, kde=True, ax=axs[0, 1], color='salmon')
    axs[0, 1].set_title("Residual Distribution")

    # 3. Residuals vs Predicted
    axs[1, 0].scatter(df['predicted'], residuals, alpha=0.5)
    axs[1, 0].axhline(0, color='gray', linestyle='--')
    axs[1, 0].set_title("Residuals vs Predicted")
    axs[1, 0].set_xlabel("Predicted")
    axs[1, 0].set_ylabel("Residual")

    # 4. Rolling Error Metrics
    df['rolling_rmse'] = residuals.rolling(rolling_window).apply(lambda x: np.sqrt(np.mean(x**2)))
    df['rolling_mae'] = residuals.rolling(rolling_window).apply(lambda x: np.mean(np.abs(x)))
    axs[1, 1].plot(df['timestamp'], df['rolling_rmse'], label='RMSE')
    axs[1, 1].plot(df['timestamp'], df['rolling_mae'], label='MAE')
    axs[1, 1].set_title(f"Rolling Error Metrics (window={rolling_window})")
    axs[1, 1].legend()

    # 5. Fold-wise Metric Bar Plot
    metric_df = pd.DataFrame(fold_metrics).T
    metric_df.plot(kind='bar', ax=axs[2, 0])
    axs[2, 0].set_title("Fold-wise Evaluation Metrics")
    axs[2, 0].set_xlabel("Fold ID")
    axs[2, 0].set_ylabel("Error")

    # 6. SHAP Heatmap (optional)
    if shap_matrix is not None:
        sns.heatmap(shap_matrix.T, cmap='viridis', ax=axs[2, 1], cbar_kws={'label': 'Mean |SHAP Value|'})
        axs[2, 1].set_title("SHAP Feature Importances Across Folds")
        axs[2, 1].set_xlabel("Fold ID")
        axs[2, 1].set_ylabel("Feature")
    else:
        axs[2, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()