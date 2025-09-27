# src/diagnostics.py
import numpy as np
import pandas as pd
import shap, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from src.config import PLOT_DIR


def run_diagnostics(df_all, fold_outputs):
    # Add SHAP drift, fold-wise checks, etc.
    # mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    # ranked = pd.Series(mean_abs_shap, index=X_full.columns).sort_values(ascending=False)
    print("Diagnostics complete.")

# 4. Fold-Wise Feature Drift
def plot_feature_drift(df_feat, feature, fold_col='fold_id', save_path=None):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_feat, x=fold_col, y=feature, palette='Blues')
    plt.title(f'Distribution of {feature} Drift Across Folds')
    plt.xlabel('Fold')
    plt.ylabel(feature)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# Residual Drift Plot (Time Series)
def plot_residual_drift(df_all, window=30, save_path = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_all['residual'] = df_all['actual'] - df_all['predicted']
    df_all['rolling_mean'] = df_all['residual'].rolling(window).mean()
    df_all['rolling_std'] = df_all['residual'].rolling(window).std()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='timestamp', y='rolling_mean', data=df_all, label='Rolling Mean Residual')
    sns.lineplot(x='timestamp', y='rolling_std', data=df_all, label='Rolling Std Residual')
    
    sns.scatterplot(x='timestamp', y='residual', data=df_all, palette='Set2')
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Residual Drift (window={window})')
    plt.xlabel('Timestamp')
    plt.ylabel('Residual')
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def setup_for_shap_importance_plot(df_feat,primary_models, top_feats, show_plot=True):
    """This functions sets up the data for plotting SHAP importance"""

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    fold_shap_dict = {}
    fold_metrics = {}   # Fold metrics for evaluation dashboard.
    for fold_id, model in primary_models.items():
        fold_data = df_feat[df_feat['fold_id'] == fold_id].copy()

        # Encode coin and symbol if present
        for col in ['coin', 'symbol']:
            if col in fold_data.columns:
                fold_data[col] = encoder.fit_transform(fold_data[[col]].astype(str))

        # Reconstruct X_fold with exact top_feats
        expected_feats = top_feats.copy()
        X_fold = pd.DataFrame({
            f: fold_data[f] if f in fold_data.columns else np.zeros(len(fold_data))
            for f in expected_feats
        })

        # Drop rows with NaNs (optional)
        X_fold = X_fold.dropna()

        # Ensure numeric types
        X_fold = X_fold.astype(float)

        # Final shape check
        if X_fold.shape[1] != len(expected_feats):
            print(f"Fold {fold_id}: Skipping SHAP — feature mismatch ({X_fold.shape[1]} vs {len(expected_feats)})")
            continue

        if X_fold.empty:
            print(f"Fold {fold_id}: Skipping SHAP — no usable data.")
            continue

        # SHAP computation
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_fold)

        # Collecting SHAP values per fold 
        mean_shap = np.abs(shap_vals).mean(axis=0)
        fold_shap_dict[fold_id] = dict(zip(X_fold.columns, mean_shap))

        print(f"\nFold {fold_id} — SHAP Importance")
        # print(f"\nX_fold_shape: {X_fold.shape},\n expected_feats: {expected_feats}, \nshap_vals: {shap_vals.mean(axis=0)}\n")
        if show_plot:
            plot_shap_importance(shap_vals, expected_feats, fold_id, top_n=20, save_path=None)
        else:
            plot_shap_importance(shap_vals, expected_feats, fold_id, top_n=20, save_path=os.path.join(PLOT_DIR,f"eval_Top20ShapFeatureForFold{fold_id}.png"))
        
    return fold_shap_dict

def plot_shap_importance(shap_values, feature_names, fold_id, top_n=20, save_path = None):
    """ # 5. Plot SHAP Feature Importance (Optional Overlay)"""
    import numpy as np
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in top_idx], mean_abs_shap[top_idx], color='teal')
    plt.title(f'Top {top_n} SHAP Feature Importances - for fold {fold_id}')
    plt.xlabel('Mean |SHAP Value|')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def build_shap_matrix(df, fold_shap_dict,show_plot = True):
    """ Builds SHAP Matrix."""

    shap_df = pd.DataFrame(fold_shap_dict).fillna(0).T  # shape: (folds x features)
    shap_clean = shap_df.T.replace([np.inf, -np.inf], np.nan).fillna(0)
    save_path = None

    shap_stats = pd.DataFrame({
        'mean_importance': shap_df.mean(axis=0),
        'std_importance': shap_df.std(axis=0),
        'stability_score': shap_df.mean(axis=0) / (shap_df.std(axis=0) + 1e-6)
    }).sort_values(by='mean_importance', ascending=False)
    
    # Plotting SHAP Drift Across Folds
    if not show_plot:
        save_path = save_path=os.path.join(PLOT_DIR,"eval_ShapDriftAcrossFolds.png")
    plt.figure(figsize=(12, 6))
    shap_stats['std_importance'].sort_values(ascending=False).plot(kind='bar', title='SHAP Drift Across Folds')
    plt.ylabel('Std Dev of Mean |SHAP|')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
    
    # Plotting SHAP Feature Importance Across Folds
    save_path = None
    if not show_plot:
        save_path = save_path=os.path.join(PLOT_DIR,"eval_ShapFeatureImportanceHeatmap.png")
    plt.figure(figsize=(14, 6))
    sns.heatmap(shap_df.T, cmap='viridis', annot=True, fmt=".3f", cbar_kws={'label': 'Mean |SHAP Value|'})
    plt.title("SHAP Feature Importances Across Folds")
    plt.xlabel("Fold ID")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

    shap_clean = shap_df.T.copy()

    # Replace inf/-inf and fill NaNs
    shap_clean = shap_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Ensure all values are finite and numeric
    shap_clean = shap_clean.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Drop constant rows (zero variance across folds)
    shap_clean = shap_clean.loc[shap_clean.std(axis=1) > 0]

    return shap_df

