# src/main.py
import sys
import os
from src.utils import suppress_warnings
import matplotlib.pyplot as plt

from src.data_loader import crypto_data_ingestion, perform_data_preprocessing
from src.feature_engineering import add_liquidity_feature, build_features
from src.model_runner import run_pipeline, run_single_split_ensemble, run_wfv_regime_routed_ensemble
from src.eda import plot_price_volume_trends, plot_univariate_distribution, plot_regime_split, plot_correlation_matrix
from src.diagnostics import setup_for_shap_importance_plot, plot_feature_drift, plot_residual_drift, build_shap_matrix
from src.evaluation import setup_for_eval_dashboard, plot_evaluation_dashboard
from src.config import PLOT_DIR
from src.tuning import run_wfv_hyperparameter_tuning

# from evaluation import generate_dashboard
# from diagnostics import run_diagnostics

if __name__ == "__main__":
    suppress_warnings()
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # DATA INGESTION - Ingests data from the two csv datasets and concats them into one df.
    df = crypto_data_ingestion()

    # Runs the piple with loaded data
    df_all, fold_outputs = run_pipeline(df)

    print("------Inside Main()-------")
    print(df_all.info(), df_all.head())

    # Building Features for EDA. Univariate, bivariate and multivariate analysis.
    df_feat = build_features(df_all, regime_aware=True)
    df_feat = df_feat.reset_index(drop=True)

    #EDA 
    # 0. Time Series Trends.
    plot_price_volume_trends(df_all, save_path=os.path.join(PLOT_DIR,"eda_price_volume_trends.png"))

    # 1. Univariate Analysis — Feature Distributions
    plot_univariate_distribution(df_feat, 'shock_z', save_path=os.path.join(PLOT_DIR,"eda_Univariate_shock_z.png"))
    plot_univariate_distribution(df_feat, 'vol_ratio_7_30', save_path=os.path.join(PLOT_DIR,"eda_Univariate_vol_ratio.png"))
    plot_univariate_distribution(df_feat, 'momentum_1d', save_path=os.path.join(PLOT_DIR,"eda_Univariate_momentum.png"))

    # 2. Bivariate Analysis — Regime vs Feature Comparison.    
    plot_regime_split(df_feat, 'shock_z', save_path=os.path.join(PLOT_DIR,"eda_shockzDist_by_regime.png"))
    plot_regime_split(df_feat, 'momentum_1d', save_path=os.path.join(PLOT_DIR,"eda_momentumDist_by_regime.png"))
    plot_regime_split(df_feat, 'vol_ratio_7_30', save_path=os.path.join(PLOT_DIR,"eda_vol_ratio_Dist_by_regime.png"))

    # 3. Multivariate Analysis — Correlation Matrix
    selected_features = ['shock_z', 'momentum_1d', 'vol_ratio_7_30', 'lag_1', 'std7_volflag']
    plot_correlation_matrix(df_feat, selected_features, save_path=os.path.join(PLOT_DIR,"eda_correlation_matrix.png"))


    # 4 Run Models
    print("Model - Single Split Ensemble Run Started.....")
    single_split_result = run_single_split_ensemble(
        df_all,
        regime_aware=True,
        blend_with_lag=5,
        top_n_features=20)
    print("Model - Single Split Ensemble Run ...Ended")
    print("Model - Walk Forward Validation Ensemble Started.....")
    metrics, df_feat, primary_models, top_feats, fold_outputs = run_wfv_regime_routed_ensemble(
        df_all,
        regime_aware=True,
        blend_with_lag=5,
        top_n_features=20)
    print("Model - Walk Forward Validation Ensemble .....Ended.")

    # Diagnostics
    print("Plotting Diagnostics...")
    plot_feature_drift(df_feat,'shock_z', save_path=os.path.join(PLOT_DIR,"eval_shock_zDriftAcrossFolds.png"))
    plot_feature_drift(df_feat, 'momentum_1d', save_path=os.path.join(PLOT_DIR,"eval_momentumDriftAcrossFolds.png"))
    plot_feature_drift(df_feat, 'vol_ratio_7_30', save_path=os.path.join(PLOT_DIR,"eval_vol_ratioDriftAcrossFolds.png"))
    print("Plotting Diagnostics...Ended.")

    # Plots eval_Top20ShapFeatureForFold{Fold_id}.png inside.
    print("Plotting eval_Top20ShapFeatureForFolds. Started....")
    fold_shap_dict = setup_for_shap_importance_plot(df_feat=df_feat,primary_models=primary_models,top_feats=top_feats,show_plot=False)
    print("Plotting eval_Top20ShapFeatureForFolds... Ended")

    print("Plotting eval_ShapDriftAcrossFolds.png & eval_ShapFeatureImportanceHeatmap.png Started....")
    #plots "eval_ShapDriftAcrossFolds.png" & eval_ShapFeatureImportanceHeatmap.png" 
    shap_df = build_shap_matrix(df_feat, fold_shap_dict=fold_shap_dict, show_plot=False)
    print("Plotting eval_ShapDriftAcrossFolds.png & eval_ShapFeatureImportanceHeatmap.png ...... Ended")

    # Evaluation
    print("Plotting Evaluation Dashboard.... Started")
    df_metrics, fold_metrics = setup_for_eval_dashboard(fold_outputs)
    plot_evaluation_dashboard(df_metrics, fold_metrics=fold_metrics, shap_matrix=shap_df, save_path=os.path.join(PLOT_DIR,"eval_ModelEvaluationDashboard.png"))
    print("Plotting Evaluation Dashboard.... Ended")
    print("Plotting Residual Drift.... Started")
    plot_residual_drift(df_metrics, save_path=os.path.join(PLOT_DIR,"eval_ResidualDrift.png"))
    print("Plotting Residual Drift.... Started")
    # Hyperparameter Tuning
    
    df_results, best_config = run_wfv_hyperparameter_tuning(df_all)


    