# /src/predict_live.py
# This file uses the tuned best config to do live prediction.

from src.model_runner import run_wfv_regime_routed_ensemble
from src.evaluation import setup_for_eval_dashboard, plot_evaluation_dashboard
from src.diagnostics import setup_for_shap_importance_plot, build_shap_matrix, plot_residual_drift
from src.data_loader import crypto_data_ingestion, perform_data_preprocessing
from src.feature_engineering import add_liquidity_feature, build_features
from src.config import EVAL_DIR
from src.model_runner import run_live_prediction, train_models
import json
import os, pickle
import numpy as np

import requests
import pandas as pd
from datetime import datetime

coins = ['Bitcoin', 'Ethereum', 'XRP', 'Tether', 'BNB', 'Solana', 'USDC',
       'Dogecoin', 'Lido Staked Ether', 'TRON', 'Cardano',
       'Wrapped stETH', 'Chainlink', 'Wrapped Beacon ETH', 'Ethena USDe',
       'Wrapped Bitcoin', 'Avalanche', 'Figure Heloc', 'Hyperliquid',
       'Sui', 'Stellar', 'Bitcoin Cash', 'Wrapped eETH', 'WETH', 'Hedera',
       'LEO Token', 'USDS', 'Litecoin',
       'Binance Bridged USDT (BNB Smart Chain)', 'Toncoin', 'Shiba Inu',
       'Cronos', 'Coinbase Wrapped BTC', 'Polkadot', 'WhiteBIT Coin',
       'Ethena Staked USDe', 'Mantle', 'World Liberty Financial',
       'Monero', 'USDT0', 'Uniswap', 'Dai', 'Aave', 'MemeCore', 'Ethena',
       'Pepe', 'Aster', 'NEAR Protocol', 'OKB', 'Story', 'Bitget Token',
       'Jito Staked SOL', 'Aptos', 'Bittensor', 'Ondo',
       'Ethereum Classic', 'Worldcoin', 'Binance Staked SOL', 'USD1',
       'Binance-Peg WETH', 'POL (ex-MATIC)', 'Arbitrum',
       'Internet Computer', 'Pi Network', 'sUSDS',
       'Jupiter Perpetuals Liquidity Provider Token',
       'BlackRock USD Institutional Digital Liquidity Fund', 'Kaspa',
       'Pump.fun', 'Falcon USD', 'Flare', 'Gate', 'Cosmos Hub', 'KuCoin',
       'VeChain', 'Fasttoken', 'Render', 'Algorand', 'Pudgy Penguins',
       'Kelp DAO Restaked ETH', 'Rocket Pool ETH', 'Kinetiq Staked HYPE',
       'USDtb', 'PayPal USD', 'Sei', 'MYX Finance',
       'Provenance Blockchain', 'BFUSD', 'Sky', 'StakeWise Staked ETH',
       'Bonk', 'Artificial Superintelligence Alliance', 'Official Trump',
       'Filecoin', 'Lombard Staked BTC', 'Liquid Staked ETH', 'Jupiter',
       'Immutable', 'Polygon Bridged USDT (Polygon)', 'Quant']

def fetch_snapshot(coin_ids=['bitcoin', 'ethereum', 'tether'], vs_currency='usd'):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': vs_currency,
        # 'ids': ','.join(coin_ids),
        'price_change_percentage': '1h,24h,7d'
    }
    response = requests.get(url, params=params).json()

    records = []
    for coin in response:
        records.append({
            'coin': coin['name'],
            'symbol': coin['symbol'].upper(),
            'price': coin['current_price'],
            '1h': coin.get('price_change_percentage_1h_in_currency', 0),
            '24h': coin.get('price_change_percentage_24h_in_currency', 0),
            '7d': coin.get('price_change_percentage_7d_in_currency', 0),
            '24h_volume': coin['total_volume'],
            'mkt_cap': coin['market_cap'],
            'date': datetime.now().strftime('%Y-%m-%d')
        })

    df = pd.DataFrame(records)
    return df

# Change this flag to True for Training the model with a new dataset pkl file.
# Change this flat to False for Predicting the trained model with a live dataset. pkl.

training_only = True

def main():

    #Training - first time.
    if training_only == True:
        # Load historical data for training
        with open("./notebooks/crypto_training_data.pkl", "rb") as f:
            df_train = pickle.load(f)


        # print(df_train['regime_shock'].value_counts())

        # Train and save models
        print("Training models for live prediction...")
        train_models(df_train)
        print("Training regime split:")

        print("✅ Models saved: model_calm.pkl, model_shock.pkl, feature_cols.pkl")

    else:
        # Load real-time or unseen data
        # df = crypto_data_ingestion()
        # df_snapshot = fetch_snapshot()
        # Load the cleaned data
        print("Loading the pickle file.. ")
        with open("./notebooks/live_crypto_data_clean.pkl", "rb") as f:
            df_live = pickle.load(f)

        print("\ndf_live: \n",df_live.shape)

        print("Building features. ")
        df_live['symbol'] = df_live['coin']

        # # Feature Engineering
        df_live = add_liquidity_feature(df_live)
        print("df_live:\n",df_live.head(), df_live.info(), df_live.shape)

        # Optional: simulate shock regime for testing
        mask = df_live['coin'] == 'bitcoin'
        n = mask.sum()  # number of bitcoin rows
        half = n // 2

        shock_idx = df_live[mask].index[:half]
        calm_idx = df_live[mask].index[half:]


        df_live.loc[mask, 'volume_24h'] *= np.random.normal(1.5, 0.3, size=n)

        # Simulate shock regime
        df_live.loc[shock_idx, 'shock_z'] = np.random.normal(2.5, 0.5, size=len(shock_idx))
        df_live.loc[shock_idx, 'vol_ratio_7_30'] = np.random.normal(1.8, 0.4, size=len(shock_idx))

        # Simulate calm regime
        df_live.loc[calm_idx, 'shock_z'] = np.random.normal(0.5, 0.2, size=len(calm_idx))
        df_live.loc[calm_idx, 'vol_ratio_7_30'] = np.random.normal(0.8, 0.2, size=len(calm_idx))


        # df_live.loc[mask, 'shock_z'] = np.random.normal(2.5, 0.5, size=n)  # above shock threshold
        # df_live.loc[mask, 'vol_ratio_7_30'] = np.random.normal(1.8, 0.4, size=n)  # elevated volatility

        # Add price-scaled feature to guide model routing
        df_live.loc[mask, 'price_scaled'] = df_live.loc[mask, 'price'] / df_live['price'].max()

        # Build features
        df_feat = build_features(df_live, regime_aware=True)
        print("Shock regime count:", df_feat['regime_shock'].sum())
        print("Calm regime count:", (df_feat['regime_shock'] == 0).sum())

        df_feat = df_feat.reset_index(drop=False)
        
        df_feat = df_feat.dropna()
        print(df_feat[['price', 'volume_24h', 'vol_ratio_7_30', 'shock_z']].describe())
        print("df_feat shape:", df_feat.shape)
        print("Shock regime count:", df_feat['regime_shock'].sum())
    # print("df_feat:\n",df_feat.head(), df_feat.info(), df_feat.shape)
    
    

        df_pred = run_live_prediction(df_live)
        print("\n=== Live Predictions ===")
        
        print(df_pred[['date', 'coin', 'prediction']].head(10))
        
        import matplotlib.pyplot as plt
        plt.plot(df_pred['prediction'])
        plt.title("Live Predictions Over Time")
        plt.show()


    # y_true = fold_outputs[2]['y_true']
    # y_pred = fold_outputs[2]['y_pred']

    # residuals = np.array(y_true) - np.array(y_pred)

    # Inspect top errors
    # top_errors = np.argsort(np.abs(residuals))[::-1][:10]
    # for i in top_errors:
    #     print(f"Index {i}: True={y_true[i]:.2f}, Pred={y_pred[i]:.2f}, Residual={residuals[i]:.2f}")

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 4))
    # plt.hist(residuals, bins=50, color='steelblue', edgecolor='black')
    # plt.title("Fold 2 Residual Distribution")
    # plt.xlabel("Residual")
    # plt.ylabel("Frequency")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # import seaborn as sns
    # sns.boxplot(data=df_feat[['volume_24h', 'shock_z', 'vol_ratio_7_30']])
    # plt.show()
    # df = perform_data_preprocessing(df_snapshot)

    # # Feature Engineering
    # df = add_liquidity_feature(df)

    # # Load best config
    # with open( os.path.join(EVAL_DIR,"wfv_best_config_20250925_0202.json"), "r") as f:
    #     best_config = json.load(f)
    #     print("Best Config:", best_config)

    # # Run model
    # metrics, df_feat, primary_models, top_feats, fold_outputs = run_wfv_regime_routed_ensemble(
    #     df,
    #     fallback_weights={1: best_config[0], 2: best_config[0], 4: best_config[0]},
    #     residual_weight=best_config[1],
    #     top_n_features=int(best_config[2]),
    #     regime_aware=True,
    #     blend_with_lag=5,
    #     shap_lock=True,
    #     verbose=True
    # )

    # # Diagnostics
    # df_all, fold_metrics = setup_for_eval_dashboard(df_feat, fold_outputs)
    # fold_shap_dict = setup_for_shap_importance_plot(primary_models, top_feats)
    # shap_df = build_shap_matrix(df_feat, fold_shap_dict)

    # # Dashboard
    # plot_evaluation_dashboard(df_all, fold_metrics, shap_matrix=shap_df)
    # plot_residual_drift(df_all)

if __name__ == "__main__":
    main()