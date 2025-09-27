import pandas as pd
import pickle
from src.data_loader import perform_data_preprocessing
from src.feature_engineering import add_liquidity_feature, build_features

def convert_csv_to_pkl(input_csv_path, output_pkl_path):
    # Load raw CSV
    df_raw = pd.read_csv(input_csv_path)

    # Preprocess
    df_clean = perform_data_preprocessing(df_raw)
    df_clean = add_liquidity_feature(df_clean, safe_mode=True)

    # Build features
    df_feat = build_features(df_clean, regime_aware=True)

    # Save as .pkl
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(df_feat, f)

    print(f"✅ Saved preprocessed data to {output_pkl_path}")

# Example usage
if __name__ == "__main__":
    convert_csv_to_pkl("sample/crypto_sample_test.csv", "sample/crypto_sample_test.pkl")