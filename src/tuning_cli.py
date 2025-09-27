import argparse
import pandas as pd
from src.tuning import run_wfv_hyperparameter_tuning
from src.data_loader import crypto_data_ingestion, perform_data_preprocessing

def main():
    parser = argparse.ArgumentParser(description="Run WFV hyperparameter tuning")
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save best config JSON')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Load and preprocess data
    df = crypto_data_ingestion()
    df = perform_data_preprocessing(df)

    # Run tuning
    df_results, best_config = run_wfv_hyperparameter_tuning(df, save_dir=args.save_dir, verbose=args.verbose)

if __name__ == "__main__":
    main()