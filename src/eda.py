# src/eda.py

import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_price_volume_trends(df, save_path=None):
    """
    Plots time series trends for price, volume, and market cap.
    Optionally saves the plot if save_path is provided.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    sns.lineplot(ax=axes[0], data=df, x='date', y='price')
    axes[0].set_title('Price Over Time')

    sns.lineplot(ax=axes[1], data=df, x='date', y='volume_24h')
    axes[1].set_title('24h Volume Over Time')

    sns.lineplot(ax=axes[2], data=df, x='date', y='mkt_cap')
    axes[2].set_title('Market Cap Over Time')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# 1. Univariate Analysis — Feature Distributions
def plot_univariate_distribution(df, feature, regime_col='regime_shock',save_path=None):
    plt.figure(figsize=(10, 5))
    sns.histplot(df[feature], kde=True, bins=50, color='steelblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# 2. Bivariate Analysis — Regime vs Feature Comparison.
def plot_regime_split(df1, feature, regime_col='regime_shock',save_path=None):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df1, x=feature, hue=regime_col, fill=True, common_norm=False, palette='Set2')
    plt.title(f'{feature} Distribution by Regime')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# 3. Multivariate Analysis — Correlation Matrix
def plot_correlation_matrix(df, features,save_path=None):
    corr = df[features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()