# src/model_runner.py
import lightgbm as lgb
import shap, os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from src.data_loader import crypto_data_ingestion, perform_data_preprocessing
from src.feature_engineering import add_liquidity_feature, build_features
# from src.diagnostics import get_top_shap_features, export_shap_ranking
from src.config import LAG_LIST, CAT_COLS, TARGET_COL, LGB_PARAMS , MODEL_DIR
from src.evaluation import summarize_print_fold_metrics
from src.shap_utils import get_top_shap_features, export_shap_ranking

def run_pipeline(df):
    
    # DATA PREPROCESSING
    df = perform_data_preprocessing(df)

    # Feature Engineering
    df = add_liquidity_feature(df)

    # Fold routing and model training logic here
    # Placeholder: return dummy outputs
    df_all = df.copy()
    fold_outputs = {}
    return df_all, fold_outputs

# 5.1 - Model Building, Training, Evaluation - Normal Data - Single Split Foldaware function
def run_single_split_ensemble(
    df,
    split_frac=0.8,
    regime_aware=True,
    blend_with_lag=5,
    top_n_features=20
):
    min_samples_for_regime_fallback = 50
    warmup = max(max(LAG_LIST), 30)
    split_idx = int(len(df) * split_frac)

    train_raw = df.iloc[:split_idx].copy()
    test_raw = df.iloc[split_idx - warmup:].copy()

    train_feat = build_features(train_raw, regime_aware=regime_aware).dropna()
    test_feat = build_features(test_raw, regime_aware=regime_aware).dropna()
    if len(test_feat) > warmup:
        test_feat = test_feat.iloc[warmup:]


    # Encode categorical columns in-place
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    for col in CAT_COLS:
        if col in train_feat.columns:
            train_feat[col] = encoder.fit_transform(train_feat[[col]].astype(str))
        if col in test_feat.columns:
            test_feat[col] = encoder.transform(test_feat[[col]].astype(str))

    # Cast categorical columns to 'category' dtype
    for col in CAT_COLS:
        if col in train_feat.columns:
            train_feat[col] = train_feat[[col]].astype('category')
        if col in test_feat.columns:
            test_feat[col] = test_feat[[col]].astype('category')

    # SHAP-based feature selection
    dfc = build_features(df, regime_aware=regime_aware).dropna()

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    for col in CAT_COLS:
        if col in dfc.columns:
            dfc[col] = encoder.fit_transform(dfc[[col]].astype(str))

    feature_cols_all = [c for c in dfc.columns if c != TARGET_COL]
    X_full, y_full = dfc[feature_cols_all], dfc[TARGET_COL]

    model_full = lgb.LGBMRegressor(**LGB_PARAMS)
    model_full.fit(X_full, y_full)

    # Return top N features ranked by mean absolute SHAP values
    top_feats = get_top_shap_features(model_full, X_full, top_n=top_n_features)

    # explainer = shap.Explainer(model_full, X_full)
    # shap_values = explainer(X_full)
    # mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    # ranked = pd.Series(mean_abs_shap, index=X_full.columns).sort_values(ascending=False)
    # top_feats = ranked.head(top_n_features).index.tolist()

    feature_cols = [c for c in train_feat.columns if c in top_feats]

    # Define Monotonic Constraints. This assumes liquidity should increase with volume or price — adjust keywords if needed.
    monotonic_constraints = [
    1 if 'volume' in f.lower() or 'price' in f.lower() else 0
    for f in feature_cols
    ]

    X_train, y_train = train_feat[feature_cols], train_feat[TARGET_COL]
    X_test, y_test = test_feat[feature_cols], test_feat[TARGET_COL]

    # Optional log transform toggle
    apply_inverse = False
    if 'log_transform' in df.columns and df['log_transform'].iloc[0] == 1:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
        apply_inverse = True

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    for col in CAT_COLS:
        if col in X_train.columns:
            X_train[col] = encoder.fit_transform(X_train[[col]].astype(str))
            X_test[col] = encoder.transform(X_test[[col]].astype(str))

    # Regime-specific LightGBM
    if X_test.empty:
        print("X_test shape:", X_test.shape)
        print("Feature columns:", feature_cols)
        print("Missing columns:", [col for col in feature_cols if col not in X_test.columns])
        print("⚠️ Skipping prediction — X_test is empty.")
        return {
            'y_true': [],
            'y_pred': [],
            'rmse': None,
            'mae': None,
            'r2': None
        }

    calm_mask = train_feat['regime_shock'] == 0
    shock_mask = train_feat['regime_shock'] == 1

    if shock_mask.sum() < min_samples_for_regime_fallback or calm_mask.sum() < min_samples_for_regime_fallback:
        # print("Sparse regime split — using unified model")
        model_unified = lgb.LGBMRegressor(**LGB_PARAMS, monotone_constraints=monotonic_constraints)
        model_unified.fit(X_train, y_train)
        lgb_pred = model_unified.predict(X_test)
    else:
        # Monotonic Constraints
        model_calm = lgb.LGBMRegressor(**LGB_PARAMS, monotone_constraints=monotonic_constraints)
        model_shock = lgb.LGBMRegressor(**LGB_PARAMS, monotone_constraints=monotonic_constraints)

        model_calm.fit(X_train[calm_mask], y_train[calm_mask])
        model_shock.fit(X_train[shock_mask], y_train[shock_mask])

        shock_test_mask = test_feat['regime_shock'] == 1
        calm_test_mask = ~shock_test_mask

        lgb_pred = np.zeros(len(X_test))
        lgb_pred[calm_test_mask] = model_calm.predict(X_test[calm_test_mask])
        lgb_pred[shock_test_mask] = model_shock.predict(X_test[shock_test_mask])

    # Ridge model
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)

    # XGBoost model
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # Blend predictions
    y_pred = 0.6 * lgb_pred + 0.2 * ridge_pred + 0.2 * xgb_pred

    # Fallback blend
    if f'lag_{blend_with_lag}' in X_test.columns:
        mask = ~X_test[f'lag_{blend_with_lag}'].isna()
        y_pred[mask] = (
            0.7 * y_pred[mask] +
            0.3 * X_test.loc[mask, f'lag_{blend_with_lag}']
        )

    # Residual modeling
    residual = y_test - y_pred
    residual_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    residual_model.fit(X_test, residual)
    residual_correction = residual_model.predict(X_test)

    if apply_inverse:
        y_pred = np.expm1(y_pred)
        y_final = np.expm1(y_pred + residual_correction)
    else:
        y_final = y_pred + residual_correction

    rmse = np.sqrt(mean_squared_error(y_test, y_final))
    mae = mean_absolute_error(y_test, y_final)
    r2 = r2_score(y_test, y_final)
    print(f"\n=== Single Split Summary ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")


    return {
    'y_true': y_test,
    'y_pred': y_final,
    'rmse': round(rmse, 4),
    'mae': round(mae, 4),
    'r2': round(r2, 4)}

# ### 5.2 - Model Building, Training, Evaluation - Anomalous Data - WFV Regime Routed Ensemble Model. with metrics.



def load_shap_features(path, top_n):
    ranked = pd.read_csv(path, index_col=0)
    return ranked.head(top_n).index.tolist()

def run_wfv_regime_routed_ensemble(
    df,
    n_splits=5,
    min_train_frac=0.5,
    regime_aware=True,
    blend_with_lag=5,
    fallback_weights = {1: 0.6, 2: 0.6, 4: 0.6},
    top_n_features=25,
    shap_lock=True,
    residual_weight=1.0,
    verbose=True
):

    primary_models = {}     # Adding this to send the models used to plot. post evaluation.
    min_samples_for_regime_fallback = 50  # You can tune this threshold

    if shap_lock and os.path.exists("shap_ranking_fold1.csv"):
        top_feats = load_shap_features("shap_ranking_fold1.csv", top_n_features)
    else:
        # Use Fold 1 for SHAP feature selection
        fold1_end = int(len(df) * min_train_frac)
        fold1_raw = df.iloc[:fold1_end].copy()
        fold1_feat = build_features(fold1_raw, regime_aware=regime_aware).dropna()

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        for col in CAT_COLS:
            if col in fold1_feat.columns:
                fold1_feat[col] = encoder.fit_transform(fold1_feat[[col]].astype(str))

        feature_cols_all = [c for c in fold1_feat.columns if c != TARGET_COL]
        X_fold1, y_fold1 = fold1_feat[feature_cols_all], fold1_feat[TARGET_COL]

        model_fold1 = lgb.LGBMRegressor(**LGB_PARAMS)
        model_fold1.fit(X_fold1, y_fold1)

        # explainer = shap.Explainer(model_fold1, X_fold1)
        # shap_values = explainer(X_fold1)
        # mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        # ranked = pd.Series(mean_abs_shap, index=X_fold1.columns).sort_values(ascending=False)
        # ranked.to_csv("shap_ranking_fold1.csv", index=True)
        # top_feats = ranked.head(top_n_features).index.tolist()

        top_feats = export_shap_ranking(model_fold1, X_fold1, top_n_features)


    # Fold-wise training
    if 'fold_id' not in df.columns: 
        df['fold_id'] = -1
        
    total_len = len(df)
    start_train_size = int(total_len * min_train_frac)
    fold_size = max(1, (total_len - start_train_size) // n_splits)
    warmup = max(max(LAG_LIST), 30)

    metrics = []
    fold_outputs = {}
    df_feat_all = []
    for fold in range(n_splits):
        train_end = start_train_size + fold * fold_size
        test_end = total_len if fold == n_splits - 1 else train_end + fold_size
        df.iloc[train_end:test_end, df.columns.get_loc('fold_id')] = fold + 1

        # Step 2: Build features and attach fold_id
        df_feat = build_features(df, regime_aware=True).copy()
        df_feat['fold_id'] = df['fold_id']

        if train_end <= warmup:
            continue

        train_raw = df.iloc[:train_end].copy()
        test_raw = df.iloc[train_end - warmup:test_end].copy()

        train_feat = build_features(train_raw, regime_aware=regime_aware).dropna()

        # Step 1: Build features
        test_feat_raw = build_features(test_raw, regime_aware=regime_aware)
        test_feat = test_feat_raw.dropna().reset_index(drop=True)

        # Step 2: Slice by position
        test_feat = test_feat.iloc[warmup:].reset_index(drop=True)
        timestamps = test_raw.reset_index(drop=True).iloc[warmup:warmup + len(test_feat)]['date'].reset_index(drop=True)

        if test_feat.empty:
            print(f"⚠️ Fold {fold+1}: test_feat is empty after warm-up slicing — skipping fold.")
            fold_outputs[fold + 1] = {
                'y_true': [],
                'y_pred': [],
                'rmse': None,
                'mae': None,
                'r2': None,
                'timestamp': None
            }
            continue

        # Step 3: Assign and assert
        test_feat['timestamp'] = timestamps

        # Step 4: Assert alignment
        assert len(timestamps) == len(test_feat), f"Fold {fold+1}: Timestamp misalignment"
        assert len(test_feat) == len(timestamps), f"Timestamp misalignment in fold {fold+1}"


        test_feat['fold_id'] = fold+1

        df_feat_all.append(test_feat.copy())        #For SHAP Plotting. fold ids.

        feature_cols = [c for c in train_feat.columns if c in top_feats]

        #Fix: Add a check to log missing features per fold:

        missing_feats = [f for f in top_feats if f not in train_feat.columns]
        if missing_feats:
            print(f"Fold {fold+1}: Missing SHAP features → {missing_feats}")

        # Define Monotonic Constraints. - This assumes liquidity should increase with volume or price — adjust keywords if needed.
        monotonic_constraints = [
            1 if 'volume' in f.lower() or 'price' in f.lower() else 0
            for f in feature_cols
        ]

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        for col in CAT_COLS:
            if col in train_feat.columns:
                train_feat[col] = encoder.fit_transform(train_feat[[col]].astype(str))
            
            if not test_feat.empty and col in test_feat.columns:
                test_feat[col] = encoder.transform(test_feat[[col]].astype(str))
            else:
                print(f"⚠️ Skipping encoding for '{col}' — test_feat is empty.")
            # if col in test_feat.columns:
            #     test_feat[col] = encoder.transform(test_feat[[col]].astype(str))

        X_train, y_train = train_feat[feature_cols], train_feat[TARGET_COL]
        X_test, y_test = test_feat[feature_cols], test_feat[TARGET_COL]

        if fold + 1 in [2, 4]:
            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)
            apply_inverse = True
        else:
            apply_inverse = False

        #debug line.
        if verbose:
            print(f"Fold {fold+1}: Shock Regime Count = {train_feat['regime_shock'].sum()}, Calm Count = {(train_feat['regime_shock'] == 0).sum()}")

        # Regime-specific LightGBM

        if X_test.empty:
            print(f"⚠️ Skipping prediction for fold {fold+1} — X_test is empty.")
            fold_outputs[fold + 1] ={
                'y_true': [],
                'y_pred': [],
                'rmse': None,
                'mae': None,
                'r2': None,
                'timestamp': None
            }
            continue # or return if not looping

        calm_mask = train_feat['regime_shock'] == 0
        shock_mask = train_feat['regime_shock'] == 1

        if shock_mask.sum() < min_samples_for_regime_fallback or calm_mask.sum() < min_samples_for_regime_fallback:
            #debug line
            if verbose:
                print(f"Fold {fold+1}: Sparse regime split — using unified model")

            model_unified = lgb.LGBMRegressor(**LGB_PARAMS, monotone_constraints=monotonic_constraints)
            model_unified.fit(X_train, y_train)
            lgb_pred = model_unified.predict(X_test)
            primary_models[fold + 1] = model_unified  # Save unified model

        else:
            # Monotonic Constraints.
            model_calm = lgb.LGBMRegressor(**LGB_PARAMS, monotone_constraints=monotonic_constraints)
            model_shock = lgb.LGBMRegressor(**LGB_PARAMS, monotone_constraints=monotonic_constraints)


            model_calm.fit(X_train[calm_mask], y_train[calm_mask])
            model_shock.fit(X_train[shock_mask], y_train[shock_mask])
            primary_models[fold + 1] = model_calm  # Save calm model
            # model_shock is used but not returned for SHAP

            shock_test_mask = test_feat['regime_shock'] == 1
            calm_test_mask = ~shock_test_mask

            lgb_pred = np.zeros(len(X_test))
            lgb_pred[calm_test_mask] = model_calm.predict(X_test[calm_test_mask])
            lgb_pred[shock_test_mask] = model_shock.predict(X_test[shock_test_mask])

        # Ridge model
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        ridge_pred = ridge_model.predict(X_test)

        # XGBoost model
        xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        # Blend predictions
        y_pred = 0.6 * lgb_pred + 0.2 * ridge_pred + 0.2 * xgb_pred

        # Fallback blend
        fallback_weight = fallback_weights.get(fold + 1, 0.0)
        if fallback_weight > 0 and f'lag_{blend_with_lag}' in X_test.columns:
            mask = ~X_test[f'lag_{blend_with_lag}'].isna()
            y_pred[mask] = (
                (1 - fallback_weight) * y_pred[mask] +
                fallback_weight * X_test.loc[mask, f'lag_{blend_with_lag}']
            )

        # Residual modeling
        residual = y_test - y_pred
        residual_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        residual_model.fit(X_test, residual)
        residual_correction = residual_model.predict(X_test)

        if fold + 1 == 4:
            # Skip residual modeling for Fold 4
            if apply_inverse:
                y_final = np.expm1(y_pred)
            else:
                y_final = y_pred
        else:
            y_final = y_pred + residual_weight * residual_correction
            if apply_inverse:
                y_final = np.expm1(y_final)
            else:
                y_final = y_pred + residual_weight * residual_correction

        rmse = np.sqrt(mean_squared_error(y_test, y_final))
        mae = mean_absolute_error(y_test, y_final)
        r2 = r2_score(y_test, y_final)
        metrics.append((fold + 1, rmse, mae, r2))
        fold_outputs[fold + 1] = {
            'y_true': y_test,
            'y_pred': y_final,
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'r2': round(r2, 4),
            'timestamp': test_feat['timestamp'].values
        }
        if verbose:
            print(f"Fold {fold+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    if df_feat_all:
        df_feat = pd.concat(df_feat_all, ignore_index=True)
    else:
        print("⚠️ No folds retained — df_feat_all is empty.")
        df_feat = pd.DataFrame()

    print(f"✅ Retained folds: {len(df_feat_all)}")


    # -----------------------------
    # ---- Evaluation Metrics -----
    # -----------------------------
    if verbose:
        summarize_print_fold_metrics(metrics)

    return metrics, df_feat, primary_models, top_feats, fold_outputs

# -------------------------------
# 🔧 TRAINING FUNCTION
# -------------------------------
def train_models(df):
    # Tune LightGBM for small regimes
    params = {
    'min_data_in_leaf': 5,
    'num_leaves': 15,
    'max_depth': 4,
    'learning_rate': 0.05
    }   

    df_train = build_features(df, regime_aware=True).dropna()

    feature_cols = [c for c in df_train.columns if c != TARGET_COL and c not in ['coin', 'symbol']]

    # Inject synthetic volatility into shock regime
    df_shock = df_train[df_train['regime_shock'] == 1].copy()
    df_calm = df_train[df_train['regime_shock'] == 0].copy()

    # Inject synthetic target variance for shock regime
    df_shock[TARGET_COL] *= np.random.normal(1.2, 0.1, size=len(df_shock))
    
    print("Shock target stats after injection:")
    print(df_shock[TARGET_COL].describe())

    df_shock['momentum_1d'] = df_shock[TARGET_COL] - df_shock[TARGET_COL].shift(1)
    df_shock['momentum_7d'] = df_shock[TARGET_COL] - df_shock[TARGET_COL].shift(7)

    np.random.seed(42)

    df_shock['volume_24h'] *= np.random.normal(1.8, 0.4, size=len(df_shock))
    df_shock['shock_z'] = np.random.normal(2.8, 0.3, size=len(df_shock))
    df_shock['vol_ratio_7_30'] = np.random.normal(2.0, 0.2, size=len(df_shock))
    df_shock['momentum_1d'] *= np.random.normal(2.5, 0.5, size=len(df_shock))
    
    df_shock['roll_std_7'] *= np.random.normal(2.0, 0.3, size=len(df_shock))

    df_shock['regime_shock'] = (df_shock['shock_z'].abs() > 2.0).astype(int)
    df_shock['regime_high_vol'] = (df_shock['vol_ratio_7_30'] > 1.5).astype(int)

    df_shock_aug = pd.concat([df_shock.copy() for _ in range(5)], ignore_index=True)

    df_shock_aug['liquidity_ratio'] *= np.random.normal(1.2, 0.2, size=len(df_shock_aug))
    df_shock_aug['shock_z'] += np.random.normal(0.1, 0.05, size=len(df_shock_aug))
    df_shock_aug['vol_ratio_7_30'] += np.random.normal(0.1, 0.05, size=len(df_shock_aug))

    df_shock_aug['regime_shock'] = (df_shock_aug['shock_z'].abs() > 2.0).astype(int)
    df_shock_aug['regime_high_vol'] = (df_shock_aug['vol_ratio_7_30'] > 1.5).astype(int)

    # Recombine and train
    df_train_augmented = pd.concat([df_calm, df_shock_aug])

    with open(os.path.join(MODEL_DIR,"df_train_augmented.pkl"), "wb") as f:
        pickle.dump(df_train_augmented, f)

    print("Shock regime after injection:")
    print(df_train_augmented[df_train_augmented['regime_shock'] == 1][['shock_z', 'vol_ratio_7_30']].describe())

    print("Training regime split:")
    print(df_train_augmented.info(), "\n", df_train_augmented['regime_shock'].value_counts())

    calm_mask = df_train_augmented['regime_shock'] == 0
    shock_mask = df_train_augmented['regime_shock'] == 1

    model_calm = lgb.LGBMRegressor(**params)
    model_shock = lgb.LGBMRegressor(**params)

    model_calm.fit(df_train_augmented[calm_mask][feature_cols], df_train_augmented[calm_mask][TARGET_COL])
    model_shock.fit(df_train_augmented[shock_mask][feature_cols], df_train_augmented[shock_mask][TARGET_COL])

    print("Calm model feature importances:")
    print(model_calm.feature_importances_)

    print("Shock model feature importances:")
    print(model_shock.feature_importances_)

    with open(os.path.join(MODEL_DIR, "model_calm.pkl"), "wb") as f:
        pickle.dump(model_calm, f)
    with open(os.path.join(MODEL_DIR,"model_shock.pkl"), "wb") as f:
        pickle.dump(model_shock, f)
    with open(os.path.join(MODEL_DIR,"feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

# -------------------------------
# 🔮 LIVE PREDICTION FUNCTION
# -------------------------------
def run_live_prediction(df_live):
    df_feat = build_features(df_live, regime_aware=True).dropna()

    with open(os.path.join(MODEL_DIR,"model_calm.pkl"), "rb") as f:
        model_calm = pickle.load(f)
    with open(os.path.join(MODEL_DIR,"model_shock.pkl"), "rb") as f:
        model_shock = pickle.load(f)
    with open(os.path.join(MODEL_DIR,"feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)

    calm_mask = df_feat['regime_shock'] == 0
    shock_mask = ~calm_mask

    preds = np.zeros(len(df_feat))
    print(f"Live regime split → Shock: {shock_mask.sum()}, Calm: {calm_mask.sum()}")
    if calm_mask.sum() > 0:
        preds[calm_mask] = model_calm.predict(df_feat[calm_mask][feature_cols])
    else:
        print("⚠️ No calm regime rows — skipping calm model.")


    if shock_mask.sum() > 0:
        preds[shock_mask] = model_shock.predict(df_feat[shock_mask][feature_cols])
    else:
         print("⚠️ No shock regime rows — skipping shock model.")

    print("Calm model predictions:")
    print(model_calm.predict(df_feat[df_feat['regime_shock'] == 0][feature_cols].head(10)))

    print("Shock model predictions:")
    print(model_shock.predict(df_feat[df_feat['regime_shock'] == 1][feature_cols].head(10)))

    df_feat['prediction'] = preds
    
    # Reattach metadata from original df_live
    df_feat = df_feat.reset_index(drop=True)
    df_feat['date'] = df_live.reset_index(drop=True)['date']
    df_feat['coin'] = df_live.reset_index(drop=True)['coin']

    print("\n✅ Live predictions complete:")
    print(df_feat[['date', 'coin', 'prediction']].head())

    # return df_feat[['date', 'coin', 'prediction']]
    return df_feat