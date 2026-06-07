import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──────────────────────────────────────────────────────────────
train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOG TRANSFORM THE TARGET
# demand values are mostly small (0.0 to 0.3), skewed right
# log1p makes the distribution more normal → models learn better
# We reverse it at the end with expm1()
# ══════════════════════════════════════════════════════════════
train['demand_raw'] = train['demand']
train['demand']     = np.log1p(train['demand'])   # log transform

# ── Feature Engineering ────────────────────────────────────────────────────
def add_features(df):
    df['hour']       = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute']     = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute']/ 60)
    df['is_peak']    = (df['hour'].between(7,9) | df['hour'].between(16,19)).astype(int)
    df['is_night']   = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_weekend'] = (df['day'] % 7 >= 5).astype(int)
    df['day_of_week']= df['day'] % 7
    df['geo4']       = df['geohash'].str[:4]
    df['geo5']       = df['geohash'].str[:5]
    df['geo6']       = df['geohash'].str[:6]
    df['quarter'] = df['minute'] // 15
    df['time_slot'] = df['hour'] * 4 + df['quarter']

    df['slot_sin'] = np.sin(2*np.pi*df['time_slot']/96)
    df['slot_cos'] = np.cos(2*np.pi*df['time_slot']/96)
    df['geo_len']       = df['geohash'].str.len()
    df['road_enc']   = df['RoadType'].map({'Residential':0,'Street':1,'Highway':2}).fillna(0)
    df['weather_enc']= df['Weather'].map({'Sunny':0,'Rainy':1,'Foggy':2,'Snowy':3}).fillna(-1)
    df['largev_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['landmark_enc']=(df['Landmarks'] == 'Yes').astype(int)
    df['weekend_peak'] = df['is_weekend'] * df['is_peak']

    df['temp_cold'] = (df['Temperature'] < 15).astype(int)
    df['temp_hot'] = (df['Temperature'] > 30).astype(int)

    df['geo_hour'] = df['geohash'] + "_" + df['hour'].astype(str)
    df['Temperature']= df['Temperature'].fillna(df['Temperature'].median())
    df['temp_high']  = (df['Temperature'] > 30).astype(int)
    df['temp_cold'] = (df['Temperature'] < 15).astype(int)
    df['temp_x_peak'] = df['Temperature'] * df['is_peak']
    df['temp_x_night']= df['Temperature'] * df['is_night']
    df['weekend_x_peak']= df['is_weekend'] * df['is_peak']
    df['temp_x_hw']   = df['Temperature'] * (df['road_enc'] == 2).astype(int)
    df['peak_x_hw']  = df['is_peak'] * (df['road_enc'] == 2).astype(int)
    df['hour_x_lanes']= df['hour'] * df['NumberofLanes']
    df['lanes_x_peak']= df['NumberofLanes'] * df['is_peak']
    df['night_x_hw'] = df['is_night'] * (df['road_enc'] == 2).astype(int)
    return df

train = add_features(train)
test  = add_features(test)

GLOBAL_MEAN = train['demand'].mean()   # mean of LOG-transformed demand
kf = KFold(n_splits=10, shuffle=True, random_state=42)

train['geo_te']      = GLOBAL_MEAN
train['geo_hour_te'] = GLOBAL_MEAN
train['geo4_te']     = GLOBAL_MEAN
train['geo5_te']     = GLOBAL_MEAN
train['geo6_te']     = GLOBAL_MEAN

for tr_idx, val_idx in kf.split(train):
    tr  = train.iloc[tr_idx]
    val = train.iloc[val_idx]

    geo_mean      = tr.groupby('geohash')['demand'].mean()
    geo_hour_mean = tr.groupby(['geohash','hour'])['demand'].mean()
    geo4_mean     = tr.groupby('geo4')['demand'].mean()
    geo5_mean     = tr.groupby('geo5')['demand'].mean()
    geo6_mean     = tr.groupby('geo6')['demand'].mean()

    train.loc[train.index[val_idx], 'geo_te']  = val['geohash'].map(geo_mean).fillna(GLOBAL_MEAN)
    train.loc[train.index[val_idx], 'geo_hour_te'] = val.apply(
        lambda r: geo_hour_mean.get((r['geohash'], r['hour']),
                  geo_mean.get(r['geohash'], GLOBAL_MEAN)), axis=1)
    train.loc[train.index[val_idx], 'geo4_te'] = val['geo4'].map(geo4_mean).fillna(GLOBAL_MEAN)
    train.loc[train.index[val_idx], 'geo5_te'] = val['geo5'].map(geo5_mean).fillna(GLOBAL_MEAN)
    train.loc[train.index[val_idx], 'geo6_te'] = val['geo6'].map(geo6_mean).fillna(GLOBAL_MEAN)

geo_mean_full      = train.groupby('geohash')['demand'].mean()
geo_hour_mean_full = train.groupby(['geohash','hour'])['demand'].mean()
geo4_mean_full     = train.groupby('geo4')['demand'].mean()
geo5_mean_full     = train.groupby('geo5')['demand'].mean()
geo6_mean_full     = train.groupby('geo6')['demand'].mean()

test['geo_te']  = test['geohash'].map(geo_mean_full).fillna(GLOBAL_MEAN)
test['geo_hour_te'] = test.apply(
    lambda r: geo_hour_mean_full.get((r['geohash'], r['hour']),
              geo_mean_full.get(r['geohash'], GLOBAL_MEAN)), axis=1)
test['geo4_te'] = test['geo4'].map(geo4_mean_full).fillna(GLOBAL_MEAN)
test['geo5_te'] = test['geo5'].map(geo5_mean_full).fillna(GLOBAL_MEAN)
test['geo6_te'] = test['geo6'].map(geo6_mean_full).fillna(GLOBAL_MEAN)

# Hour aggregate
hour_stats = train.groupby('hour')['demand'].agg(['mean','std']).reset_index()
hour_stats.columns = ['hour','hour_mean','hour_std']
train = train.merge(hour_stats, on='hour', how='left')
test  = test.merge(hour_stats, on='hour', how='left')
#train['is_weekend'] = (train['dayofweek'] >= 5).astype(int)
#test['is_weekend'] = (test['dayofweek'] >= 5).astype(int)

train['rush_hour'] = train['hour'].isin([8,9,17,18]).astype(int)
test['rush_hour'] = test['hour'].isin([8,9,17,18]).astype(int)

train['night'] = train['hour'].isin([0,1,2,3,4,5]).astype(int)
test['night'] = test['hour'].isin([0,1,2,3,4,5]).astype(int)

rt_hour = train.groupby(['RoadType','hour'])['demand'].mean()
train['rt_hour_te'] = train.apply(lambda r: rt_hour.get((r['RoadType'], r['hour']), GLOBAL_MEAN), axis=1)
test['rt_hour_te']  = test.apply(lambda r: rt_hour.get((r['RoadType'], r['hour']), GLOBAL_MEAN), axis=1)

FEATURES = [
    'hour','minute','hour_sin','hour_cos',
    'is_peak','is_night','is_weekend','day','day_of_week',
    'geo_te','geo_hour_te','geo4_te','geo5_te','geo6_te',
    'road_enc','NumberofLanes','weather_enc',
    'largev_enc','landmark_enc',
    'Temperature','temp_high',
    'temp_cold', 'temp_hot',
    'weekend_x_peak',
    'quarter', 'time_slot',
    'slot_sin', 'slot_cos',
    'peak_x_hw','hour_x_lanes','lanes_x_peak','night_x_hw',
    'rt_hour_te','hour_mean','hour_std', 'rush_hour','night'
]

X      = train[FEATURES]
y      = train['demand']          # log-transformed
y_raw  = train['demand_raw']      # original (for final R² check)
X_test = test[FEATURES]

# ══════════════════════════════════════════════════════════════
# STEP 2 — LIGHTGBM (same as before)
# ══════════════════════════════════════════════════════════════
print("Training LightGBM...")

lgb_params = {
    'objective':        'regression',
    'metric':           'rmse',
    'num_leaves':       300,
    'learning_rate':    0.015,
    'n_estimators':     7000,
    'subsample':        0.85,
    'colsample_bytree': 0.85,
    'min_child_samples':10,
    'reg_alpha':        0.03,
    'reg_lambda':       0.05,
    'random_state':     42,
    'verbose':         -1,
}

oof_lgb  = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)]
    )
    oof_lgb[val_idx] = model.predict(X.iloc[val_idx])
    pred_lgb        += model.predict(X_test) / 10
    print(f"  LGB Fold {fold+1} R²: {r2_score(y.iloc[val_idx], oof_lgb[val_idx]):.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — CATBOOST (handles categoricals natively, less tuning)
# ══════════════════════════════════════════════════════════════
print("\nTraining CatBoost...")

oof_cat  = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=500,
        eval_metric='RMSE',
        early_stopping_rounds=100,
    )
    model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
    )
    oof_cat[val_idx] = model.predict(X.iloc[val_idx])
    pred_cat        += model.predict(X_test) / 10
    print(f"  CAT Fold {fold+1} R²: {r2_score(y.iloc[val_idx], oof_cat[val_idx]):.4f}")

print("\nTraining XGBoost...")

oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 3000,
    'learning_rate': 0.03,
    'max_depth': 8,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.03,
    'reg_lambda': 0.05,
    'random_state': 42,
    'tree_method': 'hist'
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):

    model = XGBRegressor(**xgb_params)

    model.fit(
        X.iloc[tr_idx],
        y.iloc[tr_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        verbose=False
    )

    oof_xgb[val_idx] = model.predict(X.iloc[val_idx])
    pred_xgb += model.predict(X_test) / kf.n_splits

    print(
        f"XGB Fold {fold+1} R²: "
        f"{r2_score(y.iloc[val_idx], oof_xgb[val_idx]):.4f}"
    )
# ══════════════════════════════════════════════════════════════
# STEP 5 -Blend three models: LGB, CAT, XGB (weights can be tuned further)
# ══════════════════════════════════════════════════════════════
oof_blend = (
    0.30 * oof_lgb +
    0.45 * oof_cat +
    0.25 * oof_xgb
)

pred_blend = (
    0.30 * pred_lgb +
    0.45 * pred_cat +
    0.25 * pred_xgb
)

# ══════════════════════════════════════════════════════════════
# STEP 5 — REVERSE LOG TRANSFORM
# We trained on log1p(demand), so predictions are in log space.
# expm1() reverses it back to original demand values.
# ══════════════════════════════════════════════════════════════
oof_final  = np.expm1(oof_blend)
pred_final = np.expm1(pred_blend)

# Check scores
lgb_r2   = r2_score(y_raw, np.expm1(oof_lgb))
cat_r2   = r2_score(y_raw, np.expm1(oof_cat))
xgb_r2   = r2_score(y_raw, np.expm1(oof_xgb))
blend_r2 = r2_score(y_raw, oof_final)

print(f"\nLightGBM  OOF R²: {lgb_r2:.4f}")
print(f"CatBoost  OOF R²: {cat_r2:.4f}")
print(f"XGBoost OOF R²: {xgb_r2:.4f}")
print(f"Blend     OOF R²: {blend_r2:.4f}  ← use this")

# ── Save Submission ────────────────────────────────────────────────────────
submission = pd.DataFrame({
    'Index':  test['Index'].values,
    'demand': pred_final

})
print(pred_final.min(), pred_final.max())
print(submission.head(10))
print(submission['demand'].describe())
print(submission['demand'].min(), submission['demand'].max())
submission.to_csv('submission_blend.csv', index=False)
print("\nSaved: submission_blend.csv")


# ══════════════════════════════════════════════════════════════
# BONUS — STEP 6: OPTUNA HYPERPARAMETER TUNING (run separately)
# Uncomment and run this block alone after the above works.
# It tries 50 different param combinations and picks the best.
# ══════════════════════════════════════════════════════════════

# import optuna
# optuna.logging.set_verbosity(optuna.logging.WARNING)
#
# def objective(trial):
#     params = {
#         'objective':        'regression',
#         'metric':           'rmse',
#         'verbose':          -1,
#         'num_leaves':       trial.suggest_int('num_leaves', 50, 300),
#         'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'n_estimators':     2000,
#         'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'min_child_samples':trial.suggest_int('min_child_samples', 10, 50),
#         'reg_alpha':        trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
#         'reg_lambda':       trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
#         'random_state':     42,
#     }
#     oof = np.zeros(len(train))
#     for tr_idx, val_idx in kf.split(X):
#         m = lgb.LGBMRegressor(**params)
#         m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
#               eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
#               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
#         oof[val_idx] = m.predict(X.iloc[val_idx])
#     return r2_score(y, oof)
#
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)
# print("Best params:", study.best_params)
# print("Best R²:", study.best_value)


