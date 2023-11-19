import gc
import os
import time
import warnings
from itertools import combinations
from warnings import simplefilter

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit
import xgboost as xgb 
import catboost as cbt 
import logging

logger = logging.getLogger('mylogger')

warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

is_offline = True
is_train = True
is_infer = True
max_lookback = np.nan
split_day = 435

data_path = r'/usr/src/kaggle_/optiver-trading-at-the-close'
path_train  = data_path+  '/train.csv'
train_df = pd.read_csv(path_train)
df = train_df.dropna(subset=["target"])
df.reset_index(drop=True, inplace=True)
df.shape

def reduce_mem_usage(df, verbose=0):

    """
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")

    return df

from numba import njit, prange

@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            if mid_val == min_val:  # Prevent division by zero
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)
    return imbalance_features

def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]

    # Calculate the triplet imbalance
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)

    return features

# generate features
def imbalance_features(df):
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    # V1
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["size_imbalance"] = df.eval("bid_size / ask_size")
    
    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
        
    # V2
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        
    # V3 shift_features
    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
            
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size']:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
    
    # V4 
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)


    return df.replace([np.inf, -np.inf], 0)


# generate time & stock features
def other_features(df):
    df["dow"] = df["date_id"] % 5
    df["seconds"] = df["seconds_in_bucket"] % 60
    df["minute"] = df["seconds_in_bucket"] // 60

    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())
    return df

# generate_rolling_features
def rolling_and_expanding_features(df,):

    window_size = [2, 3, 5, 10]
    
    # F_rolling
    rolling_features = ['bid_size', 'ask_size', 'bid_price', 'ask_price', 'imbalance_size', 'matched_size', 'wap']
    for window_size_i in window_size:
        for feature in rolling_features:
            df[f'{feature}_rolling_std_{window_size_i}'] = df.groupby('stock_id')[feature].transform(lambda x: x.rolling(window=window_size_i, min_periods=1).std())
            df[f'{feature}_rolling_median_{window_size_i}'] = df.groupby('stock_id')[feature].transform(lambda x: x.rolling(window=window_size_i, min_periods=1).median())
            

    # F_expanding calc_relative_delta
    for window_size_i in window_size:
        denominator_ = df['mid_price'].expanding(window_size_i).max() - df['mid_price'].expanding(window_size_i).min()
        df[f'{feature}_relativedelta_{window_size_i}_upside']  = ( df['mid_price'].expanding(window_size_i).max() - df['mid_price'])/denominator_
        df[f'{feature}_relativedelta_{window_size_i}_downside'] = ( df['mid_price'] - df['mid_price'].expanding(window_size_i).min())/denominator_

    return df


# generate all features
def generate_all_features(df):
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    df = imbalance_features(df)
    df = other_features(df)
    # df = rolling_and_expanding_features(df,)
    gc.collect()
    
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
    
    return df[feature_name]

def zero_sum(prices, volumes):
    std_error = np.sqrt(volumes)
    step = np.sum(prices)/np.sum(std_error)
    out = prices-std_error*step 
    return out

weights = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
    0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
    0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
    0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
    0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
    0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
    0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
    0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
    0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
    0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
    0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
    0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
    0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
    0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
    0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
    0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
]

weights = {int(k):v for k,v in enumerate(weights)}

if is_offline:
    df_train = df[df["date_id"] <= split_day]
    df_valid = df[df["date_id"] > split_day]
    print("Offline mode")
    print(f"train : {df_train.shape}, valid : {df_valid.shape}")
else:
    df_train = df
    print("Online mode")


if is_train:
    global_stock_id_feats = {
        "median_size": df_train.groupby("stock_id")["bid_size"].median() + df_train.groupby("stock_id")["ask_size"].median(),
        "std_size": df_train.groupby("stock_id")["bid_size"].std() + df_train.groupby("stock_id")["ask_size"].std(),
        "ptp_size": df_train.groupby("stock_id")["bid_size"].max() - df_train.groupby("stock_id")["bid_size"].min(),
        "median_price": df_train.groupby("stock_id")["bid_price"].median() + df_train.groupby("stock_id")["ask_price"].median(),
        "std_price": df_train.groupby("stock_id")["bid_price"].std() + df_train.groupby("stock_id")["ask_price"].std(),
        "ptp_price": df_train.groupby("stock_id")["bid_price"].max() - df_train.groupby("stock_id")["ask_price"].min(),
    }
    if is_offline:
        df_train_feats = generate_all_features(df_train)
        print("Build Train Feats Finished.")
        df_valid_feats = generate_all_features(df_valid)
        print("Build Valid Feats Finished.")
        df_valid_feats = reduce_mem_usage(df_valid_feats)
    else:
        df_train_feats = generate_all_features(df_train)
        print("Build Online Train Feats Finished.")

    df_train_feats = reduce_mem_usage(df_train_feats)

model_dict_list = [
    #         {
    #     'model': lgb.LGBMRegressor,
    #     'name': 'lgb',
    #     'params': {
    #         "objective": "mae",
    #         "n_estimators": 3000,
    #         "num_leaves": 128,
    #         "subsample": 0.6,
    #         "colsample_bytree": 0.6,
    #         "learning_rate": 0.05,
    #         "n_jobs": 12,
    #         "device": "cuda",
    #         "verbosity": 1,
    #         "importance_type": "gain",
    #         },
    #           "callbacks": [
    #             lgb.callback.early_stopping(stopping_rounds=100),
    #             lgb.callback.log_evaluation(period=100),]
    # },

    # {'model': xgb.XGBRegressor,
    #  'name':'xgb',
    #  'params': {
    #      'tree_method': 'gpu_hist',
    #      'objective': 'reg:absoluteerror',
    #      'n_estimators': 3000,
    #      "learning_rate": 0.05,
    #      'early_stopping_rounds': 100,
    #      'gpu_id': 0,
    #      "verbose": 1,
    #  }
    #       ,
    # "callbacks":None,
    # },

    {'model': cbt.CatBoostRegressor,
     'name':'lgb',
     'params': {
                'objective': 'RMSE',
                'iterations': 3000,
                "learning_rate": 0.025,
                "verbose": 1,
                'early_stopping_rounds': 100,
                'task_type':"GPU",
                'depth':8,
     },
    "callbacks":[
    ],
    }
]


for model_dict in model_dict_list:

    name = model_dict['name']
    print(f'now model is {name}')
    model_ = model_dict['model']
    model_params = model_dict['params']
    call_back_func  = model_dict['callbacks']
    if is_train:
        feature_name = list(df_train_feats.columns)
        print(f"Feature length = {len(feature_name)}")
        offline_split = df_train['date_id']>(split_day - 45)
        df_offline_train = df_train_feats[~offline_split].copy(deep = True)
        df_offline_valid = df_train_feats[offline_split].copy(deep = True)
        df_offline_train_target = df_train['target'][~offline_split].copy(deep = True)
        df_offline_valid_target = df_train['target'][offline_split].copy(deep = True)

        print("Valid Model Trainning.")
        _model = model_(**model_params)
        if name == 'lgb':
            _model.fit(
                df_offline_train[feature_name],
                df_offline_train_target,
                eval_set=[(df_offline_valid[feature_name], df_offline_valid_target)],
                callbacks = call_back_func 
                
            )
        else:
            _model.fit(
                df_offline_train[feature_name],
                df_offline_train_target,
                eval_set=[(df_offline_valid[feature_name], df_offline_valid_target)],
            )
        
        del df_offline_train, df_offline_train_target
        gc.collect()

        # infer
        df_train_target = df_train["target"]
        print("Infer Model Trainning.")
        infer_params = model_params.copy()
        best_iter_n = _model.best_iteration_ if hasattr(_model, "best_iteration_") else _model.best_iteration
        infer_params["n_estimators"] = int(1.2 * best_iter_n)
        infer__model =  model_(**model_params)
        infer__model.fit(df_train_feats[feature_name], df_train_target,
                         eval_set=[(df_offline_valid[feature_name], df_offline_valid_target)],
                                     callbacks = call_back_func 
                         )

        if is_offline:   
            # offline predictions
            df_valid_target = df_valid["target"]
            offline_predictions = infer__model.predict(df_valid_feats[feature_name])
            offline_score = mean_absolute_error(offline_predictions, df_valid_target)
            print(f"Offline Score {np.round(offline_score, 4)}")


    if is_infer:
        import optiver2023
        env = optiver2023.make_env()
        iter_test = env.iter_test()
        counter = 0
        y_min, y_max = -64, 64
        qps, predictions = [], []
        cache = pd.DataFrame()
        for (test, revealed_targets, sample_prediction) in iter_test:
            now_time = time.time()
            cache = pd.concat([cache, test], ignore_index=True, axis=0)
            if counter > 0:
                cache = cache.groupby(['stock_id']).tail(21).sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id']).reset_index(drop=True)
            feat = generate_all_features(cache)[-len(test):]
            lgb_prediction = infer__model.predict(feat)
            lgb_prediction = zero_sum(lgb_prediction, test['bid_size'] + test['ask_size'])
            clipped_predictions = np.clip(lgb_prediction, y_min, y_max)
            sample_prediction['target'] = clipped_predictions
            env.predict(sample_prediction)
            counter += 1
            qps.append(time.time() - now_time)
            if counter % 10 == 0:
                print(counter, 'qps:', np.mean(qps))
            
        time_cost = 1.146 * np.mean(qps)
        print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")