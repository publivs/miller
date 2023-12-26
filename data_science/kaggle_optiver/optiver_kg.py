# %%
import gc  
import os  
import time  
import warnings 
from itertools import combinations  
from warnings import simplefilter 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

import lightgbm as lgb  
import catboost as cbt 
import xgboost as xgb

from numba import njit, prange
import numpy as np  
import pandas as pd
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import KFold, TimeSeriesSplit
# from catboost import  EShapCalcType, EFeaturesSelectionAlgorithm # 特征筛选器
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import cudf

from itertools import combinations 

import logging
logger = logging.getLogger('mylogger')

# %%
# -------------------------- #
# 计算调试开关
is_offline = True
# -------------------------- #

is_train = True
is_infer = True 
max_lookback = np.nan 
split_day = 435 
lgb_accelerator = 'cuda' if is_offline else 'gpu'
model_target_col = 'target'

weights = np.array(
                    [
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
                    )

weights = {int(k):v for k,v in enumerate(weights)}

if is_offline:
    data_path =r'/usr/src/kaggle_/optiver-trading-at-the-close'
else:
    # data_path =r'/usr/src/kaggle_/optiver-trading-at-the-close'
    data_path = r'/kaggle/input/optiver-trading-at-the-close'
path_train  = data_path+  '/train.csv'

df_train = pd.read_csv(path_train,
                       engine='pyarrow',
                        # dtype_backend="pyarrow"
                        )

#  生成股票的子预测
import numba as nb
from numba import prange

# df_train["stock_return"] = np.log(df_train.groupby(["stock_id", "date_id"])["wap"].transform(lambda x: x / x.shift(6))).shift(-6) * 10_000
# df_train['index_return']=df_train["stock_return"] - df_train["target"]
# df_train = df_train.dropna(subset= ['index_return'])

print("stocks returns generate finished!.")
df = df_train.dropna(subset=["target"])
df.reset_index(drop=True, inplace=True)
df.shape
print('Data Loaded!')

# %%

def generate_features(df):

    features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price','far_price', 'near_price', 'ask_price', 
                'bid_price', 'wap','imb_s1', 'imb_s2']
    
    df['imb_s1'] = df.eval('(bid_size-ask_size)')/df.eval('(bid_size+ask_size)')
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)')/df.eval('(matched_size+imbalance_size)')
    
    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            if i>j:
                df[f'{a}_{b}_imb'] = df.eval(f'({a}-{b})/({a}+{b})')
                features.append(f'{a}_{b}_imb')

    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            for k,c in enumerate(prices):
                if i>j and j>k:
                    max_ = df[[a,b,c]].max(axis=1)
                    min_ = df[[a,b,c]].min(axis=1)
                    mid_ = df[[a,b,c]].sum(axis=1)-min_-max_
                    df[f'{a}_{b}_{c}_imb2'] = (max_-mid_)/(mid_-min_)
                    features.append(f'{a}_{b}_{c}_imb2')
    
    return df[features]

def reduce_mem_usage(df, verbose=0):
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
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)
    return imbalance_features

def calculate_triplet_imbalance_numba(price, df):
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
    features_array = compute_triplet_imbalance(df_values, comb_indices)
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)
    return features

def imbalance_features(df):
    df = cudf.from_pandas(df)
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price)") / 2
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)")/df.eval("(bid_size+ask_size)")

    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)")/df.eval("(matched_size+imbalance_size)")

    df["size_imbalance"] = df["bid_size"] / df["ask_size"]

    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] =  df[f"{c[0]}_{c[1]}_imb"] = np.divide(df[c[0]] - df[c[1]], df[c[0]] + df[c[1]])
    
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    
    # Calculate various statistical aggregation features

    for func in ["mean", "std", "skew", "kurt"]:
        if func == 'mean':
            df[f"all_prices_{func}"] = df[prices].mean( axis=1)
            df[f"all_sizes_{func}"] = df[sizes].mean( axis=1)
        elif func == 'std':
            df[f"all_prices_{func}"] = df[prices].std( axis=1)
            df[f"all_sizes_{func}"] = df[sizes].std( axis=1)
        elif func == 'skew':
            df[f"all_prices_{func}"] = df[prices].to_pandas().skew( axis=1)
            df[f"all_sizes_{func}"] = df[sizes].to_pandas().skew( axis=1)
        elif func == 'kurt':
            df[f"all_prices_{func}"] = df[prices].to_pandas().kurt( axis=1)
            df[f"all_sizes_{func}"] = df[sizes].to_pandas().kurt( axis=1)
        

    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 7]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
    
    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'market_urgency', 'imbalance_momentum', 'size_imbalance']:
        for window in [1, 2, 3, 7]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

    # V5
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'market_urgency', 'imbalance_momentum', 'size_imbalance']:
        for window in [1, 2, 3, 7]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
    
    return df.to_pandas().replace([np.inf, -np.inf], 0)

def calc_tri_features(df):
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
    df[triplet_feature.columns] = triplet_feature.values
    return df

def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  
    df["minute"] = df["seconds_in_bucket"] // 60  
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())
    return df

def rolling_features(df,):
    window_size = [ 3, 5, 7]
    # F_rolling
    rolling_features = [ 'mid_price','imbalance_size', 'matched_size', 'wap']
    for window_size_i in window_size:
        for feature in rolling_features:
            df[f'{feature}_rolling_std_{window_size_i}'] = df.groupby('stock_id')[feature].transform(lambda x: x.rolling(window=window_size_i, min_periods=1).std(engine='numba'))
            df[f'{feature}_rolling_median_{window_size_i}'] = df.groupby('stock_id')[feature].transform(lambda x: x.rolling(window=window_size_i, min_periods=1).median(engine='numba'))
    return df

def relativedelta_features(df):
    # F_expanding calc_relative_delta
    window_size = [2,3,5,7]
    rolling_features = [ 'bid_price', 'ask_price', 'imbalance_size', 'matched_size', 'wap']
    for window_size_i in window_size:
        for feature in rolling_features:
            denominator_ = df['mid_price'].expanding(window_size_i).max(engine='numba') - df['mid_price'].expanding(window_size_i).min(engine='numba')
            df[f'{feature}_relativedelta_{window_size_i}_upside']  = ( df['mid_price'].expanding(window_size_i).max(engine='numba') - df['mid_price'])/denominator_
            df[f'{feature}_relativedelta_{window_size_i}_downside'] = ( df['mid_price'] - df['mid_price'].expanding(window_size_i).min(engine='numba'))/denominator_
    return df


@njit(parallel = True)
def calculate_rsi(prices, period=14):
    rsi_values = np.zeros_like(prices)

    for col in prange(prices.shape[1]):
        price_data = prices[:, col]
        delta = np.zeros_like(price_data)
        delta[1:] = price_data[1:] - price_data[:-1]
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
        else:
            rs = 1e-9  # or any other appropriate default value
            
        rsi_values[:period, col] = 100 - (100 / (1 + rs))

        for i in prange(period-1, len(price_data)-1):
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
            if avg_loss != 0:
                rs = avg_gain / avg_loss
            else:
                rs = 1e-9  # or any other appropriate default value
            rsi_values[i+1, col] = 100 - (100 / (1 + rs))
    return rsi_values

@njit(parallel=True)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    rows, cols = data.shape
    macd_values = np.empty((rows, cols))
    signal_line_values = np.empty((rows, cols))
    histogram_values = np.empty((rows, cols))

    for i in prange(cols):
        short_ema = np.zeros(rows)
        long_ema = np.zeros(rows)

        for j in range(1, rows):
            short_ema[j] = (data[j, i] - short_ema[j - 1]) * (2 / (short_window + 1)) + short_ema[j - 1]
            long_ema[j] = (data[j, i] - long_ema[j - 1]) * (2 / (long_window + 1)) + long_ema[j - 1]

        macd_values[:, i] = short_ema - long_ema

        signal_line = np.zeros(rows)
        for j in range(1, rows):
            signal_line[j] = (macd_values[j, i] - signal_line[j - 1]) * (2 / (signal_window + 1)) + signal_line[j - 1]

        signal_line_values[:, i] = signal_line
        histogram_values[:, i] = macd_values[:, i] - signal_line

    return macd_values, signal_line_values, histogram_values

@njit(parallel=True)
def calculate_bband(data, window=20, num_std_dev=2):
    num_rows, num_cols = data.shape
    upper_bands = np.zeros_like(data)
    lower_bands = np.zeros_like(data)
    mid_bands = np.zeros_like(data)

    for col in prange(num_cols):
        for i in prange(window - 1, num_rows):
            window_slice = data[i - window + 1 : i + 1, col]
            mid_bands[i, col] = np.mean(window_slice)
            std_dev = np.std(window_slice)
            upper_bands[i, col] = mid_bands[i, col] + num_std_dev * std_dev
            lower_bands[i, col] = mid_bands[i, col] - num_std_dev * std_dev

    return upper_bands, mid_bands, lower_bands

def process_stock(stock_id, values):
    new_df = pd.DataFrame()
    new_df.index = values.index
    # RSI
    col_rsi = [f'rsi_{col}' for col in values.columns]
    rsi_values = calculate_rsi(values.values)
    new_df[col_rsi] = rsi_values
    gc.collect()

    # MACD
    macd_values, signal_line_values, histogram_values = calculate_macd(values.values)
    col_macd = [f'macd_{col}' for col in values.columns]
    col_signal = [f'macd_sig_{col}' for col in values.columns]
    col_hist = [f'macd_hist_{col}' for col in values.columns]
    new_df[col_macd] = macd_values
    new_df[col_signal] = signal_line_values
    new_df[col_hist] = histogram_values
    gc.collect()

    # Bollinger Bands
    bband_upper_values, bband_mid_values, bband_lower_values = calculate_bband(values.values, window=20, num_std_dev=2)
    col_bband_upper = [f'bband_upper_{col}' for col in values.columns]
    col_bband_mid = [f'bband_mid_{col}' for col in values.columns]
    col_bband_lower = [f'bband_lower_{col}' for col in values.columns]
    new_df[col_bband_upper] = bband_upper_values
    new_df[col_bband_mid] = bband_mid_values
    new_df[col_bband_lower] = bband_lower_values
    return new_df

def add_TA_features(df):

    prices = ["reference_price","wap","far_price","near_price"]
    from concurrent.futures import ProcessPoolExecutor
    result_lst = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        # 提交每个任务到进程池
        futures = [executor.submit(process_stock, stock_id, group) for stock_id, group in df.groupby('stock_id')[prices]]

        # 获取任务的结果
        for future in futures:
            result_lst.append(future.result())

    # 将结果汇总
    ta_index_df = pd.concat(result_lst)
    res_df = pd.merge(ta_index_df, df, left_index=True, right_index=True)
    return res_df

def generate_all_features(df):
    # Select relevant columns for feature generation
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    # Generate imbalance features

    df = imbalance_features(df)
    df = reduce_mem_usage(df)

    df =  calc_tri_features(df)
    df = reduce_mem_usage(df)

    df = other_features(df)
    df = reduce_mem_usage(df)

    df = rolling_features(df)
    df = reduce_mem_usage(df)

    df = relativedelta_features(df)
    df = reduce_mem_usage(df)

    df = add_TA_features(df)
    df = reduce_mem_usage(df)

    gc.collect()  
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
    return df[feature_name]

def select_features(df,method = 'corr',select_ratio = 0.8):
    def corr_feature_selection(df,n_components):
        corr_se = df.corr().abs().sum()
        correlated_features  = corr_se.sort_values().iloc[int(np.round(n_components)):].index
        df_selected = df.drop(correlated_features,axis=1)
        return df_selected
    if method == "corr":
        k = len(df.columns)*select_ratio
        df = corr_feature_selection(df, k)
        return df
    elif method == 'no':
        return df

print('Feature function Loaded!')

# %%

if is_offline:
    df_train = df[df["date_id"] <= split_day]
    df_valid = df[df["date_id"] > split_day]
    print("Offline mode")
    print(f"train : {df_train.shape}, valid : {df_valid.shape}")
else:
    df_train = df
    print("Online mode")

if is_train:
    stock_group = df_train.groupby("stock_id")
    global_stock_id_feats = {
            "median_size": stock_group["bid_size"].median() + stock_group["ask_size"].median(),
            "std_size": stock_group["bid_size"].std() + stock_group["ask_size"].std(),
            "ptp_size": stock_group["bid_size"].max() - stock_group["bid_size"].min(),
            "median_price": stock_group["bid_price"].median() + stock_group["ask_price"].median(),
            "std_price": stock_group["bid_price"].std() + stock_group["ask_price"].std(),
            "ptp_price": stock_group["bid_price"].max() - stock_group["ask_price"].min(),
            "median_far_price": stock_group["far_price"].median(),
            "median_near_price": stock_group["near_price"].median(),
            "median_imbalance_size_for_buy_sell": stock_group["imbalance_size_for_buy_sell"].median(),
            "matched_size":stock_group["matched_size"].median(),
        }
    if is_offline:
        df_train_feats = generate_all_features(df_train)
        print("Build Train Feats Finished.")

        df_valid_feats = generate_all_features(df_valid)
        print("Build Valid Feats Finished.")
        df_valid_feats = reduce_mem_usage(df_valid_feats)

        df_valid_feats = select_features(df_valid_feats)
        target_col = df_valid_feats.columns
        df_train_feats = df_train_feats[target_col]
    else:
        df_train_feats = generate_all_features(df_train)
        df_train_feats = select_features(df_train_feats)
        target_col = df_train_feats.columns
        df_train_feats = df_train_feats[target_col]
        print("Build Online Train Feats Finished.")

    df_train_feats = reduce_mem_usage(df_train_feats).ffill().bfill()

print('Processing of all features in the dataframe (df) is completed!')

# %%

model_dict_list = [
                {
        'model': lgb.LGBMRegressor,
        'name': 'lgb',
        "params":{
        "objective": "mae",
        "n_estimators": 3000,
        "num_leaves": 128,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "learning_rate": 0.00871,
        'max_depth': 11,
        "n_jobs": 4,
        "device": lgb_accelerator,
        "verbosity": 1,
        "importance_type": "gain",
        "early_stopping_rounds":100,
        "max_bin":128,
        'verbose':1
        }
        ,
        "callbacks": [
        lgb.callback.log_evaluation(period=100),
        ]
    },

    # {
    # "model":LinearRegression,
    # "name":"linear",
    # "params":{
    #     'fit_intercept': True,
    #     'n_jobs':4
    #         }
    # }

    # {'model':xgb.XGBRegressor,
    #  "name":"xgb",
    # "params":{
    
    # 'tree_method'        : 'hist', 
    # 'device'             : 'cuda',
    # 'objective'          : 'reg:absoluteerror',
    # 'random_state'       : 42,
    # 'colsample_bytree'   : 0.7,
    # 'learning_rate'      : 0.07,
    # 'max_depth'          : 6,
    # 'n_estimators'       : 3500,                         
    # 'reg_alpha'          : 0.025,
    # 'reg_lambda'         : 1.75,
    # 'min_child_weight'   : 1000,
    # "n_jobs": 4,
    # 'early_stopping_rounds': 100,  # 设置早停的轮数
    # },

    # },

    # {
    # 'model': cbt.CatBoostRegressor,
    # 'name':'catboost',
    # 'params': {
    #                    'task_type'           : "CPU",
    #                    'objective'           : "MAE",
    #                    'eval_metric'         : "MAE",
    #                    'bagging_temperature' : 0.5,
    #                    'colsample_bylevel'   : 0.7,
    #                    'learning_rate'       : 0.065,
    #                    'od_wait'             : 25,
    #                    'max_depth'           : 7,
    #                    'l2_leaf_reg'         : 1.5,
    #                    'min_data_in_leaf'    : 1000,
    #                    'random_strength'     : 0.65, 
    #                    'verbose'             : 0
    # },
    # # "callbacks":[
    # # ],
    # }

]


# voting_regressor = VotingRegressor(
#     estimators=[(model_dict['name'],model_dict['model'](**model_dict['params']))for model_dict  in model_dict_list],
#     verbose=1)

# voting_regressor.fit(df_train_feats, df_train['target'])

print('Params Loaded!')



# %%



feature_name = list(df_train_feats.columns)
print(f"Feature length = {len(feature_name)}")

def zero_sum(prices, volumes):
    std_error = np.sqrt(volumes)
    step = np.sum(prices)/np.sum(std_error)
    out = prices-std_error*step 
    return out

for model_dict in model_dict_list:

    name = model_dict['name']
    print(f'now model is {name}')
    model_ = model_dict['model']
    model_params = model_dict['params']
    # call_back_func  = model_dict['callbacks']
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
                callbacks =  [lgb.callback.log_evaluation(period=100)]
            )
        elif name == 'linear':
                        _model.fit(
                df_offline_train[feature_name],
                df_offline_train_target,
            )
        else:
            _model.fit(
                df_offline_train[feature_name],
                df_offline_train_target,
                eval_set=[(df_offline_valid[feature_name], df_offline_valid_target)],
            )
        
        # del df_offline_train, df_offline_train_target
        gc.collect()

        # infer
        df_train_target = df_train["target"]
        print("Infer Model Trainning.")
        infer_params = model_params.copy()
        if hasattr(_model, "n_estimators"):
            best_iter_n = _model.best_iteration_ if hasattr(_model, "best_iteration_") else _model.best_iteration
            infer_params["n_estimators"] = int(1.2 * best_iter_n)
        infer__model =  model_(**model_params)
        if name == 'lgb':
            infer__model.fit(df_train_feats[feature_name], df_train_target,
                            eval_set=[(df_offline_valid[feature_name], df_offline_valid_target)],
                            callbacks =  [lgb.callback.log_evaluation(period=100)]
                            )
            A = pd.DataFrame({'feats':infer__model.feature_name_,'importance':infer__model.feature_importances_})
            B = A.loc[A.importance>1]

            selected_features = B.head(62)['feats'].tolist()
            target_col = feature_name = selected_features
            # 使用选择的特征重新训练模型
            model_selected_features = lgb.LGBMRegressor(**model_params)
            infer__model.fit(df_train_feats[selected_features], df_train_target,
                            eval_set=[(df_offline_valid[selected_features], df_offline_valid_target)],
                            callbacks =  [lgb.callback.log_evaluation(period=100)]
                            )
        elif name == 'linear':
            infer__model.fit(df_train_feats[feature_name], df_train_target,
                            )
        else:
            infer__model.fit(df_train_feats[feature_name], df_train_target,
                            eval_set=[(df_offline_valid[feature_name], df_offline_valid_target)],
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
            if target_col.__len__() >0:
                feat = feat[target_col].fillna(0)
            model_prediction = infer__model.predict(feat)
            model_prediction = zero_sum(model_prediction, test['bid_size'] + test['ask_size'])
            clipped_predictions = np.clip(model_prediction, y_min, y_max)
            sample_prediction['target'] = clipped_predictions
            env.predict(sample_prediction)
            counter += 1
            qps.append(time.time() - now_time)
            if counter % 10 == 0:
                print(counter, 'qps:', np.mean(qps))
            
        time_cost = 1.146 * np.mean(qps)
        print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")
 




# %%
