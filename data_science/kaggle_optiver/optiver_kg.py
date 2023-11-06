import pandas as pd 
import numpy as np 

import optiver2023

data_path = r'/usr/src/kaggle_/optiver-trading-at-the-close'
path_train  = data_path+  '/train.csv'
train_df = pd.read_csv(path_train)
 
def generate_features(df):

    features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
               'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2']
    
    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')
    
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

TRAINING = True

if TRAINING:
    df_train = pd.read_csv(path_train)
    df_ = generate_features(df_train)

import lightgbm as lgb 
import xgboost as xgb 
import catboost as cbt 
import numpy as np 
import joblib 
import os 

model_path ='/root/miller/data_science/kaggle_optiver/models/'

N_fold = 2

if TRAINING:
    X = df_.values
    Y = df_train['target'].values

    X = X[np.isfinite(Y)]
    Y = Y[np.isfinite(Y)]

    index = np.arange(len(X))

models = []


def train(model_dict, modelname='lgb'):
    if TRAINING:
        model = model_dict[modelname]
        model.fit(X[index%N_fold!=i],
                   Y[index%N_fold!=i], 
                    eval_set=[(X[index%N_fold==i], Y[index%N_fold==i])], 
                    )
        models.append(model)
        joblib.dump(model, f'{model_path}/{modelname}_{i}.model')
    else:
        models.append(joblib.load(f'{model_path}/{modelname}_{i}.model'))
    return 

model_dict = {
    'lgb': lgb.LGBMRegressor(objective='regression_l1', n_estimators=500,device = 'cuda'),
    'xgb': xgb.XGBRegressor(tree_method='hist', objective='reg:absoluteerror', n_estimators=500,gpu_id = 0,early_stopping_rounds=100),
    'cbt': cbt.CatBoostRegressor(objective='MAE', iterations=3000,early_stopping_rounds=100),
}

for i in range(N_fold):
    train(model_dict, 'lgb')
#     train(model_dict, 'xgb')
    train(model_dict, 'cbt')


env = optiver2023.make_env()
iter_test = env.iter_test()

counter = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    feat = generate_features(test)
    sample_prediction['target'] = np.mean([model.predict(feat) for model in models], 0)
    env.predict(sample_prediction)
    counter += 1