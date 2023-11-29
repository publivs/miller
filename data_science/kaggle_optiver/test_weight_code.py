import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

data_path =r'/usr/src/kaggle_/optiver-trading-at-the-close'
path_train  = data_path+  '/train.csv'
data = pd.read_csv(path_train)

num_stocks = data["stock_id"].nunique()
num_dates = data["date_id"].nunique()
num_updates = data["seconds_in_bucket"].nunique()

print(f"# stocks         : {num_stocks}")
print(f"# dates          : {num_dates}")
print(f"# updates per day: {num_updates}")

stock_returns = np.zeros((num_stocks, num_dates, num_updates))
index_returns = np.zeros((num_stocks, num_dates, num_updates))
target_values = np.zeros((num_stocks, num_dates, num_updates))

# 学到了学到了,tqdm这个东西

weights = np.array([
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
        ])


# 太菜了，不会apply.
for (stock_id, date_id), frame in tqdm(data.groupby(["stock_id", "date_id"])):
    # frame["stock_return"] = ((frame["wap"] / frame["wap"].shift(6)).shift(-6) - 1) * 10_000 # 这段看不懂没关系
    frame["stock_return"] = np.log(frame["wap"] / frame["wap"].shift(6)).shift(-6) * 10_000
    frame["index_return"] = frame["stock_return"] - frame["target"]

    stock_returns[stock_id, date_id] = frame["stock_return"].values
    index_returns[stock_id, date_id] = frame["index_return"].values
    target_values[stock_id, date_id] = frame["target"].values
    
index_return = np.mean(index_returns, axis=0) # 这里发生了什么？？？

lr = LinearRegression()
y = index_return.reshape(-1)
X = stock_returns.reshape((num_stocks, -1)).T

mask = ~((np.isnan(y) | np.isnan(X).any(axis=1)))
X, y = X[mask], y[mask]

lr.fit(X, y)

print(" Fit ".center(80, ">"))
print("Coef:", lr.coef_)
print("Intercept:", lr.intercept_)
print("R2:", r2_score(y, lr.predict(X)))

lr.coef_ = lr.coef_.round(3)
lr.intercept_ = 0.0
print(" Round with 3 digits ".center(80, ">"))
print("Coef:", lr.coef_)
print("Sum of Coef:", lr.coef_.sum())
print("R2:", r2_score(y, lr.predict(X)))

# 
x_matrix = pd.DataFrame(X) # stocks

diff_values = x_matrix@weights - y # 这就是原target


y = y.copy() # indexs

target = frame["stock_return"] - frame["index_return"]