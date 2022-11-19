import pandas as pd
import numpy as np
import backtrader as bt
import akshare as ak
import requests

import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 75  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题

def calculateEMA(period, closeArray, emaArray=[]):
    length = len(closeArray)
    nanCounter = np.count_nonzero(np.isnan(closeArray))
    if not emaArray:
        emaArray.extend(np.tile([np.nan], (nanCounter + period - 1)))
        firstema = np.mean(closeArray[nanCounter:nanCounter + period - 1])
        emaArray.append(firstema)
        for i in range(nanCounter + period, length):
            ema = (2 * closeArray[i] + (period - 1) * emaArray[-1]) / (period + 1)
            emaArray.append(ema)
    return np.array(emaArray)


def calculateMACD(closeArray, shortPeriod=12, longPeriod=26, signalPeriod=9):
    ema12 = calculateEMA(shortPeriod, closeArray, [])
    ema26 = calculateEMA(longPeriod, closeArray, [])
    diff = ema12 - ema26

    dea = calculateEMA(signalPeriod, diff, [])
    macd = (diff - dea)

    fast_values = diff
    slow_values = dea
    diff_values = macd

    return fast_values, slow_values, diff_values

def get_atr(quote_df,N_input =100):
    def calc_tr(t_high,t_low,pre_close):
        tr = max(t_high -t_low, t_high - pre_close,pre_close - t_low)
        return tr
    tr_lst = []
    atr_lst = [ ]
    for i in range(len(quote_df)):
        se_i = quote_df.iloc[i]
        if i < N_input:
            N = i+1
            tr = calc_tr(se_i['high'],se_i['high'],se_i['pre_close'])
            tr_lst.append(tr)
            atr = sum(tr_lst)/N
            atr_lst.append(atr)
    return atr_lst

rate_simga = 2

def get_direction(dea_t,dif_t,break_value):
    if dif_t - dea_t >break_value:
        return 1
    if dif_t - dea_t < - break_value:
        return -1

def diff_integrate():
    pass





# 股指
quote_hs_300 = ak.stock_zh_index_daily('sh000300')
quote_hs_300['date'] = pd.to_datetime(quote_hs_300['date'])
quote_hs_300.set_index('date',inplace=True)
quote_hs_300['pre_close'] = quote_hs_300.close.shift(1)



def calc_error_integrate():
    pass

'''
https://tvc4.investing.com/8463b893927927345bc436a112d04e51/1668654369/6/6/28/history?symbol=166&resolution=D&from=1455160204&to=1459048201
'''
