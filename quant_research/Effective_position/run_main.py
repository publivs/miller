from Approximation import (Approximation, Mask_dir_peak_valley,
                                          Except_dir, Mask_status_peak_valley,
                                          Relative_values)

from performance import Strategy_performance
from collections import (defaultdict, namedtuple)
from typing import (List, Tuple, Dict, Union, Callable, Any)

import datetime as dt
import empyrical as ep
import numpy as np
import pandas as pd
import talib
import scipy.stats as st
from IPython.display import display

from sklearn.pipeline import Pipeline

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

import akshare as ak
# 画图

# 原文章
# https://www.joinquant.com/view/community/detail/f5d05b8233169adbbf44fb7522b2bf53?type=1&page=1

def plot_pivots(peak_valley_df: pd.DataFrame,

                show_dir: Union[str,List,Tuple]='dir',
                show_hl: bool = True,
                show_point:bool = True,
                title: str = '',
                ax=None):

    if ax is None:

        fig, ax = plt.subplots(figsize=(18, 6))

        line = peak_valley_df.plot(y='close', alpha=0.6, title=title, ax=ax)

    else:

        line = peak_valley_df.plot(y='close', alpha=0.6, title=title, ax=ax)

    if show_hl:

        peak_valley_df.plot(ax=line,
                            y='PEAK',
                            marker='o',
                            color='r',
                            mec='black')

        peak_valley_df.plot(ax=line,
                            y='VALLEY',
                            marker='o',
                            color='g',
                            mec='black')

    if show_point:

        peak_valley_df.dropna(subset=['POINT']).plot(ax=line,
                                                     y='POINT',
                                                     color='darkgray',
                                                     ls='--')
    if show_dir:

        peak_valley_df.plot(ax=line,
                            y=show_dir,
                            secondary_y=True,
                            alpha=0.3,
                            ls='--')

    return line

def get_clf_wave(price: pd.DataFrame,


                 rate: float,
                 method: str,
                 except_dir: bool = True,
                 show_tmp: bool = False,
                 dropna: bool = True) -> pd.DataFrame:


    if except_dir:

        # 修正
        perpare_data = Pipeline([('approximation', Approximation(rate, method)),
                ('mask_dir_peak_valley',Mask_status_peak_valley('dir')),
                ('except', Except_dir('dir')),
                ('mask_status_peak_valley', Mask_dir_peak_valley('status'))
                ])
    else:

       # 普通
        perpare_data = Pipeline([('approximation', Approximation(rate, method)),
                ('mask_dir_peak_valley',Mask_dir_peak_valley('dir')),
                ('mask_status_peak_valley', Mask_status_peak_valley('dir'))])

    return perpare_data.fit_transform(price)

quote_hs_300 = ak.stock_zh_index_daily('sh000300')
quote_hs_300['date'] = pd.to_datetime(quote_hs_300['date'])
quote_hs_300.set_index('date',inplace=True)

begin, end = '2020-02-01','2020-07-20'

# 方式一
flag_frame1: pd.DataFrame = get_clf_wave(quote_hs_300,None,'a',False)
flag_df1 = flag_frame1.loc[begin:end,['close','dir']]
flag_df1 = flag_df1.rename(columns={'dir':'方式1划分上下行'})
line = flag_frame1.loc['2021-01-01':'2021-07-30'].plot(figsize=(18, 6), y='close', color='red',
                    title='沪深300收盘价、DIF线与DEA线(2021-01-04至2021-07-30)')

flag_frame1.loc['2021-01-01':'2021-07-30'].plot(ax=line, y=['dif', 'dea'],
             secondary_y=True, color=['#3D89BE', 'darkgray']);
# 画图
line = flag_df1.plot(y='方式1划分上下行', secondary_y=True, figsize=(
    18, 5), ls='--', color='darkgray', title='沪深300与上下行划分(方式1)')
flag_df1.plot(y='close', ax=line, color='r');


# 方式2:划分上下行
flag_frame2: pd.DataFrame = get_clf_wave(quote_hs_300,0.5,'b',False)
flag_df2 = flag_frame2.loc[begin:end,['close','dir']]
flag_df2 = flag_df2.rename(columns={'dir':'方式2划分上下行'})
# 画图
line = flag_df2.plot(y='方式2划分上下行', secondary_y=True, figsize=(
    18, 5), ls='--', color='darkgray', title='沪深300与上下行划分(方式2,Rate=0.5)')
flag_df2.plot(y='close', ax=line, color='r');


# 方式3:划分上下行 -- 最标准的算法
flag_frame3: pd.DataFrame = get_clf_wave(quote_hs_300,2,'c',True)
flag_df3 = flag_frame3.loc[begin:end,['close','dir']]
flag_df3 = flag_df3.rename(columns={'dir':'方式3划分上下行'})
# 画图
line = flag_df3.plot(y='方式3划分上下行', secondary_y=True, figsize=(
    18, 5), ls='--', color='darkgray', title='沪深300与上下行划分(方式3,Rate=2)')

flag_df3.plot(y='close', ax=line, color='r');


status_frame: pd.DataFrame = get_clf_wave(quote_hs_300, 2, 'c', True)
dir_frame: pd.DataFrame = get_clf_wave(quote_hs_300, 2, 'c', False)
fig, axes = plt.subplots(2,figsize=(18,12))
# 画图
plot_pivots(dir_frame.iloc[330:450],
            show_dir=['dir'],
            show_hl=True,
            title='沪深300指数波段划分结果展示(方法3,Rate=2)',ax=axes[0])

plot_pivots(status_frame.iloc[330:450],
            show_dir=['status'],
            show_hl=True,
            title='沪深300指数波段划分结果展示(方法3-修正,Rate=2)',ax=axes[1]);