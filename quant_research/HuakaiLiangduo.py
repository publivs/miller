import pandas as pd
import numpy as np
import backtrader as bt
import akshare as ak

import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 75  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题


#
# 股指
hs_300_weight = ak.index_stock_cons_weight_csindex('000300')
sp_500 = 'a'

#外汇


#商品期货 [原油、黄金、有色金属指数]+
ak.fx_spot_quote()
ak.stock_zh_index_daily
ak.currency_hist