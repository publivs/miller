import pandas as pd
import numpy as np
import backtrader as bt
import akshare as ak
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 75  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题

'''
参考资料:
1、https://blog.csdn.net/qq_43382509/article/details/106029241
2、https://max.book118.com/html/2021/0221/7101200043003056.shtm
'''

# 获取沪深300 及其成分
hs_300_weight = ak.index_stock_cons_weight_csindex(
    '000300'
)

quote_hs_300 = ak.stock_zh_index_daily('sh000300')

# 获取一只基金
'''
中信红利价值 = 900011
'''

fund_info = ak.fund_name_em()

target_fund_info  = fund_info.query(''' 基金代码=='900011' ''')

target_fund_quote = ak.fund_financial_fund_info_em('900011')

# 基金持仓--个股
ak.fund_portfolio_hold_em('900011')

#基金持仓--行业
ak.fund_portfolio_industry_allocation_em('900011')

# 获取申万一级行业
sws_level_1_url = 'https://www.swsresearch.com/institute-sw/api/index_publish/details/timelines/?swindexcode=801010'
sw_level_1 = ak.index_level_one_hist_sw()