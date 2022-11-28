#导入包
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as of  # 这个为离线模式的导入方法
import datetime
from scipy import interpolate
from datetime import timedelta
from dateutil.relativedelta import relativedelta

#数据导入
path = r'C:\Users\kaiyu\Desktop\miller\work_need\cython_learn\bond_cython_\real_ytm.csv'
data_policyfinancialdebt = pd.read_csv(path)

#使用插值法将收益率补齐
duration = sorted(np.concatenate((data_policyfinancialdebt['标准期限(年)'].values,data_policyfinancialdebt['剩余期限(年)'].values)))
duration.pop(0)
duration.pop()
f1=interpolate.interp1d(x=data_policyfinancialdebt['剩余期限(年)'],y=data_policyfinancialdebt['最优报买入收益率(%)'],kind='slinear')
f2=interpolate.interp1d(x=data_policyfinancialdebt['剩余期限(年)'],y=data_policyfinancialdebt['最优报卖出收益率(%)'],kind='slinear')
rates_new_buy = f1(duration)
rates_new_sell = f2(duration)

# 画收益率曲线图
line1 = go.Scatter(y=rates_new_buy, x=duration,mode='lines+markers', name='最优报买入收益率(%)')   # name定义每条线的名称
line2 = go.Scatter(y=rates_new_sell, x=duration,mode='lines+markers', name='最优报卖出收益率(%)')
fig = go.Figure([line1, line2])
fig.update_layout(
    title = '收益率曲线', #定义生成的plot 的标题
    xaxis_title = '期限', #定义x坐标名称
    yaxis_title = '收益率(%)'#定义y坐标名称
)
fig.show()


#计算债券的现金流列表，每一现金流对应的零息利率，每一现金流距离指定时间点间的时间距离
def cal_cashrtime(bar,couponrate,startdate,next_coupon_date,enddate, duration , rate_list,freq = 1):
    """
   计算债券的现金流列表，每一现金流对应的零息利率，每一现金流距离指定时间点间的时间距离
   Args:
       startdate:  需折现到的日期
       coupon_date: 下一次付息日
       enddate: 债券到期鈤日
       freq: 年付息次数
       duration: 用于插值法的期限list
       rate_list: 用于插值法的利率list
   Returns:
       现金流list，现金流时间距离list,现金流对应零息利率list
    """
    cashflow = []
    time_list = []
    date_temp = next_coupon_date
    while(enddate>=date_temp):
        cashflow.append(bar * couponrate)
        time_list.append((date_temp-startdate)/timedelta(365))
        date_temp = (date_temp + relativedelta(years=1))
    cashflow.append(bar)
    time_list.append((enddate-startdate)/timedelta(365))
    #插值法获取零息利率
    f=interpolate.interp1d(x=duration,y=rate_list,kind='slinear')
    r_list = list(f(time_list))
    return cashflow,time_list,r_list


def cy_cal_cashrtime(bar,couponrate,startdate,next_coupon_date,enddate, duration , rate_list,freq = 1):

    pass
