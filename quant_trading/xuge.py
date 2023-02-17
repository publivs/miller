import sys,os

import requests
import pandas as pd
import numpy as np
import backtrader as bt
import time
import json

'''
url
获取的数据格式:OHLC,change_vol,pct_change
'''
def connect_url(target_url,req_headers):
    con_continnue = True
    while con_continnue:
        try:
            res_ = requests.get(target_url,headers=req_headers)
            if res_ is not None:
                con_continnue = False
            else:
                time.sleep(5)
                res_ = requests.get(target_url,headers=req_headers)
        except Exception as e:
            print("链接,出异常了！")
    return res_


def get_data():

    req_headers = {
                'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en',
                }

    start_time = 1533038400

    next_time_delta = 9705600

    end_time = int(pd.to_datetime(pd.Timestamp(time.time(),unit='s').date()).timestamp())

    timestamp_lst = range(start_time,end_time+next_time_delta,next_time_delta)

    res_lst = []
    for time_i in  timestamp_lst:
        df_i = get_df_data(req_headers,time_i)
        np.random.randint(0,3)
        res_lst.append(df_i)
    res_df = pd.concat(res_lst)
    res_df.tick_at = res_df.tick_at.astype('M8[s]')
    res_df = res_df.set_index('tick_at')

def get_df_data(req_headers,timestamp):

    target_url = f'''https://api-ddc-wscn.awtmt.com/market/kline?prod_code=XAUUSD.OTC&timestamp={timestamp}&tick_count=499&period_type=14400&fields=tick_at%2Copen_px%2Cclose_px%2Chigh_px%2Clow_px%2Cturnover_volume%2Cturnover_value%2Caverage_px%2Cpx_change%2Cpx_change_rate%2Cavg_px'''

    res_ = connect_url(target_url,req_headers)
    res = json.loads(res_.text)
    df_i  = pd.DataFrame(res['data']['candle']['XAUUSD.OTC']['lines'],columns=res['data']['fields'])
    # df_i['tick_at'] = df_i.tick_at.apply(lambda x:pd.Timestamp(x,unit = 's'))
    return df_i

def strategy_main():
    cerebro = bt.Cerebro()
    mod_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_path = mod_path.join('./xauusd_4h_from_20180731.csv')
    df = pd.read_csv(data_path,parse_dates=['tick_at'])
    df = df.set_index('tick_at')
    df = df[['open_px','high_px', 'low_px', 'close_px',]]
    data = bt.feeds.PandasDirectData(df)
    cerebro.adddata(data)
    cerebro.broker.setcash(10000)

class Mean_revert(bt.strategy):
        params = (
                ('roll_num',55),
                ('first_price_input',95),
                ('add_position_price',40),
                ('stake',0.1),
                ('max_hold_day',120)
                )

        def __init__(self) -> None:
            self.ma_55 = bt.indicator.MovingAverageSimple(self.data.close,
                                                        period=self.params.roll_num)
            self.sell_size = 0
            self.hold_day = 0
            self.first_price = 0
            self.add_position = 0
        def notify_order(self,order):
            if order.status in [order.Submitted, order.Accepted]:
                return
            # 如果order为buy/sell executed,报告价格结果
            if order.status in [order.Completed]: 
                if order.isbuy():
                    self.log(f'买入:\n价格:{order.executed.price},\
                    成本:{order.executed.value},\
                    手续费:{order.executed.comm}')
                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm
                else:
                    self.log(f'卖出:\n价格：{order.executed.price},\
                    成本: {order.executed.value},\
                    手续费{order.executed.comm}')
                self.bar_executed = len(self) 

            # 如果指令取消/交易失败, 报告结果
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log('交易失败')
                print(order.status,order.Margin)
            self.order = None

        def notify_trade(self, trade):

            if not trade.isclosed:
                return

            self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                    (trade.pnl, trade.pnlcomm))  # pnl：盈利  pnlcomm：手续费

        def next(self):
            if not self.position : # 无持仓的情况下
                if self.data.close[0] > self.ma_55 + self.params.first_price_input:
                    self.order = self.sell(size = 0.1)
                    self.first_price = self.data.close[0] # 记录一下进场价格
                self.hold_day += 1

            else: # 有持仓的情况下
                # 加仓逻辑1
                if self.data.close[0]>(self.first_price + 40)>(self.ma_55+100):
                    self.order = self.sell(size = 0.1)
                    self.hold_day += 1
                    self.add_position = 1
                # 加仓逻辑2
                elif (
                    (self.data.close[0]> (self.first_price + 100)) and (self.data.close[0]> (self.ma_55 + 100))
                    ):
                    self.order = self.sell(size = 0.2)
                    self.hold_day += 1
                    self.add_position = 1
                # 清仓逻辑
                elif self.data.close[0]< self.ma_55 -40:
                    self.close()
                elif self.hold_day>120:
                    self.close()


