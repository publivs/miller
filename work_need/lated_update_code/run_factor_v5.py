# -*- coding: utf-8 -*-
"""
Created on 2024/1/9

1、修改优化画图格式
2、修改界面

@author: renguocheng
"""

'''
构建factor的一个方法,因子效果

用来做因子的效果检验

需要返回
    1、每个月的IC
    2、计算5分类的等权组合收益，并分年度统计
    3、计算多空收益，并分年度统计
    4、计算月度IC均值和

'''
import re
import pandas as pd 
import numpy as np 
import math
import datetime as dt
from scipy.stats.mstats import winsorize
from datetime import datetime
# from sklearn.preprocessing import StandardScaler as standarize
# from datetime import timedelta
# import matplotlib.pyplot as plt
# import seaborn as sns
from application.extension import db
import time
from loguru import logger
import json
from application.modules.factor_client.adapt_data import query_data_and_load
import gc


# from function.Param import *

# PATH = r"C:\Users\1\Desktop\changjiang\monitor_data\factor_csvs"
# pattern = r'_[0-9]{4}-[0-9]{2}-[0-9]{2}_[M|semi|D|Y|A|W].*' 

# plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体standarize
# plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 30)
def get_all_data(db,):
     pass

def rq2code(rq_code):
	if 'XSHG' in rq_code:
		return rq_code.split('.')[0] + '.SH' 
	elif 'XSHE' in rq_code:
		return rq_code.split('.')[0] + '.SZ'
	else:
		return rq_code

def standarize(df):
	df['values'] = df['values'] - df['values'].dropna().mean()/ df['values'].dropna().std()
	return df 

def winsorize(df,how='3sigma'):
	series = df['values']
	maxx = np.quantile(series,0.99)
	minn = np.quantile(series,0.01)
	series.mask(series > maxx, maxx).mask(series < minn, minn)
	df['values'] = series
	return df

def refresh_all_pcg_data(pct_delay):
    import akshare as ak
    start = pct_delay.index.min().strftime('%Y%m%d')
    end = pct_delay.index.max().strftime('%Y%m%d')
    
    for stock in pct_delay.columns : 
        stock_i = stock[0:6]
        if stock_i[0] == '6' or  stock_i[0] == '0':
            stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=stock_i,  
                                                    period="daily",
                                                    start_date= start , 
                                                    end_date= end ,
                                                    adjust="qfq")
            rev_se = stock_zh_a_hist_df[['日期','涨跌幅']].set_index('日期')['涨跌幅']
            pct_delay[stock] = rev_se/100
    return pct_delay


class FactorBacktest():
    
    def __init__(self,
                 name,
                 creator,
                 start_date,
                 end_date,
                 lag_day =0,
                 period = 5,
                 stock_pool = 'windA',
                 direction =1,
                 seg_num = 10,
                 IC_type = 'spearman',
                 exch_code = 'A',
                 is_default_calc = '0',
                 calculate_id = None):
        
        self.is_default_calc = is_default_calc
        self.calc_id = calculate_id
        self.factor_name = name
        self.exch_code = exch_code
        self.creator = creator
        self.start_date = start_date
        if end_date == '':
            self.end_date = dt.datetime.now().strftime("%Y%m%d")
            logger.info(f'采用默认计算,最大日期为{self.end_date}')
        else:
            self.end_date = end_date
        self.lag_day = int(lag_day)#是否需要延迟一天计算收益率
        self.period = int(period) #收益率天数
        self.IC_type = IC_type
        self.stock_pool = stock_pool#股票池
        self.direction = int(direction)#因子方向
        self.seg_num = int(seg_num) #分类数量
        self.data_all = dict()
        self.result = dict()
        self.check_args()
        self.init_default_calc_args()


    def check_args(self):
        if self.period <1 or self.period >250 :
            raise Exception('收益率调整天数请保证在1~250')
        if self.lag_day<0 or self.lag_day>10:
            raise Exception('延迟天数请保证在0~10')
        if self.seg_num<2 or self.seg_num>25:
            raise Exception('分组的组数请保证在2~25')

    def update_factor_calc_status(self,status,msg,result=''):
        # def delete_duplicated_status(calc_id,creator):
        # # 异步任务的逻辑
        #     t0 = time.time()
        #     update_query = f"delete from factors_clac_job  WHERE vc_creator ='{creator}' and vc_id = '{calc_id}' "
        #     with db.engine.connect() as connection:
        #         connection.execute(update_query)
        #     t1 = time.time()
        #     logger.info(f"耗时:{t1 - t0},因子计算状态更新中...")

        # try:
        #     delete_duplicated_status(self.calc_id,self.creator)
        # except Exception as e:
        #     logger.error(e)
        if status == '0':
            stauts_dict = {
                "vc_id":self.calc_id,
                "vc_creator":self.creator,
                "vc_result":result,
                "d_date":dt.datetime.now().strftime("%Y%m%d"),
                "vc_message":msg,
                "vc_status":status,
            }
            df = pd.DataFrame([stauts_dict])
            df.to_sql(name='factors_clac_job', con=db.engine, if_exists='append', index=False)
        
        else:
            t0 =time.time()
            update_query = f''' UPDATE factors_clac_job
                SET vc_status = '{status}',
                    vc_result = '{result}',
                    vc_message = '{msg}'
                WHERE  vc_creator ='{self.creator}' and vc_id = '{self.calc_id}' '''
            with db.engine.connect() as connection:
                connection.execute(update_query)
            t1 = time.time()
            logger.info(f"耗时:{t1 - t0},因子计算状态更新中...")
        


    def init_default_calc_args(self):
        '''
        Instru:
            计算之前会获取数据,对参与计算的参数和因子数据进行检查和准备。
            如果不是提交因子数据之后的默认计算,将取更新因子计算任务表的数据
        '''

        if self.is_default_calc != '1':
            self.update_factor_calc_status('0','Calculating...')


        sql_ = f'''select max(trade_date) as end_date, min(trade_date) as start_date
                        from factors_exposure_values
                        where factor_name = '{self.factor_name}' 
                        and creator = '{self.creator}'
                        and exch_code = '{self.exch_code}'
                '''
        start_and_end = pd.read_sql(sql_, con=db.engine)
        sql_info = f'''select * from factors_info 
                        where factor_name = '{self.factor_name}' 
                        and creator = '{self.creator}'
                        and exch_code = '{self.exch_code}' '''
        info_df = pd.read_sql(sql_info, con=db.engine)
        if start_and_end.empty:
            raise Exception('调用计算之前请确保提交了暴露数据')
        if info_df.empty:
            raise Exception('因子信息查询数据为空...')
        if self.is_default_calc == '1':
            self.start_date = str(start_and_end.start_date.values[0])
            self.end_date = str(start_and_end.end_date.values[0])
            self.direction = int(info_df.factor_sign.values[0]) if info_df.factor_sign.values[0] is not None else 1
            logger.info(f'因子默认计算重置时间为:开始日为{self.start_date}，结束日为{self.end_date}...')
        # 当是非默认计算的时候要更新状态数据
 

    def get_factor_data(self):
        #获取个股因子值

        # global PATH
        # file_name = self.factor_name + '.feather'
        # file = PATH +"\\" + re.sub(pattern, '', self.factor_name) + "\\" + file_name
        # self.factor_data = pd.read_feather(file)

        # 这里用多线程访问数据库来优化

        # factor_sql = f'''select trade_date as trd_dt,scr_id ,factor_values as values from factors_exposure_values where factor_name = '{self.factor_name}' 
        #          and trade_date >= {self.start_date} and trade_date <= {self.end_date}
        # ''' 
        # self.factor_data = pd.read_sql(factor_sql,con=db.engine)

        import concurrent.futures
        from concurrent.futures import as_completed

        def query_database(factor_name, exch_code,creator,start_date, end_date):
            
            factor_sql = f'''select trade_date as trd_dt, scr_id, factor_values as values 
                            from factors_exposure_values 
                            where factor_name = '{factor_name}' 
                            and creator = '{creator}'
                            and exch_code = '{exch_code}'
                            and trade_date >= '{start_date}' 
                            and trade_date <= '{end_date}' '''
            factor_data = pd.read_sql(factor_sql, con=db.engine)
            factor_data = (factor_data)

            # 降低精度
            factor_data['trd_dt'] = factor_data['trd_dt'].astype('Int32')
            factor_data['values'] = factor_data['values'].astype('float32')

            return factor_data
        # #------------------------------------------------切割时间范围成份的多线程版本------------------------------------------------#
        # 2.5年一个线程取数,如果总线程数量不到4,那就取4
        years_diff = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days / 365
        split_num = math.ceil(years_diff / 2.5)
        split_num = max(split_num, 4)

        date_ranges = pd.date_range(start=self.start_date,
                                    end=self.end_date,
                                    periods=split_num+1).strftime('%Y%m%d')
        date_ranges_str = [(self.factor_name,self.exch_code,self.creator,date_ranges[i], date_ranges[i+1],) for i in range(len(date_ranges)-1)]

        df_dict = {}
        for args_num in range(date_ranges_str.__len__()):
            args = date_ranges_str[args_num]
            df_dict[f'df_{args_num}']= {'func':query_database,'args':args}
        
        data_getter = db.threadings_db
        data_getter.get_args_dict(df_dict)
        t0 = time.time()
        res_dict = data_getter.get_data_threadings()
        t1 = time.time()
        self.factor_data = pd.concat(res_dict.values(),ignore_index=True).sort_values('trd_dt')

        # #------------------------------------------------------------end----------------------------------------------------------#
        # self.factor_data = query_database(self.factor_name, self.exch_code,self.creator,self.start_date, self.end_date)
        if self.factor_data.empty:
            raise Exception(f'因子计算:市场代码:{self.exch_code},获取因子{self.factor_name}的暴露值数据不足,无法计算。(暴露值最好提交数据超过30个交易日)')

        # self.factor_data = self.factor_data[self.factor_data['trd_dt'] >= self.start_date]
        if self.direction == -1:
           self.factor_data['values'] = - self.factor_data['values'] 
        # self.factor_data = self.factor_data[self.factor_data['values']  != 0 ]
        # self.factor_data = self.factor_data.rename(columns = {'date':'trd_dt',
        #                                                       'stock_code':'scr_id',})  
        self.factor_data = self.factor_data.drop_duplicates()
        self.factor_data = self.factor_data.dropna()
        self.factor_data = self.factor_data[~self.factor_data['values'].isin([np.inf,-np.inf])]
        # 
        if 'XSHE' in self.factor_data.scr_id.iloc[0] or 'XSHG' in self.factor_data.scr_id.iloc[0]:
            self.factor_data.scr_id = self.factor_data.scr_id.apply(lambda x : rq2code(x))

        if True:  # 暂时不需要
            # 去极值、标准化
            self.factor_data = self.factor_data.groupby('trd_dt').apply(
                                                                    lambda x:standarize(winsorize(x))
                                                                    )
            self.factor_data = self.factor_data.reset_index(drop=True)
        logger.info(f'''因子数据内存占用:{round(self.factor_data.memory_usage().sum() / 2**20)} Mb''')
        
            
    def get_stock_pool(self):
        # self.delist = pd.read_feather(r'E:\python\data\stock \delist.feather')
        pool_list = self.stock_pool
        self.pool_list = pool_list
        # name2code = {'hs300':'000300.SH',  
        #              'szqz':'399001.SZ',
        #              'cybzz':'399006.SZ',#创业板指
        #              'zz800':'000906.SH',
        #              'zz1000':'000852.SH',
        #              'gz2000':'399303.SZ',
        #              'windA':'881001.WI',
        #              'wind_weipan':'8841431.WI',
        #              'zz500':'000905.SH',
        #              'kc50':'000688.SH',
        #              'zx100':'399005.SZ',#中小100
        #              'sz50':'000016.SH',#上证50
        #              'sz100':'000132.SH',
        #              'zzhongli':'000922.CSI',#中证红利
        #              '300jiazhi':'000919.CSI',#300价值
        #              'szzs':'000001.SH',
        #              'szzz':'399106.SZ',#:深证综指
        #              'cybz':'399102.SZ',#:创业板综
        #              'zxbz':'399101.SZ',#:中小综指
        #              'AMC_growth_index_v2':'AMC_growth_index_v2'}
        # self.pool_list = [name2code[index]+'.IDX' for index in pool_list]
        if isinstance(pool_list,str):
            pool_list = f" '{pool_list}' " 
        else:
            pool_list = pool_list.__str__()[1:-1]
        sql = f'''select trd_dt, scr_idx_id, scr_idx_name, scr_id, scr_name, eff_dt,
                    expr_dt, cur_flag
         from iv_index_members where  scr_idx_id  in ({pool_list})'''
        self.pool = pd.read_sql(sql,con=db.engine)
        self.pool = (self.pool)
        if self.pool.empty:
            raise Exception('因子计算:无法从证券池中获取数据信息...')
        self.pool.rename(columns = {'s_con_windcode':'scr_id'},inplace=True)
        self.pool = self.pool[~self.pool.scr_id.str.startswith('9')]#剔除B股股
        self.pool = self.pool[~self.pool.scr_id.str.startswith('2')]#剔除B
        #self.pool = self.pool[~self.pool.scr_id.str.contains('BJ')]#剔除北交所个股

        self.pool = self.pool.drop_duplicates() 

        # if self.stock_pool == 'citics1':
        #     self.pool = pd.read_excel(r'E:\python\data\stock\index_citics1.xlsx')
        # if self.stock_pool != '':
        #     pool_list = self.stock_pool.split('+')
        #     for pool in pool_list:
        #         stock_pool_name = name2code[pool]
        #         pool_temp = pd.read_hdf(r'E:\python\data\stock\index\index' + stock_pool_name.split('.')[0] + '_1_M.h5' )
        #         pool_temp = pool_temp.rename(columns = {'s_con_windcode':'scr_id'})
        #         self.pool = pd.concat([self.pool,pool_temp])  

        #     self.pool = self.pool[~self.pool.scr_id.str.startswith('9')]#剔除B股股
        #     self.pool = self.pool[~self.pool.scr_id.str.startswith('2')]#剔除B
        #     #self.pool = self.pool[~self.pool.scr_id.str.contains('BJ')]#剔除北交所个股

        #     self.pool = self.pool.drop_duplicates()
    
    def get_stock_price(self):
        #self.price = pd.read_hdf(r'E:\python\data\stock\stock_quotation_all.h5','adjprice')
        logger.info('开始获取股票行情')
        def get_stock_data(pool_list,start_date,end_date):
            sql = f'''SELECT 
                        pool.scr_idx_id,
                            pool.scr_id,
                            pool.eff_dt,
                            pool.expr_dt,
                            stock_quot.trd_dt,
                            stock_quot.close_prc,
                            stock_quot.chg_pct 
                        FROM 
                            iv_index_members AS pool 
                        inner JOIN 
                            iv_stock_quot AS stock_quot
                        ON 
                            pool.scr_id = stock_quot.scr_id
                        WHERE  
                            pool.scr_idx_id IN ({pool_list})
                            and stock_quot.trd_dt >= {start_date}
                            and stock_quot.trd_dt <= {end_date}  '''

            df = pd.read_sql(sql,con=db.engine)
            # 降低精度
            df[['close_prc','chg_pct']] = df[['close_prc','chg_pct']].astype('float32')
            df[['eff_dt','expr_dt']] = df[['eff_dt','expr_dt']].astype('Int32')
            return df
        
        def get_data_stocks_threadings(self):
            pool_list = self.pool_list
            if isinstance(pool_list,str):
                pool_list = f" '{pool_list}' " 
            else:
                pool_list = pool_list.__str__()[1:-1]

            split_num = 4 
            date_ranges = pd.date_range(start=self.start_date,
                                        end=self.end_date,
                                        periods=split_num+1).strftime('%Y%m%d')
            date_ranges_str = [(pool_list,date_ranges[i], date_ranges[i+1],) for i in range(len(date_ranges)-1)]

            df_dict = {}
            for args_num in range(date_ranges_str.__len__()):
                args = date_ranges_str[args_num]
                df_dict[f'df_{args_num}']= {'func':get_stock_data,'args':args}
            
            data_getter = db.threadings_db
            data_getter.get_args_dict(df_dict)
            t0 = time.time()
            res_dict = data_getter.get_data_threadings()
            t1 = time.time()
            price = pd.concat(res_dict.values(),ignore_index=True).sort_values('trd_dt')
            return price
        
        def get_data_from_cache(self):
            if isinstance(self.pool_list,list):
                df_lst = []
                for index in self.pool_list:
                    df = query_data_and_load(index,self.start_date,self.end_date)
                    df_lst.append(df)
                df = pd.concat(df_lst,ignore_index=True)
            else:
                df = query_data_and_load(self.pool_list,self.start_date,self.end_date)
            return df
                
        # 利用多线程并发从数据库取数
        try:
            self.price = get_data_from_cache(self)
            if self.price is None or len(self.price) == 0:
                logger.info('缓存获取数据失败,正在从数据库读取中')
                self.price = get_data_stocks_threadings(self)
        except Exception as e:
            logger.info('缓存获取数据失败,正在从数据库读取中')
            self.price = get_data_stocks_threadings(self)

        logger.info(f'股票行情获取完毕')
        if self.price.empty :
            raise Exception('因子计算:无法获取股票数据行情信息...')
        self.price = self.price[['scr_id','trd_dt','close_prc']].drop_duplicates()

        # 转换成Piovt
        self.price_stock_table = pd.pivot_table(self.price,index = 'trd_dt',columns = 'scr_id',values = 'close_prc')

        del self.price
        gc.collect()
        logger.info(rf'''股票数据内存占用:{round(self.price_stock_table.memory_usage().sum() / 2**20)} Mb...''')
        # self.price_stock_table.index = pd.to_datetime(self.price_stock_table.index)

    def get_indus_index(self):
        #叠加行业指数配置
        
        def get_indus_index_db(self):
            index_sql = f'''
                            select t.scr_id,
                            quot.trd_dt,
                            quot.close_prc,
                            quot.chg_pct
                            from iv_index_info t
                            inner join iv_index_quot quot
                            on t.scr_id = quot.scr_id 
                            where 
                                -- 申银万国
                                    (
                                    t.pub_inst_name = '申银万国指数'
                                    and t.idx_style = '申万行业'
                                    and t.ds_src = 'WIND'
                                    and t.idx_type = '647002001'
                                    and t.delist_dt is null
                                    )
                                or
                                --  中信指数
                                (
                                t.pub_inst_name  = '中信证券股份有限公司'
                                and t.idx_style  = '一级行业指数'
                                and t.delist_dt  is null
                                )
                            and quot.trd_dt >= {self.start_date}
                            and quot.trd_dt <= {self.end_date}            
                            '''
            df = pd.read_sql_query(index_sql,con=db.engine)
            # 降低精度
            if not df.empty:
                df['trd_dt'] = df['trd_dt'].astype('Int32')
                df[['close_prc','chg_pct']] = df[['close_prc','chg_pct']].astype('float32')
            return df
        
        def get_indus_index_cache(self):
            index_name = 'industry_index_quote'
            df = query_data_and_load(index_name,self.start_date,self.end_date)
            return df
        
        try:
            self.price_index = get_indus_index_cache(self)
            if self.price_index is None or len(self.price_index) == 0:
                self.price_index = get_indus_index_db(self)
        except Exception as e: 
            logger.info('缓存获取数据失败,正在从数据库读取中')
            self.price_index = get_indus_index_db(self)

        # self.price_index[['close_prc','chg_pct']] = self.price_index[['close_prc','chg_pct']].astype('float16')
        # logger.info(f'指数行情获取完毕,{index_sql}')
        # if self.price_index.empty:
        #     raise Exception('因子计算:无法获取指数行情信息...')
        # self.price_index = pd.read_feather(r'E:\python\data\index\index_quota.feather') 

        self.price_index = self.price_index[['scr_id','trd_dt','close_prc']].drop_duplicates(subset=['scr_id','trd_dt'])
        self.price_index_table = pd.pivot(self.price_index,index = 'trd_dt',columns = 'scr_id',values = 'close_prc')
        self.price_table = pd.merge(self.price_stock_table,
                                    self.price_index_table,
                                    how = 'outer',
                                    left_index = True,right_index = True)
        self.price_table = self.price_table.ffill(limit = 5)
        self.pct = (self.price_table/self.price_table.shift(self.period)-1).shift(-self.period)#未来period天的收益率
        self.pct_daily = (self.price_table/self.price_table.shift(1)-1)
        self.pct_delay = self.pct.shift(-self.lag_day)#考虑lag天数需求

        # 释放冗余变量
        del self.price_index
        del self.price_index_table
        del self.price_table
        del self.pct
        gc.collect()
        logger.info(f'''PCT数据内存占用如下,Delay_df:{round(self.pct_delay.memory_usage(deep=True).sum() / 2**20,4)} Mb,Daily_df:{round(self.pct_daily.memory_usage(deep=True).sum()/2**20)} Mb...''')
        # 考虑lag天数需求
        # logger.info(f'检查点:收益率后延之后,包含日期如下:{self.pct_delay.index}...')

    def backtest(self):
        #循环计算每月收益率
        #需要开始时间，结束时间，和选股池
        #返回5分类各期的选股池和日度收益，以及月度IC情况
        
        print('start_calculate')
        trd_dt_sql = f''' select cal_dt  from pub_workday pw
							where is_workday = '1'
							and exch_code  = 'SH'
                            and cal_dt >= {self.start_date}
                            and cal_dt <= {self.end_date}
							order by cal_dt '''
        trade_days = pd.read_sql(trd_dt_sql,con=db.engine)
        if trade_days.empty:
            raise Exception('因子计算:无法获取回测所需要的交易日信息......')
        # trade_days = pd.read_excel(r'E:\python\data\trade_days.xlsx',header = None)
        
        if type(self.period) == int:
            trade_days_freq = trade_days.iloc[::self.period]
        else:
            trade_days_freq = trade_days.copy()
        #period= 1,IC每日计算；period = 5，IC每5天计算；period= 20，IC每20天计算
        # logger.info(f'检查点:交易日如下:{trade_days_freq}')

        #  放置计算结果的数据堆
        self.result['IC'] = pd.DataFrame(columns = ['pearson','spearman'])
        self.result['result'] = pd.DataFrame(columns = list(range(0,self.seg_num)))
        self.result['result_daily'] = pd.DataFrame(columns = list(range(0,self.seg_num)))
        self.data_all['cover_rate'] = pd.DataFrame(columns = ['factor_num','stock_num'])

        # 池子数据拷出
        delist = self.pool[['scr_id','eff_dt','expr_dt']].copy()
        # 池子数据整理
        self.pool['eff_dt'] = pd.to_datetime(self.pool['eff_dt'].astype('Int32').astype('string'))
        self.pool['expr_dt'] =  pd.to_datetime(self.pool['expr_dt'].astype('Int32').astype('string'))
        delist['eff_dt'] = pd.to_datetime(delist['eff_dt'].astype('Int32').astype('string'))
        delist['expr_dt'] =  pd.to_datetime(delist['expr_dt'].astype('Int32').astype('string'))

        #整理交易日和开始日起始日数据
        trade_days_freq['cal_dt'] = pd.to_datetime(trade_days_freq['cal_dt'].astype('int').astype('string'))
        self.start_date = pd.to_datetime(self.start_date)
        self.end_date = pd.to_datetime(self.end_date)

        #整理pct数据
        self.pct_delay.index = pd.to_datetime(self.pct_delay.index.astype('int').astype('string'))
        self.pct_daily.index = pd.to_datetime(self.pct_daily.index.astype('int').astype('string'))
        
        #  整理因子数据
        df = self.factor_data.pivot_table(index = 'trd_dt',columns = 'scr_id',values = 'values')
        df.index = pd.to_datetime(df.index.astype('int').astype('string'))
        # ------------------- 因为数据不足有一些搞笑情况这里特殊处理一下 ------------------------- #
        # self.pct_delay = refresh_all_pcg_data(self.pct_delay)
        # self.pool = self.pool.loc[self.pool.scr_id.isin(self.pct_delay.columns)]
        # --------------------------------------- end ---------------------------------------- #

        self.pct_delay.columns = self.pct_delay.columns.str.slice(0,6)
        self.pct_daily.columns = self.pct_daily.columns.str.slice(0,6)

        for i in range(0,len(trade_days_freq)):#每个计算日循环
         #i =358
        #因子需要计算的日子为：有未来收益，在开始结束日期内.
            date_now = trade_days_freq.iloc[i,0]
            if date_now > self.start_date \
                and date_now < self.end_date - dt.timedelta(self.period) \
                and date_now in self.pct_delay.index \
                and date_now >= df.index.min():      
                # date_now_1 = trade_days_freq.iloc[i+1,0] #date_now = dt.datetime(2011,9,26)
                #获取最新因子值
                # logger.info(date_now)
                # date_now_1 = trade_days_freq.iloc[i+1,0] #date_now = dt.datetime(2011,9,26)
                print(date_now)#date_now = pd.to_datetime('2023-05-24')
                factor = df.loc[:date_now]
                factor = factor.iloc[-1,:].dropna()#获取最新因子值
                pct_cum = self.pct_delay.loc[date_now,:]
                pct_daily = self.pct_daily.loc[date_now:].iloc[self.lag_day+1:].iloc[:self.period]
                if 0:#剔除ST
                    delist_ = delist[delist['entry_dt'] < date_now]
                    delist_ = delist_[delist_['remove_dt'] > date_now]
                    delist_ = delist_['sec_code'].drop_duplicates()
                    factor = factor[~factor.index.isin(list(delist_.values))]
                if self.stock_pool != '':#
                    pool = self.pool[ ((self.pool.expr_dt > date_now)|(self.pool.expr_dt.isna()))&(self.pool.eff_dt <= date_now)]
                    factor = factor[factor.index.isin(pool.scr_id.str.slice(0,6))] # 保证因子的数据在池子里有票
                    factor = factor[factor.index.isin(pct_cum.index.str.slice(0,6))] # 保证因子的数据在行情里有票
                    self.data_all['cover_rate'].loc[date_now] = [len(factor),len(pool)]
                if len(factor) > 0:
                    # print(len(factor))
                    factor = pd.DataFrame(factor)
                    factor.columns =['values']
                    factor = factor[~factor.index.duplicated(keep='first')]
                    pct_cum = pct_cum[~pct_cum.index.duplicated(keep='first')]
                    pct_daily = pct_daily[~pct_daily.index.duplicated(keep='first')]
                    
                    factor['pct_cum'] = pct_cum

                    factor['value_rank'] = factor['values'].rank(pct = True,ascending = False)
                    factor['pct_rank'] = factor['pct_cum'].rank(pct = True,ascending = False)
                    factor['group'] = pd.cut(factor['value_rank'],self.seg_num,labels = list(range(self.seg_num)))
                    self.result['IC'].loc[date_now] = [factor[['values','pct_cum']].dropna(how = 'any').corr().fillna(0).iloc[0,1],
                                                       factor[['values','pct_cum']].dropna(how = 'any').corr('spearman').fillna(0).iloc[0,1]]
                    ret_period = []
                    ret_daily = pd.DataFrame([],columns = range(0,self.seg_num))
                    for seg in list(range(0,self.seg_num)):
                        ret_period.append(pct_cum[factor[factor.group == int(seg)].index].mean())
                        ret_daily[seg] = (pct_daily[factor[factor.group == int(seg)].index].mean(axis = 1))    
                        
                    self.result['result'].loc[date_now] = ret_period
                    self.result['result_daily'] = pd.concat([self.result['result_daily'],ret_daily])

    def generate_resp_result(self):
        from copy import deepcopy
        self.output_dict = deepcopy(self.result)
        logger.info(f'''计算结果数据如下: \n {self.output_dict['result'].to_markdown(tablefmt='grid')}''')
        if self.output_dict['result'].empty:
            raise Exception(f'{self.factor_name}因子计算结果全部为空,请调整参数或者检查已经提交的数据')
        # IC输出:IC,rolling_IC,IC_cum
        self.output_dict['IC']['rank_IC_ma'] = self.output_dict['IC']['spearman'].rolling(int(120/self.period),min_periods = 1).mean()
        #period = 1,120;persiod = 5,24,period = 20,6
        self.output_dict['IC']['rank_IC_cumsum'] = self.output_dict['IC']['spearman'].cumsum()    
        self.output_dict['IC']['IC_ma'] = self.output_dict['IC']['pearson'].rolling(int(120/self.period),min_periods = 1).mean()
        #period = 1,120;persiod = 5,24,period = 20,6
        self.output_dict['IC']['IC_cumsum'] = self.output_dict['IC']['pearson'].cumsum()

        if self.IC_type == 'spearman':
            self.output_dict['IC_cumsum'] = self.output_dict['IC']['rank_IC_cumsum']
            self.output_dict['factor_IC'] =  self.output_dict['IC'][['spearman','rank_IC_ma']]
        if self.IC_type == 'pearson':
            self.output_dict['IC_cumsum'] = self.output_dict['IC']['IC_cumsum']
            self.output_dict['factor_IC'] =  self.output_dict['IC'][['pearson','IC_ma']]
        
        # 多空收益输出: 多空净值
        self.result['result'] = self.result['result'].fillna(0)
        self.result['result_daily'] = self.result['result_daily'].fillna(0)
        
        self.output_dict['result']['long-short'] = self.output_dict['result'][0]-self.output_dict['result'][self.seg_num-1]
        self.output_dict['result_daily']['long-short'] = self.output_dict['result_daily'][0]-self.output_dict['result_daily'][self.seg_num-1]

        long_short_df = self.output_dict['result_daily']['long-short'].copy()
        long_short_df.index = [str(x) for x in long_short_df.index]
        long_short_df = long_short_df.to_frame()
        long_short_df['ls_nav'] = (long_short_df+1).cumprod()
        long_short_df['ls_nav_max'] = long_short_df['ls_nav'].rolling(100000,min_periods = 1).max()
        long_short_df['max_drawdown'] = (long_short_df['ls_nav'] - long_short_df['ls_nav_max'])/long_short_df['ls_nav_max'] # 最大回车
        self.output_dict['long_short_df'] = long_short_df

        # 分组年化收益率
        d = (self.output_dict['result'][list(range(self.seg_num))]+1).cumprod()
        
        d_pct = d.resample('A').last()/d.resample('A').last().shift(1)-1
        d_pct.loc[d_pct.index[0]] = d.resample('A').last().iloc[0]-1
        d_pct.index = [str(x)[:4] for x in d_pct.index ]
        self.output_dict['seg_annual_rev'] = d_pct

        # 分组超额收益
        self.output_dict['result_daily']['mean'] = self.output_dict['result_daily'][list(range(self.seg_num))].mean(axis = 1)
        self.output_dict['result_ar'] = pd.DataFrame()
        for i in list(range(self.seg_num)):
            self.output_dict['result_ar'][i] = self.output_dict['result_daily'][i]-self.output_dict['result_daily']['mean']
        
        # 分组超额净值
        self.output_dict['seg_alpha_nav'] = (self.output_dict['result_ar'][list(range(self.seg_num))]+1).cumprod()

        # 因子覆盖率,从之前的数据保存字典里面拷贝出来
        self.output_dict['cover_rate'] = self.data_all['cover_rate'] # 这里先把数据从data_all里面拷出来

        # 只输出6张图所需要的数据
        self.output_dict.pop('IC')
        self.output_dict.pop('result')
        
    def generate_default_df(self):
        # 默认信息展示表的数据在这里也顺便做一次提交
        try:
            if self.IC_type == 'spearman':
                # self.result['IC']['spearman'].mean()/self.result['IC']['spearman'].std() * np.sqrt(240/self.period)
                self.result['IC']['spearman'].mean()
                mean_ic = self.output_dict['factor_IC']['spearman'].mean()
                ir =  self.output_dict['factor_IC']['spearman'].mean()/self.output_dict['factor_IC']['spearman'].std() * np.sqrt(240/self.period)
            elif self.IC_type == 'pearson':
                # self.result['IC']['pearson'].mean()/self.result['IC']['pearson'].std() * np.sqrt(240/self.period)
                mean_ic = self.output_dict['factor_IC']['pearson'].mean()
                ir =  self.output_dict['factor_IC']['pearson'].mean()/self.output_dict['factor_IC']['pearson'].std() * np.sqrt(240/self.period)
            # 整理一次数据
            calc_time = datetime.now()
            self.defalt_result_df = pd.DataFrame({
                        'creator': [self.creator],
                        'factor_name': [self.factor_name],
                        'mean_ic': [mean_ic],
                        'ic_ir': [ir],
                        'calc_time':[calc_time],
                        'exch_code':[self.exch_code]
                    })
        except Exception as e:
            logger.error('因子默认信息表计算失败...')

    # def result_show(self):
    #     #画4幅图
    #     global PATH
    #     sns.plotting_context("talk")

    #     # self.result['IC']['rank_IC_ma'] = self.result['IC']['spearman'].rolling(int(120/self.period),min_periods = 1).mean()
    #     # #period = 1,120;persiod = 5,24,period = 20,6
    #     # self.result['IC']['rank_IC_cumsum'] = self.result['IC']['spearman'].cumsum()    
    #     # self.result['IC']['IC_ma'] = self.result['IC']['pearson'].rolling(int(120/self.period),min_periods = 1).mean()
    #     # #period = 1,120;persiod = 5,24,period = 20,6
    #     # self.result['IC']['IC_cumsum'] = self.result['IC']['pearson'].cumsum()            
        
    #     if self.direction == 1:
    #         title = self.factor_name + "_预测收益率天数" + str(self.period) +'_lag_day_'+ str(self.lag_day) + '\n_' + self.stock_pool +'_'+ str(self.seg_num) +'分组效果，0分组因子值最大'
    #     else:
    #         title = self.factor_name + "_预测收益率天数" + str(self.period) +'_lag_day_'+ str(self.lag_day) + '\n_' + self.stock_pool + '_'+ str(self.seg_num) +'分组效果，' + str(self.seg_num-1) + '因子值最大'
        
    #     fig = plt.figure(figsize = [40,40])
    #     fig.title = title
    #     plt.suptitle(title)
        
    #     #图1：IC和rankIC
    #     #IC及IC均值-spearman
    #     ax1 = fig.add_subplot(2,1,1)
    #     x_axis = [str(x) for x in list(self.result['IC']['spearman'].index)]
    #     ax1.bar(x_axis,list(self.result['IC']['spearman']))
    #     ax1.plot(x_axis,list(self.result['IC']['rank_IC_ma']))
    #     ax1.set_title('IC均值:' + str(self.result['IC']['spearman'].mean())[:6] +',年化ICIR:' +
    #                  str(self.result['IC']['spearman'].mean()/self.result['IC']['spearman'].std() * np.sqrt(240/self.period))[:6])
    #     ax2 = ax1.twinx()

    #     #累计IC
    #     ax2.plot(x_axis,self.result['IC']['spearman'].cumsum(),color = 'red',label = 'cumsumIC')
    #     plt.legend()

    #     #IC及IC均值-pearson
    #     ax1 = fig.add_subplot(2,1,2)
    #     #x_axis = [str(x) for x in list(self.result['IC']['IC_2'].index)]
    #     ax1.bar(x_axis,list(self.result['IC']['pearson']))
    #     ax1.plot(x_axis,list(self.result['IC']['IC_ma']))
    #     ax1.set_title('IC均值:'  + str(self.result['IC']['pearson'].mean())[:6] +',ICIR:' +
    #                  str(self.result['IC']['pearson'].mean()/self.result['IC']['pearson'].std() * np.sqrt(240/self.period))[:6])
    #     ax2 = ax1.twinx()
    #     #累计IC
    #     ax2.plot(x_axis,self.result['IC']['pearson'].cumsum(),color = 'red',label = 'cumsumIC')
    #     plt.legend()        
        
    #     # 实例化:图2
    #     fig = plt.figure(figsize = [40,40])
    #     fig.title = title
    #     plt.suptitle(title)        

    #     # 因子分组分年收益率
    #     ax1 = fig.add_subplot(2,3,1)
    #     d = (self.result['result'][list(range(self.seg_num))]+1).cumprod()
    #     d_pct = d.resample('A').last()/d.resample('A').last().shift(1)-1
    #     d_pct.loc[d_pct.index[0]] = d.resample('A').last().iloc[0]-1
    #     d_pct.index = [str(x)[:4] for x in d_pct.index ]

    #     x = np.arange(len(d_pct))
    #     bar_with = 0.1
    #     tick_label = d_pct.index
    #     i = 0
    #     for col in d_pct.columns:
    #         ax1.bar(x + bar_with * i,
    #                 d_pct[col],
    #                 bar_with,
    #                 label = col,
    #                 tick_label = tick_label)
    #         i = i + 1
            
    #     ax1 = fig.add_subplot(2,3,2)
    #     years = ((d.index.max() - d.index.min())/timedelta(1)/365)
    #     anual_ret = d.iloc[-1:,].apply(lambda x:x**(1/years))-1
    #     ax1.bar(anual_ret.columns,anual_ret.values[0])
        
            
    #     #多空收益率        
    #     ax1 = fig.add_subplot(2,3,3)
    #     temp = self.result['result']['long-short'].copy()
    #     temp.index = [str(x) for x in temp.index]
    #     temp['ls_nav'] = (temp+1).cumprod()
    #     temp['ls_nav_max'] = temp['ls_nav'].rolling(100000,min_periods = 1).max()
    #     temp['max_drawdown'] = (temp['ls_nav'] - temp['ls_nav_max'])/temp['ls_nav_max'] # 最大回车

    #     ax1.fill_between(temp.index,temp['max_drawdown'],color = 'grey')
    #     ax1.bar(temp.index,temp['long-short'],color = 'blue')
    #     ax2 = ax1.twinx()
    #     ax2.plot(temp.index,(temp['ls_nav']).values,color = 'red')
    #     plt.legend(labels = [str(x) for x in range(0,self.seg_num)])
        
        
    #     #因子覆盖率        
    #     ax1 = fig.add_subplot(2,3,4)
    #     ax1.plot(self.data_all['cover_rate'].index,self.data_all['cover_rate']['factor_num'])
    #     ax1.plot(self.data_all['cover_rate'].index,self.data_all['cover_rate']['stock_num'])
    #     plt.legend(labels = ['factor_num','stock_num']) 
        
    #     # 分组超额收益情况
    #     self.result['result']['mean'] = self.result['result'][list(range(self.seg_num))].mean(axis = 1)
    #     self.result['result_ar'] = pd.DataFrame()
    #     for i in list(range(self.seg_num)):
    #         self.result['result_ar'][i] = self.result['result'][i]-self.result['result']['mean']
        
    #     # 分组超额净值
    #     ax1 = fig.add_subplot(2,3,5)
    #     self.result['seg_alpha_rev'] = (self.result['result_ar'][list(range(self.seg_num))]+1).cumprod()
    #     ax1.plot( self.result['seg_alpha_rev'] )
    #     plt.legend(labels = [str(x) for x in range(0,self.seg_num)])
        
    #     plt.savefig(PATH + '\\' + re.sub(pattern, '', self.factor_name) + '\\' + title.replace('\n','') +'.png')

    #     plt.show()

    def set_weight(self):
        weight_df = pd.DataFrame()
        for i in self.data_all.keys():
            temp = self.data_all[i]
            # print(i)
            if len(temp.group.drop_duplicates()) > 1:
                print(len(temp))
                temp = temp[temp.group == 0][['group']]
                temp['trd_dt'] = str(i)[:10].replace('-','')
                temp['TARGET_WEIGHT'] = 1/len(temp)
                temp = temp.reset_index()
                temp = temp.rename(columns = {'scr_id':'TICKER'})
                temp = temp[['trd_dt','TICKER','TARGET_WEIGHT']]
                temp['NAME'] = '1'
                weight_df = pd.concat([temp,weight_df])
            else:
                print(i)
        if len(weight_df) > 0:
            weight_df.to_excel(r'E:\python\rqdata\backtest_factor\result\\' + self.factor_name + '_' + self.stock_pool + '.xlsx')
        
    def distribution(self,split_num = 20):
        import numpy as np
        self.dist = {}
        for d in self.factor_data.trd_dt.drop_duplicates():
            temp_ = self.factor_data[self.factor_data['trd_dt'] == d]
            maxx = temp_['values'].max()
            minn = temp_['values'].min()
            x = []
            y = []
            for i in np.linspace(minn,maxx,split_num):
                x.append(i)
                y.append((temp_['values'] <= i).sum())
            dist_df = pd.DataFrame(y,index = x,columns = ['cumsum'])
            dist_df['num'] = (dist_df['cumsum'] - dist_df['cumsum'].shift(1)).fillna(0)
            self.dist[d] = dist_df
            dist_df.plot(title = d)

    def calc_icir_main(self):

        try:
            self.get_stock_pool()
            logger.info('因子计算:证券池获取完毕...')
            self.get_stock_price()
            logger.info('因子计算:行情获取完毕...')
            self.get_indus_index()
            logger.info('因子计算:行业指数行情获取完毕...')
            self.get_factor_data()
            logger.info('因子计算:因子数据获取完毕...')
            self.backtest()
            logger.info('''因子计算:回测完毕...''')
            self.generate_resp_result()
            logger.info('''因子计算:正在返回数据...''')

            # 这里生成一下默认计算的数据
            if self.is_default_calc == '1':
                self.generate_default_df()

            # self.result_show()
            # self.set_weight()
            # self.distribution()

            # 格式化一下最终的响应数据
            res_dict = {}
            for k,v in self.output_dict.items():
                v.index =pd.DatetimeIndex(v.index).strftime('%Y%m%d')
                v = v.dropna()
                v = v.reset_index()
                res_dict[k] = v.to_json(orient='records')
            
            #  默认信息数据提交到默认信息表
            if self.is_default_calc == '1':
                try:
                    logger.info(f'''默认计算数据如下: \n {self.defalt_result_df.to_markdown(tablefmt='grid')}''')
                    self.defalt_result_df['default_resp_data'] = json.dumps(res_dict)
                    self.defalt_result_df.to_sql(name='factors_default_result', con=db.engine, if_exists='append', index=False)
                    logger.info(f"因子默认计算结束,数据提交完毕...")
                except Exception as e:
                    logger.error('因子信息默认信息提交的时候提交失败...')

            # 非默认计算的数据提交到计算任务结果表 
            else:
                result = json.dumps(res_dict)
                self.update_factor_calc_status('1','ok',result)

            return  res_dict
        
        except Exception as e:
            self.update_factor_calc_status('2',f'计算失败,错误信息:{e}',)
            raise Exception(e)
    
if __name__ == '__main__':

    #检测因子在不同股票池和不同时间阶段的有效性
    factor_name = 'ppreversal_2005-01-01_D'
    start_date = '2005-01-01'
    end_date = '2023-11-1'
    lag_day = 0
    period = 20
    stock_pool = 'sz50+300jiazhi+zzhongli'
    #stock_pool = 'szzz'
    #stock_pool = 'citics1'
    
    stock_pool = 'hs300'
    stock_pool = 'zz500+zz1000'
    stock_pool = 'zz1000+gz2000'
    #stock_pool = 'hs300'
    stock_pool = 'zz800'
    stock_pool = 'AMC_growth_index_v2'
    #stock_pool = 'wind_weipan'
    stock_pool = 'zz1000'
    stock_pool = 'windA'
    direction = -1#1的话，因子值最大的为第一分组
    IC_type = 'spearman'#pearson/spearman
    seg_num = 5
    benchmark = 'hs300'
    bt_client = FactorBacktest(factor_name,
                           start_date,
                           end_date,lag_day,
                           period,
                           stock_pool,
                           direction,
                           seg_num,
                           IC_type)
    bt_client.get_stock_pool()
    bt_client.get_stock_price()
    bt_client.get_factor_data()
    bt_client.backtest()
    # bt_client.result_show()
    # bt_client.set_weight()
    # bt_client.distribution()
    #test = self.data_all