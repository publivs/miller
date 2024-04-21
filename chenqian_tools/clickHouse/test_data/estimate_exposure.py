


import pandas as pd 
import numpy as np 
import os

import pyarrow as pa
import pyarrow.parquet as pq

def write_df_to_parquet(df, parquet_file_path):
    """
    将 pandas DataFrame 输出到 Parquet 文件

    Parameters:
        df (pandas.DataFrame): 要输出的 DataFrame 对象
        parquet_file_path (str): 输出的 Parquet 文件路径
    """
    # 将 DataFrame 转换为 PyArrow Table
    table = pa.Table.from_pandas(df)

    # 将 PyArrow Table 写入 Parquet 文件
    pq.write_table(table, parquet_file_path,compression=None)

def degrade_inaccuracy(df,degrage_level = 'low'):
    '''
    降低精度,方便读取和存储
    1、把objective全部切换成字符串
    2、float类型转为float16
    3、所有的obj全部强转为STR
    4、所有datetime和时间序列类型转成datetime64
    '''
    def quick_numeric(se,degrage_level):
        dtype_str = se.dtype.__str__()

        # 调整float
        if 'float' in dtype_str:
            # se = pd.to_numeric(se,downcast='float')
            if degrage_level  == 'low':
                se = se.astype('float16')
            else:
                se = se.astype('float32')
        # 调整int
        if 'int' in dtype_str:
            if degrage_level  == 'low':
                se = pd.to_numeric(se,downcast='integer')
            else:
                se = se.astype('int32')
        if 'datetime' in dtype_str:
            se = se.astype('datetime64[ns]')
        return se

    # 针对OBJ切换成string
    obj_columns = df.loc[:,df.dtypes == 'O'].columns
    df[obj_columns] = df[obj_columns].astype('str')

    # 调节精度优化读写
    df.loc[:,~df.columns.isin(obj_columns)] = df.loc[:,~df.columns.isin(obj_columns)].apply(lambda se:quick_numeric(se,degrage_level))
    return df

from concurrent.futures import ProcessPoolExecutor
from functools import partial

def generate_data(date, stock_list):
    np.random.seed(0)
    mean_value = 0
    std_deviation = 10000
    data = np.random.normal(mean_value, std_deviation, size=len(stock_list)).round(16)
    df = pd.DataFrame(data, columns=['factor_values'])
    df['scr_id'] = stock_list.astype('string').astype('int')
    df['exch_code'] = 'A'
    df['date'] = date
    df = df[['date','scr_id','factor_values','exch_code']]
    df['scr_id'] = df['scr_id'].astype('string')
    return df

def generate_factor_data(factor):
    date_list = pd.date_range('20020101', '20221231').strftime('%Y%m%d').astype('int')
    df = pd.read_csv(r'chenqian_tools\clickHouse\test_data\stocks_info.csv')
    stock_list = df['证券代码']

    with ProcessPoolExecutor(max_workers=4) as executor:
        # 使用 functools.partial 绑定 stock_list 参数
        generate_data_partial = partial(generate_data, stock_list=stock_list)
        res_lst = list(executor.map(generate_data_partial, date_list))

    df_all = pd.concat(res_lst, ignore_index=True)
    # 进行其他处理，比如去除数据的精度等
    df_all = degrade_inaccuracy(df_all, degrage_level='low')
    return df_all

if __name__ == '__main__':
    factor_name = 'rgc_v0_momentum_1d'
    df = generate_factor_data(factor_name)
    file_path = r'chenqian_tools\clickHouse\factors_parquet'
    parquet_path = file_path+ "\\" + f'{factor_name}.parquet'
    # 调用函数写入 Parquet 文件
    # write_df_to_parquet(df, parquet_path)
    df.to_parquet(parquet_path,compression=None,engine='fastparquet') # 必须用fastParquet作为engine
    
