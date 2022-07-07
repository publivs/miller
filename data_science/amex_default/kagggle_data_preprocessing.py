import pandas as pd
import numpy as np
import os,sys

# 增加上层路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chenqian_tools.hdf_helper import *

data_path = r'''D:\amex-default-prediction'''
file_name = r'train_labes.csv'

target_path = r'''D:\amex-default-prediction\train_data.h5'''
chunked_df = pd.read_csv(r'''D:\amex-default-prediction\train_data.csv''',chunksize=1000000)
h5_client = h5_helper(target_path,'1')
for df_i  in chunked_df:
    df_i = degrade_incuracy(df_i,degrage_level = 'medium')
    h5_client.append_table(df_i,table_name='data')