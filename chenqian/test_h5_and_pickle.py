import os
from statistics import mode
from certifi import where
import pandas as pd
import numpy as np
import pickle





#
fw = open('test_1.pkl','wb')

# hdf5
'''
# 详情请参考这个
https://pandas.pydata.org/pandas-docs/stable/reference/io.html#hdfstore-pytables-hdf5
'''

class h5_helper():
    def __init__(self,store_path):
        self.store_path = store_path

    def read_table(self,table_name):
        '''
        读全表的时候用这个方法,拿数据的都是get_node
        '''
        store_client = pd.HDFStore(self.store_path,mode='r')
        # 这里还可以再包一些其他参数
        df = store_client.get(table_name)
        store_client.close()
        return df

    def select_table(self,table_name,where_str = None):
        '''

        '''
        store_client = pd.HDFStore(self.store_path,mode='r')
        # 这里还可以再包一些其他参数,读取尽量只用只读
        df = store_client.select(table_name,where=where_str)
        store_client.close()
        return df

    def append_table(self,df,table_name):
        '''
        根据以前的数据向里面追加额外数据
        '''
        print('数据写入中...')
        store_client = pd.HDFStore(self.store_path,mode='a')
        # 这里to_hdf还是调用的是append和put,后续看需不不需要分成两个方法
        df.to_hdf(self.store_path,table_name,format='table',mode='a',append=True)
        store_client.close()
        print('数据写入完毕...')

    def delete_table(self,table_name):
        store_client = pd.HDFStore(self.store_path,mode='r')
        store_client.remove(table_name)
        store_client.close()


# --------测试读取和写入部分---------- #
store_path = 'temp.h5'
write_mode = 'a'
h5_client = h5_helper(store_path)


store_client = pd.HDFStore(store_path,mode='r')
store_client.close()

# 造假数据
B = pd.DataFrame(['2021-08-20', '000300.SH', '600547.SH', '有色金属', float(0.9999),float(0.99999999), float(9999)],
                index=['pdate', 'code', 'stock', 'industry', 'weight', 'return', 'rn'],
                columns=  [100086]).T
B[['weight', 'return',]] = B[['weight', 'return',]].astype('float')
B['rn'] = B['rn'] .astype('int32')


A = pd.read_csv('test_data.csv',index_col='index')
# 造出虚拟数据

# 写入到二进制pickle
A.to_pickle('test_pickle.pkl')

B = A.iloc[0:10]
C = A.iloc[10:20]

# 写入到hdf5
B.to_hdf('temp.h5','temp',format='table',mode='a',append=True)
C.to_hdf('temp.h5','temp',format='table',mode='a',append=True)

# 追加行錯誤
'''
未解决
追加行

'''