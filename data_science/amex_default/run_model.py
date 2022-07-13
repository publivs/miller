import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(r"C:\Users\kaiyu\Desktop\miller")

from chenqian_tools.hdf_helper import *

'''
binary_features: ['B_31','D_87']

P_2:是当前信用评级的模型
'''
data_path = '''D:\\amex-default-prediction'''
test = 'train_data.h5'

h5_file = h5_helper(f'''{data_path}\{test}''',)
train_ = h5_file.get_table('data')

print(train_)