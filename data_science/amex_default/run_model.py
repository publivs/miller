import pandas as pd
import numpy as np
import gc
from  matplotlib import pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(r"C:\Users\kaiyu\Desktop\miller")

from chenqian_tools.hdf_helper import *

def prepro_df(df):
    df['S_2'] = pd.to_datetime(df.S_2)
    features = [x for x in df.columns.values if x not in ['customer_ID', 'target']]
    df['n_missing'] = df[features].isna().sum(axis=1)
    df_out = df.groupby(['customer_ID']).nth(-1).reset_index(drop=True)
    del df
    _ = gc.collect()
    return df_out

'''
binary_features: ['B_31','D_87']

P_2:是当前信用评级的模型
'''
data_path = '''D:\\amex-default-prediction'''
test = 'train_data.h5'

h5_file = h5_helper(f'''{data_path}\{test}''',)
train_ = h5_file.get_table('data')
train_ = prepro_df(train_)


# corr_matrix
corr = train_.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(1, 1, figsize=(20,14))
sns.heatmap(corr, mask=mask, center=0, linewidths=1, annot=True, fmt=".2f", ax=ax)
plt.show()