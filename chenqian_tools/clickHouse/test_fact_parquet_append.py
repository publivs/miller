import pandas as pd
from fastparquet import write
from fastparquet import ParquetFile

df = pd.DataFrame({'scr_id': ["4","5","6"], 'factor_values': [7.000,8.000,9.000],
                   'exch_code':['A','A','A'],'date':[20240224,20240224,20240224]})
df = df[['date','scr_id','factor_values','exch_code']]
target_file_path =  rf'''C:\Users\1\Desktop\miller\chenqian_tools\clickHouse\factors_parquet\rgc_v0_momentum_1d.parquet'''
df.to_parquet(target_file_path,engine='fastparquet',append=True)
# pf = ParquetFile(target_file_path)
# df_read = pf.to_pandas()
df_read = pd.read_parquet(target_file_path)
print(df_read)
