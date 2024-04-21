import pandas as pd
import time
parquet_path = rf'''C:\Users\1\Desktop\miller\chenqian_tools\clickHouse\factors_parquet\rgc_v0_momentum_1d.parquet'''
t0 = time.time()
df = pd.read_parquet(parquet_path)
t1 = time.time()
print(t1 - t0)
