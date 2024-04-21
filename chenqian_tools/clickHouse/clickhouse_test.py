from clickhouse_driver import Client
import pandas as pd 
# def test_clickhouse_connection():
#     try:
#         # 创建 ClickHouse 客户端连接
#         client = Client(host='127.0.0.1', port=9000, user='default', password='123456')

#         # 执行测试查询
#         query = 'SELECT 1'
#         result = client.execute(query)

#         # 检查查询结果
#         assert result == [(1,)], "Test failed: ClickHouse query did not return expected result."

#         print("Test passed: Successfully connected to ClickHouse and executed query.")
#     except Exception as e:
#         print("Test failed:", str(e))

import clickhouse_driver 
def test_clickhouse_connection_driverconn():
    try:
        con_dict = dict(host='127.0.0.1', port=9000, user='default', password='123456',database = 'factor_exposure_values')
        conn = clickhouse_driver.connect(**con_dict)
        query = 'SELECT 1'
        df = pd.read_sql_query(query, conn)

    except Exception as e:
        print("Test failed:", str(e))

def sqlalchemy_test():
    from clickhouse_sqlalchemy import make_session
    from sqlalchemy import create_engine
    import pandas as pd

    conf = dict(server_host='127.0.0.1', port=8123, user='default', password='123456',db = 'default')

    connection = 'clickhouse://{user}:{password}@{server_host}:{port}/{db}'.format(**conf)
    engine = create_engine(connection, pool_size=100, pool_recycle=3600, pool_timeout=20)

    # sql = 'SHOW TABLES'

    # session = make_session(engine)
    # cursor = session.execute(sql)
    # try:
    #     fields = cursor._metadata.keys
    #     df = pd.DataFrame([dict(zip(fields, item)) for item in cursor.fetchall()])
    # finally:
    #     cursor.close()
    #     session.close()
    # 新增
    df.to_sql(name='rgc_v0_momentum_1d', con=engine, index=False, if_exists='append')

if __name__ == "__main__":

    test_clickhouse_connection_driverconn()

    file_p = rf'''C:\Users\1\Desktop\miller\chenqian_tools\clickHouse\factors_parquet\rgc_v0_momentum_1d.parquet'''
    df = pd.read_parquet(file_p)
