from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(session_options={"autocommit": False, "autoflush": True})

import threading
import asyncio
import gc


def init_databases(app: Flask):
    db.init_app(app)
    db.threadings_db = GetDataFromDB(app) # 注意这里,我把这个取数类直接赋到了db对象上
class GetDataFromDB:
    def __init__(self,app) -> None:
        '''
        Examples:
            df_dict = {
            'bond_quote': {'func': get_bond_quote, 'args': (start_date, end_date, bond_check_list)},
            'bond_info': {'func': get_bond_info, 'args': (start_date, end_date, bond_check_list)},
            'exec_info': {'func': get_exec_info, 'args': (start_date, end_date, bond_check_list)},
            'cash_flow_info': {'func': get_cash_flow_data, 'args': (start_date, end_date, bond_check_list)},
            'bond_rate_class': {'func': get_bond_rate_class, 'args': (start_date, end_date)},
            'working_day': {'func': get_working_day, 'args': (start_date, end_date)}
        }
        '''
        self.register_2_app(app)
        
    def register_2_app(self,app_obj):
        self.app = app_obj
    
    def get_args_dict(self,df_arg_dict):
        self.df_arg_dict = df_arg_dict

    def get_data_synchronous(self,):
        df_result = {}
        for key, value in self.df_arg_dict.items():
            func = value['func']
            args = value['args']
            df_result[key] = func(*args)
        return df_result
    
    def execute_func(self, key, func, args):
       with self.app.app_context():
            return func(*args)
        
    def get_data_threadings(self):
        import concurrent
        
        df_result = {}
        threadings_num = self.df_arg_dict.__len__()
        with concurrent.futures.ThreadPoolExecutor(threadings_num) as executor:
            futures = {}
            for key, value in self.df_arg_dict.items():
                func = value['func']
                args = value['args']
                futures[key] = executor.submit(self.execute_func, key, func, args)
            
            for key, future in futures.items():
                df_result[key] = future.result()
            del futures[key]
            gc.collect()  # 调用垃圾回收器释放内存
        return df_result
    
    async def get_data_async(self, key, func, args):
        with self.app.app_context():
            result = await self.loop.run_in_executor(None, func, *args)
        return key, result
    
    async def get_all_data_async(self):
        tasks = []
        for key, value in self.df_arg_dict.items():
            func = value['func']
            args = value['args']
            task = asyncio.create_task(self.get_data_async(key, func, args))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        df_result = {key: result for key, result in results}
        return df_result
    
    def get_data_coroutines(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        df_result = self.loop.run_until_complete(self.get_all_data_async())
        return df_result
    