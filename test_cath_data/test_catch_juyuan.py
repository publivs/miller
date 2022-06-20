import pandas as pd
import re
# import bs4
import json
import requests


base_url = 'https://dd.gildata.com/'

cookie = '''rememberMeFlag=true; sSerial=TVRBNmRIQjVlbU56YW1zd09HZHBiR1JoZEdGQU1USXo%3D; SESSION=4c5cf2d2-c0b9-4322-8e15-82e8881d7db9'''
# Override the default request headers:
req_headers = {
                'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en',
                'Cookie':cookie,
                }

# 全分支的节点树
all_tree_url = f'/api/productGroupTreeWithTables/8/802/ALL_TREE'


# 获取主节点的信息

res_all_tree = requests.get(base_url+all_tree_url,headers=req_headers)
res_all_tree = json.loads(res_all_tree.text)

# 获取所有data_set的信息
all_dataset_name = res_all_tree[0]['nodes']

# 获取该对应第i个data_set 的信息
i_dataset_name = all_dataset_name[0]

# 获取该dataset下,group_i对应表的信息
i_group_info  = i_dataset_name['nodes'][0]

# 获取该group下对应的table的信息
i_table_info = i_group_info['nodes'][0]



# ------------------------------------------- 拆离 ---------------------------------------- #


table_id = i_table_info['id']

# 对应的表信息
# column_url = f'/api/column/{table_id}'
# slave_column_url = f'/api/slaveColumn/{table_id}'
# table_index_unique_url = f'/api/tableIndexByUnique/{table_id}'
# table_change_desc_url = f'/api/tableChangeDesc/query/{table_id}'
# table_modify_date_url = f'/api/getTableModifyDate/{table_id}'

# 样例数据
# example_info_url =  f'/api/exampleData/readExampleHtml/{table_id}'

# QA数据
# Q_A_info_url = f'/api/qa/query/{table_id}'

# res_ = requests.get(base_url+column_url,headers=req_headers)
# res_.encoding = 'utf-8'
# res = json.loads(res_.text)


def get_all_table_info(table_id,header,base_url):
    def get_url_data(table_id,req_headers,base_url,table_url,data_struc_type='0'):
        res_ = requests.get(base_url+table_url,headers=req_headers)
        res_.encoding = 'utf-8'

        # 标准的DF类型
        if data_struc_type == '0':
            res = json.loads(res_.text)
            res_df = pd.DataFrame(res)

        # 标准的DataFrame类型
        if data_struc_type == '1': 
            res = json.loads(res_.text)
            res_df = pd.DataFrame.from_dict(res,orient = 'index')

        # 返回的html静态类型
        if data_struc_type == '2':
            res_df = pd.read_html(res_.text)[0]

        return res_df

    column_url = f'/api/column/{table_id}'
    slave_column_url = f'/api/slaveColumn/{table_id}'
    table_index_unique_url = f'/api/tableIndexByUnique/{table_id}'
    table_change_desc_url = f'/api/tableChangeDesc/query/{table_id}'
    table_modify_date_url = f'/api/getTableModifyDate/{table_id}'

    example_info_url =  f'/api/exampleData/readExampleHtml/{table_id}'

    Q_A_info_url = f'/api/qa/query/{table_id}'

    column_df = get_url_data(table_id,header,base_url,column_url)
    slave_column_df = get_url_data(table_id,header,base_url,slave_column_url)
    table_index_unique_df = get_url_data(table_id,header,base_url,table_index_unique_url,'1')
    table_change_desc_df = get_url_data(table_id,header,base_url,table_change_desc_url)

    table_modify_date_df = get_url_data(table_id,header,base_url,table_modify_date_url,'1')

    example_info_df = get_url_data(table_id,header,base_url,example_info_url,'2')
    Q_A_info_df = get_url_data(table_id,header,base_url,Q_A_info_url)

    return column_df,slave_column_df,table_index_unique_df,table_change_desc_df,table_modify_date_df,example_info_df,Q_A_info_df

column_df,\
slave_column_df,\
table_index_unique_df,\
table_change_desc_df,\
table_modify_date_df,example_info_df,Q_A_info_df = get_all_table_info(table_id,req_headers,base_url)




