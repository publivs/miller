from numpy import empty
import pandas as pd
import numpy as np
import re
# import bs4
import json
import requests
import time

from hdf_helper import *

def get_url_data(table_id,req_headers,base_url,table_url,data_struc_type='0',dict_sub_key = ''):

        con_continnue = True
        while con_continnue:
            # try:
            #     res_ = requests.get(base_url+table_url,headers=req_headers)
            try:
                res_ = requests.get(base_url+table_url,headers=req_headers)
                if res_ is not None:
                    con_continnue = False
                else:
                    time.sleep(5)
                    res_ = requests.get(base_url+table_url,headers=req_headers)
            except Exception as e:
                print("链接,出异常了！")

        res_.encoding = 'utf-8'
        if not res_.text.__len__() == 0:
        # 标准的DF类型
            if data_struc_type == '0':
                res = json.loads(res_.text)
                res_df = pd.DataFrame(res)

            # 标准的DataFrame类型
            if data_struc_type == '1':
                res = json.loads(res_.text)
                if not dict_sub_key:
                    res_df = pd.DataFrame.from_dict(res,orient = 'index')
                else:
                    res_df = pd.DataFrame.from_dict(res[dict_sub_key],orient = 'index')

        # 返回的html静态类型
            if data_struc_type == '2':
                try:
                    res_df = pd.read_html(res_.text)[0]
                except:
                    print(base_url+table_url,'html出错......')
                    res_df = pd.DataFrame()
                    return res_df

            return res_df
        else:
            return pd.DataFrame()

def get_all_table_info(table_id,header,base_url):

    table_info_url = f'/api/table/{table_id}'
    column_url = f'/api/column/{table_id}'
    slave_column_url = f'/api/slaveColumn/{table_id}'
    table_index_unique_url = f'/api/tableIndexByUnique/{table_id}'
    table_change_desc_url = f'/api/tableChangeDesc/query/{table_id}'
    table_modify_date_url = f'/api/getTableModifyDate/{table_id}'

    example_info_url =  f'/api/exampleData/readExampleHtml/{table_id}'

    Q_A_info_url = f'/api/qa/query/{table_id}'

    table_info_df = get_url_data(table_id,header,base_url,table_info_url,'1','data').T
    column_info_df = get_url_data(table_id,header,base_url,column_url)
    slave_column_info_df = get_url_data(table_id,header,base_url,slave_column_url)
    table_index_unique_df = get_url_data(table_id,header,base_url,table_index_unique_url,'1').T
    table_change_desc_df = get_url_data(table_id,header,base_url,table_change_desc_url)

    table_modify_date_df = get_url_data(table_id,header,base_url,table_modify_date_url,'1').T

    example_info_df = get_url_data(table_id,header,base_url,example_info_url,'2')
    Q_A_info_df = get_url_data(table_id,header,base_url,Q_A_info_url)
    # 涉及的QA部分的数据
    return table_info_df,column_info_df,slave_column_info_df,table_index_unique_df,table_change_desc_df,table_modify_date_df,example_info_df,Q_A_info_df



def catch_data_main(table_info,req_headers,base_url,h5_group_path_rela):
    table_id = table_info['id']
    table_name = table_info['tableName']

    table_info_df,\
    column_info_df,\
    slave_column_info_df,\
    table_index_unique_df,\
    table_change_desc_df,\
    table_modify_date_df,example_info_df,Q_A_info_df = get_all_table_info(table_id,req_headers,base_url)

    # 根文件夹

    # ------------------------------------------ 下面的部分用递归改变 ------------------------------------------ #
    # 创建文件夹
    # data_set_path = i_dataset_name['groupName']
    # data_set_path_rela = all_data_set_path+'\\'+data_set_path
    # if not os.path.exists(data_set_path_rela):
    #     os.mkdir(data_set_path_rela)

    # # 创建一个文件夹
    # group_path = i_group_info['groupName']
    # group_path_rela = all_data_set_path+'\\'+data_set_path+'\\'+group_path
    # if not os.path.exists(group_path_rela):
    #     os.mkdir(group_path_rela)



    # # 聚源一张表就一个HDFS对象
    # table_name = i_table_info['tableName']
    # table_cn_name = i_table_info['groupName']
    # h5_group_path_rela = all_data_set_path+'\\'+data_set_path+'\\'+group_path+'\\'+table_name+'_'+table_cn_name

    h5_client = h5_helper(h5_group_path_rela+'_'+'table'+'.h5')

    if not table_info_df.empty:
        table_info_df = table_info_df.astype('str')
        h5_client.append_table(table_info_df,'table_info_df')

    if not column_info_df.empty:
        h5_client.append_table(column_info_df,'column_info_df')

    if not slave_column_info_df.empty:
        h5_client.append_table(slave_column_info_df,'slave_column_info_df')

    if not table_index_unique_df.empty:
        table_index_unique_df = table_index_unique_df.astype('str')
        h5_client.append_table(table_index_unique_df,'table_index_unique_df')

    if not table_change_desc_df.empty:
        h5_client.append_table(table_change_desc_df,'table_change_desc_df')

    if not table_modify_date_df.empty:
        h5_client.append_table(table_modify_date_df,'table_modify_date_df')

    if not example_info_df.empty:
        h5_client.append_table(example_info_df,'example_info_df')

    # ---------------------------------------------------------------------------------- #
    def update_qa_info(table_id,table_name,qa_table_name,h5_group_path_rela,Q_A_info_df):
        str_q = '问题:'+ Q_A_info_df.qaQuestion + '\n'
        qa_table_name = '该问题涉及的表单:' + qa_table_name +'\n'
        str_a = '回答:' + Q_A_info_df.qaAnswer +'\n'
        str_time = '最后修改的时间:'+ Q_A_info_df.lastModifiedDate
        str_qa_info = str_q+qa_table_name+str_a+str_time + ' \n \n \n'

        # # # 写入部分 # # #
        with open(f'''{h5_group_path_rela}_QA.txt''','a',encoding='utf-8') as f:
            f.write(str_qa_info)
        return str_qa_info

    if Q_A_info_df.__len__() != 0:
        # Q_A_info_df = Q_A_info_df.loc[:,~Q_A_info_df.columns.str.contains('tables')]
        for no_ in range(Q_A_info_df.__len__()):
            em_str = ''
            qa_table_name = [em_str + i['tableName'] + '' for i in Q_A_info_df.iloc[0]['tables']].__str__()[1:-1]
            update_qa_info(table_id,table_name,qa_table_name,h5_group_path_rela,Q_A_info_df.iloc[no_])


def initial_file_path(tree_nodes,
                    res_path = 'Juyuan_datafile',
                    next_level_key = 'nodes',
                    path_key = 'groupName',
                    table_name_key  = 'tableName',
                    table_id_key = 'id'):
    #
    initial_path = res_path
    next_level_key = next_level_key
    path_key = path_key
    table_name_key = table_name_key
    table_id_key = table_id_key

    # 传进来是列表对象就继续遍历子节点
    if isinstance(tree_nodes,list):
            for i in  range(tree_nodes.__len__()):
                    sub_node_i = tree_nodes[i]
                    res_path = initial_file_path(sub_node_i,res_path,next_level_key,path_key,table_name_key)

    # 不是列表继续搜索,搜到列表位置
    else:
        write_mark = '0'
        if tree_nodes[next_level_key] is None:
            write_mark = '1'
        else:
            if (tree_nodes[next_level_key].__len__() > 0):
                pass
            else:
                write_mark = '1'

        if write_mark == '0':
            if res_path is None:
                res_path = initial_path
            append_path  = '\\'+ tree_nodes[path_key]
            append_path = append_path.replace('/','_')
            res_path = res_path + append_path
            print(res_path)
            sub_nodes = tree_nodes[next_level_key]
            if not os.path.exists(res_path):
                os.mkdir(res_path)
            res_path = initial_file_path(sub_nodes,res_path,next_level_key,path_key,table_name_key)
            return res_path

        # nodes一般是大于0的，如果为0那么基本是遍历到基层的子节点了,在这里实例化HDFS对象存数据
        else:
            # print('准备实例化_hdf5!!!')
            table_id = tree_nodes['id']
            # obj_id = tree_nodes['OBJ_ID']
            table_name = tree_nodes['tableName']
            table_cn_name = tree_nodes['groupName']
            if table_cn_name is None:
                pass
            h5_path = f'''{res_path}\\{table_name}_{table_id}'''
            print(h5_path)
            # sleep_time = time.sleep(np.random.randint(3,4))
            return res_path

    last_path = '\\'+res_path.split('\\')[-1:][0]
    res_path = res_path.replace(last_path,'')
    return res_path





# -------------------------------------------------------------------------------------- #

base_url = 'https://dd.gildata.com/'

cookie = '''rememberMeFlag=true; sSerial=TVRBNmRIQjVlbU56YW1zd09HZHBiR1JoZEdGQU1USXo%3D; SESSION=03fed8d9-4f80-4393-850d-72f90fa4a28b'''

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
all_dataset_name = res_all_tree[0]
# 数据集的path
all_data_set_path = 'Juyuan_datafile'
    # os.remove(all_data_set_path)
if  not os.path.exists(all_data_set_path):
        os.mkdir(all_data_set_path)

for i in range (len(all_dataset_name['nodes']) ):
    if i >= 11:
        initial_file_path(all_dataset_name['nodes'][i])
# 获取该对应第0个data_set 的信息
# for i in range(all_dataset_name.__len__()):
#     i_dataset_name = all_dataset_name[i]

# # 获取该dataset下,group_0对应表的信息
#     for j in range(i_dataset_name['nodes'].__len__()):
#         i_group_info  = i_dataset_name['nodes'][j]

# # 获取该group下对应的table_0的信息
#         for k in range(i_group_info['nodes'].__len__()):
#                 i_table_info = i_group_info['nodes'][k]
#                 no_ = i*100+j*10+k
#                 if no_ >710:
#                     print('*'*10+'执行到了,no_:  '+str(no_)+'*'*10)
#                     sleep_time = np.random.randint(3,6)
#                     table_id = i_table_info['id']
#                     table_name = i_table_info['tableName']
#                     table_cn_name = i_table_info['groupName']
#                     catch_data_main(table_id,req_headers,base_url)
#                     time.sleep(sleep_time)
#                 else:
#                     pass



# 上次执行的位置 710

# 执行的位置 550 546
# ------------------------------------------- 拆离 ----------------------------------------- #











# -------------------------------------- 分割线一定要长 ------------------------------------------- #

