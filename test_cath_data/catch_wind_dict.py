from numpy import empty
import pandas as pd
import numpy as np
import re
# import bs4
import json
import requests
import time
import sys,os
# catch_handler

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(r"C:\Users\kaiyu\Desktop\miller")
from chenqian_tools.hdf_helper import *

def connect_tree_url(id):
     param = f'''%5B%22%5C%22{id}%5C%22%22%2C%22%5C%229490%5C%22%22%2C%22%5C%22%E6%8A%95%E5%86%B3%E9%A1%B9%E7%9B%AE%E7%BB%842%5C%22%22%2C%221%22%2C%22true%22%5D'''
     paylaod = {'way':'1','action':'2','interfaceId':'101','param':param,}
     res = requests.post(f'''{handler}''',data=paylaod,headers=req_headers)
     res = json.loads(res.text)
     return res

def connect_table_info_url(table_name):
     param = f'''%5B%22%5C%22{table_name}%5C%22%22%2C%22%5C%229490%5C%22%22%5D'''
     paylaod = {'way':'1','action':'2','interfaceId':'103','param':param,}
     res = requests.post(f'''{handler}''',data=paylaod,headers=req_headers)
     res = json.loads(res.text)
     return res

sleep_time = 0.1
cookie = '''
756427508369C797662551B7F6E4F407=C6CACBE14EA3FA51;A3F660A4EF3974D931B9F0B5DF001429=EF1A176E364C1077;9A6BF28197922FF55D8513E9ECCA188F=EF1A176E364C1077;2B4A55418B6F245261035C86D649790D=975C5CF37A9074B9CD6AD670C8F4469974A484ABCCB85602; 9A6BF28197922FF59BBE2239EA9F9CBA=7464E5EA30AFECDB; 551B07A7F9C056607698EEE32DBBD64F=D0A85E0373137FDD
'''
cookie = cookie.replace(' ','').replace('\n','')
# cookie = cookie.replace('\n','')
req_headers = {
                'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                'Cookie':cookie,
                'Connection':'keep-alive',
                'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
                }

handler = '''https://wds.wind.com.cn/rdf/Home/ServiceHandler'''
pay_load_query = {'way':'1','action' :'2','interfaceId':'103','params':'%255B%2522%255C%2522AShareIntroduction%255C%2522%2522%252C%2522%255C%25229490%255C%2522%2522%255D'}
res = requests.post(f'''{handler}''',data=pay_load_query,headers=req_headers)
res = json.loads(res.text)


menu_paylaod_pp = '''%5B%22%5C%22%20%5C%22%22%2C%22%5C%229490%5C%22%22%2C%22%5C%22%E6%8A%95%E5%86%B3%E9%A1%B9%E7%9B%AE%E7%BB%842%5C%22%22%2C%221%22%2C%22true%22%5D'''
menu_paylaod = {'way':'1','action':'2','interfaceId':'101','param':menu_paylaod_pp,}

# 获取基础的目录树
menu_res = requests.post(f'''{handler}''',data=menu_paylaod,headers=req_headers)
res = json.loads(menu_res.text)

def initial_path(tree_nodes,res_path ='wind_datafile',):

     if isinstance(tree_nodes,list):
          for tree_node in tree_nodes:
               initial_path(tree_node,res_path)

     if isinstance(tree_nodes,dict):
          if 'Content' in tree_nodes.keys():
               content = tree_nodes['Content']
               if 'Data' in content.keys():
                    data = content['Data']
                    initial_path(data,res_path)

          if 'DataType' in tree_nodes.keys():
               if tree_nodes['DataType'] == 1:
                    get_table_info_main(tree_nodes,res_path)
                    time.sleep(sleep_time)

               if tree_nodes['DataType'] == 0:
                    id = tree_nodes['ID']
                    data = connect_tree_url(id)
                    append_path  = '\\'+ tree_nodes['Text']
                    append_path = append_path.replace('/','_')
                    res_path = res_path + append_path
                    if  not os.path.exists(res_path):
                         os.mkdir(res_path)
                    initial_path(data,res_path)

def get_table_info_main(tree_nodes,res_path):

          def update_qa_info(QA,res_path):
               str_q = '问题:'+ QA['Name'] + '\n'
               str_a = '回答:' + QA['ContentNoHtml']  +'\n'
               str_qa_info = str_q+str_a + ' \n \n \n'
               # # # 写入部分 # # #
               with open(f'''{res_path}_QA.txt''','a',encoding='utf-8') as f:
                    f.write(str_qa_info)
               return str_qa_info

          ID_ = tree_nodes['ID']
          next_nodes = connect_table_info_url(ID_)
          table_data = next_nodes['Content']['Data']

          filed_df = pd.DataFrame(table_data['FieldList'])

          other_data = {k:v for k,v in table_data.items() if k not in ('FieldList','TopicList','SampleData')}
          other_df = pd.DataFrame.from_dict(other_data,orient = 'index').T
          other_df = other_df.astype('str')

          if table_data['SampleData'].__len__() > 0 :
               sample_data = json.loads(table_data['SampleData'])
               sample_df = pd.DataFrame(sample_data['rows'],columns=sample_data['fieldName'])
          else:
               sample_df = pd.DataFrame()

          append_path = '\\'+ tree_nodes['Text']
          append_path = append_path.replace('/','_')

          h5_path = res_path + append_path

          qa_data = table_data['TopicList']

          if qa_data is not None:
               for qa in qa_data:
                    update_qa_info(qa,h5_path)

          h5_group_path = h5_path +'.h5'
          if not os.path.exists(h5_group_path):
               h5_client = h5_helper(h5_group_path)
               h5_client.append_table(filed_df,'filed_df')
               h5_client.append_table(other_df,'other_df')
               h5_client.append_table(sample_df,'sample_df')

          print(h5_path)

all_data_set_path = 'wind_datafile'

if  not os.path.exists(all_data_set_path):
        os.mkdir(all_data_set_path)

nodes = res['Content']['Data']
initial_path(nodes,res_path =all_data_set_path,)
