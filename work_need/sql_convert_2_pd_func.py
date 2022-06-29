import pandas as pd
import numpy as np
import random

def update_str(str_need_sub):
    return str_need_sub.replace('\n','').replace('\t','').replace('  ','').replace('   ','')

def expend_shuffle(values_array,repeat_num = 6000000):
    repeat_times = repeat_num/values_array.__len__()
    values_array = np.repeat(values_array,repeat_times)
    np.random.shuffle(values_array)
    return values_array

def generate_test_df():
    detail_asset_code_new_label = [
                            502010201,704010201,701010201,704010101,101010101,
                            104020103,502060101,802010101,806010101,505010101,
                            805030101,805020101,505020101,'C01051001','C01015001',
                            ]

    currency_cd_new_lab = list(range(0,9))+[999]

    is_mezzanine = [np.nan,'N','Y']

    special_bussiness_type = [2,3,4,6,16,np.nan]
    currency_cd_new_label = ['1','2','3']
    stanproject_id = [
                    '3001W000000097',
                    '3001W000000041',
                    '3001W000000113',
                    '3001W000000031',
                    '3001W000000108',
                    '3001W000001833',
                    '4001W000000431',
                    '3001PAP0000298',
                    '3001PAP0000315',
                    '3001PAP0000298',
                    '3001PAP0000353',
                    '3001PAP0000407',
                    '3001PAP0000469',
                    '3001PAP0000504',
                    'W000010235',
                    '1001P2017024',
                    '1001P2017026',
                    '1001P2017025',
                    '1001P2017028',
                    '1001P2017027',]

    taa_security_id = ['601398.SH','601229.SH','A02105.PF','A02160.PF','A02129.PF','A02132.PF','A02133.PF','A02176.PF']
    node_id = [
        'N000000002',
        'N000000003',
        'N000000004',
        'N000000005',
        'N000001014',
        'N000002440',
                ]

    arrange_cmb = '3'
    security_id = ['A02856','A03159']
    special_business_type = [str(i) for i in list(range(1,21))]

    '601398.SH','601229.SH','A02105.PF','A02160.PF','A02129.PF','A02132.PF','A02133.PF','A02176.PF'

    data_matrix = pd.DataFrame(
        # 0,index=[0],columns=['node_id','stanproject_id',
        #                                     'special_bussiness_type','is_mezzanine',
        #                                     'currency_cd_new_lab','detail_asset_code_new']
                                )



    data_matrix['node_id'] = expend_shuffle(node_id)
    data_matrix['stanproject_id'] = expend_shuffle(stanproject_id)
    data_matrix['special_bussiness_type'] = expend_shuffle(special_bussiness_type)
    data_matrix['is_mezzanine'] = expend_shuffle(is_mezzanine)
    data_matrix['currency_cd_new_lab'] = expend_shuffle(currency_cd_new_lab)
    data_matrix['detail_asset_code_new_label'] = expend_shuffle(detail_asset_code_new_label)
    data_matrix['currency_cd_new_label'] = expend_shuffle(currency_cd_new_label)
    data_matrix['security_id'] = expend_shuffle(security_id)
    data_matrix['special_business_type'] = expend_shuffle(special_business_type)
    data_matrix['taa_security_id'] = expend_shuffle(taa_security_id)
    data_matrix.columns  = [i.upper() for i in data_matrix.columns]

    df = data_matrix

    return df





# # convert_str_1


# # 第一部分的数据

# # 筛选 :DETAIL_ASSET_CODE_NEW_LABEL
# query_str_A = '''(DETAIL_ASSET_CODE_NEW_LABEL in  (['807010101',
#                                                 '803010101',
#                                                 '803020101',
#                                                 '803030101',
#                                                 '805010101',
#                                                 '806010101',
#                                                 '505070101',
#                                                 '505140101',
#                                                 '505160101']))
#                                                 '''
# query_str_A = update_str(query_str_A)
# A = df.query(query_str_A)

# #
# # 筛选 currency_cd_new_label
# B = A.query('''CURRENCY_CD_NEW_LABEL == '1' ''')

# # 第一个case_when,求并的补
# C_str = '''~( (IS_MEZZANINE == 'Y' )
#             |(SECURITY_ID in (['A02856','A03159']))
#             )'''
# C_str = update_str(C_str)
# C = B.query(C_str)


# # 第二个case_when,求并的补
# D_str =   ''' ~(  (
#             SPECIAL_BUSINESS_TYPE in (['8', '12', '13', '15'])

#             & ( ('Y') in IS_MEZZANINE|('Y') in IS_MEZZANINE )

#             & ( CURRENCY_CD_NEW_LABEL in (['1']) )
#             )
#             |(STANPROJECT_ID in (['1001P2017029',
#                                     '1001P2017024',
#                                     '1001P2017026',
#                                     '1001P2017025',
#                                     '1001P2017028',
#                                     '1001P2017027',
#                                     '1001P2017007',
#                                     '1001P2020005',
#                                     '1001P2019002',
#                                     '1001P2020002',
#                                     '1001P2019006'])
#             )
#             )'''
# D_str = update_str(D_str)
# D = C.query(D_str)

# # 第一部分数据完


# # convert_str_2
# # 第二部分的结构 OR后面的
# A2_str = '''NODE_ID in 	(['N00000004',
#                            'N00000005',
#                            'N00000003',
#                            'N00000002',
#                            'N00001014'])'''
# A2_str = update_str(A2_str)
# A2 = df.query(A2_str)

# B2_str = '''CURRENCY_CD_NEW_LABEL in (['1']'''
# B2_str = df.query(B2_str)
# B2 = A2.query(B2_str)

# C2 = B2.loc[(
#             B2.SPECIAL_BUSINESS_TYPE.isin(['8', '12', '13', '15'])
#             & ( B2.IS_MEZZANINE.str.contains('Y')|B2.IS_MEZZANINE.str.contains('Y')|B2.IS_MEZZANINE.str.contains('Y') )
#             & ( B2.CURRENCY_CD_NEW_LABEL.isin(['1']) )
#                 )
#             |(B2.STANPROJECT_ID.isin(['1001P2017029',
#                                     '1001P2017024',
#                                     '1001P2017026',
#                                     '1001P2017025',
#                                     '1001P2017028',
#                                     '1001P2017027',
#                                     '1001P2017007',
#                                     '1001P2020005',
#                                     '1001P2019002',
#                                     '1001P2020002',
#                                     '1001P2019006']))
#                                     ]
# # 第二部分数据完


# # rel为0 为概述盖层的第一个


