import itchat
import pandas as pd
from copy import deepcopy
# 登录
from itchat.content import *



temp_cache_dict = {}

class AutoReporter:
    def __init__(self) -> None:
        import time,datetime
        self.date_ = datetime.datetime.now().date()

    def get_report_model(self):
        date_ = self.date_
        self.model_string = f'''九江路支行{date_.year}年\xa0{date_.month}月{date_.day}日：\n1、新开立借记卡 ? \n2、手机银行新签 ? \n3、转介码扫码关注 ? \n4、“上海建行”关注 ? \n5、“会员成长”参与 ? \n6、快捷绑卡有交易 ? \n7、一二类钱包活 ? \n8、到店客户企微添加 ? \n\xa0\xa0\xa0\xa0\xa0\xa0\xa0其中：‼️支行直营经理网点专属码\xa0: ? \n\xa0\xa0\xa0\xa0\xa0\xa0网点营销人员企微 ? \n9、养老金账户开立 ? \n10、CTS码上通开立 ? '''
        return self.model_string


def generate_send_string():
    return_text = f''' '''

@itchat.msg_register([TEXT,NOTE])
def text_reply(msg):
    response = deepcopy(msg)
    res_string = response.Text
    # 车牌标记
    target_string = '我通过了你的朋友验证请求'
    if target_string in res_string:
        ret_msg = '我是客户经理产小月'
        itchat.send_msg(ret_msg, response['User']['UserName'])

    report_string ='''报业绩'''
    if report_string in res_string:
            report_ = AutoReporter()
            temp_cache_dict[f'''daily_repoct_{report_.date_.strftime('%Y%m%d')}''']  = report_
            itchat.send_msg(report_.get_report_model(), response['User']['UserName'])
            temp_msg = '我是客户经理产小月' + ',' + '请按照模板输入对应数字,' +'开头请输入:报业绩2'
            itchat.send_msg(temp_msg, response['User']['UserName'])

    report_string ='''报业绩2'''
    if report_string in res_string:
            report_ = AutoReporter()
            report_string = report_.get_report_model()
            replace_list = [ str_ for str_ in res_string]
            for item in replace_list:
                    report_string = report_string.replace('?', item, 1)
            print(report_string)
            itchat.send_msg(report_string, response['User']['UserName'])

@itchat.msg_register([TEXT,NOTE],isGroupChat=True)
def daily_task_msg(msg):
    response = msg
    return '收到'
# 登录
itchat.auto_login(hotReload=True)
# 开始接收微信消息
itchat.run()