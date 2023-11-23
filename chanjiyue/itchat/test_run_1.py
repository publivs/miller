import itchat
import pandas as pd
# 登录
from itchat.content import *
import time
def generate_send_string():
    return_text = f''' '''

@itchat.msg_register([TEXT,NOTE])
def text_reply(msg):
    response = msg
    # 车牌标记
    target_string = '我通过'
    if target_string in msg.Text:
        itchat.send_msg(u"[%s]收到好友@%s 的信息：%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S",
                                                            time.localtime(msg['CreateTime'])),
                                                            msg['User']['NickName'],
                                                            msg['Text']),
                                                            'filehelper')
        print(msg)


@itchat.msg_register([TEXT,NOTE],isGroupChat=True)
def daily_task_msg(msg):
    response = msg
    return '收到'
# 登录
itchat.auto_login(hotReload=True)
# 开始接收微信消息
itchat.run()