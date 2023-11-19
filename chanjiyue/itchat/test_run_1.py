import itchat
# 登录
itchat.auto_login()
itchat.run()

# 获取好友列表
friends = itchat.get_friends(update=True)
# 找到指定好友
friend = None
for item in friends:
    if item['RemarkName'] == '某某某':
        friend = item
        break
if friend:
    # 发送消息
    itchat.send('Hello, World!', toUserName=friend['UserName'])