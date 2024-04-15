'''
2NSDU20521004276
'''
import os
import time
import uiautomator2 as u2
import numpy as np 
# 连接手机#
def connect_phone(device_name):
    d = u2.connect(device_name)
    if not d.service("uiautomator").running():
    # 启动uiautomator服务
        print("start uiautomator")
        d.service("uiautomator").start()
        time.sleep(2)
    if not d.agent_alive:
        print("agent_alive is false")
        u2.connect()
        d = u2.connect(device_name)
    return d
def run(device_name):
    d = connect_phone(device_name)
    d.app_start("com.ccb.longjiLife")
    d.set_fastinput_ime(True)
    time_start = time.time()

    while True:
        start = time.time()
        time.sleep(1.5)
        if not d(textContains="再填一单").exists:
            screen_width, screen_height = d.window_size()
            # 定义滑动起始点和终点的坐标
            start_x = screen_width // 2  # 横坐标屏幕中点
            start_y = screen_height - 100  # 起始纵坐标在屏幕底部上方100像素处
            end_x = screen_width // 2  # 横坐标屏幕中点
            end_y = 100  # 终点纵坐标在屏幕顶部下方100像素处
            # 滑动屏幕
            d.swipe(start_x, start_y, end_x, end_y)
            time.sleep(1.0)
            d(textContains="再填一单").click()
        else:
            d(textContains="再填一单").click()
        time.sleep(0.5)
        if d(textContains="310501").exists:
            x0 = (396 + 1023) / 2  # 计算横坐标的中点
            y0 = (550+ 430 ) / 2  # 计算纵坐标的中点
            # 点击指定位置的标签
            time.sleep(0.5)
            d.click(x0, y0)
            d.clear_text()
            comp_str = (np.random.randint(0,1000))
            input_text = 31050170450000000000 + comp_str
            d.send_keys(str(input_text),True)
            time.sleep(0.5)
            if d(textContains="同收款单位").exists:
                input_block = d(textContains="同收款单位")
                input_block.click()
            time.sleep(0.5)
            if 1:
                # 定义要点击的位置的坐标
                x = (87 + 960) / 2  # 计算横坐标的中点
                y = (1106 + 1229) / 2  # 计算纵坐标的中点
                # 点击指定位置的标签
                d.click(x, y)
                d.clear_text()
                money_str = np.random.randint(0,15000)
                d.send_keys(str(money_str),True)
            time.sleep(0.5)
            if 2:
                # 定义要点击的位置的坐标
                x1 = (405 + 505) / 2  # 计算横坐标的中点
                y1 = (1242 + 1306) / 2  # 计算纵坐标的中点
                # 点击指定位置的标签
                d.click(x1, y1)
                d.clear_text()
                list = ['货款' ,'营业款','现金','划款','转账','租金','中介费']
                target_str = np.random.choice(list)
                d.send_keys(target_str,True)
            time.sleep(0.5)
            if d(textContains="下一步").exists:
                d(textContains="下一步").click()


        print("本次花费时间:", time.time() -start)
        print("总共花费时间:", (time.time() -time_start) / 60)

if __name__ == '__main__':
    # 此处填设备编号：由1024我的小表妹原创
    # device_name = "2NSDU20521004276"
    '''
    无法启动的时候输入指令
    1、adb shell am clear-debug-app
    2、adb start-server
    '''
    device_name =  'kjpzgannir7lireu'
    run(device_name)


