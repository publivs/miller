import json
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import bs4
import requests
import pytesseract
from PIL import Image
import cv2 as cv
import numpy as np

chrome_options = Options()
chrome_options.add_argument('window-size=1920x3000') #指定浏览器分辨率
chrome_options.add_argument('--disable-gpu') #谷歌文档提到需要加上这个属性来规避bug
chrome_options.add_argument('--hide-scrollbars') #隐藏滚动条, 应对一些特殊页面
chrome_options.add_argument('--ignore-certificate-errors') #忽略一些莫名的问题
chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])  # 开启开发者模式
chrome_options.add_argument('--disable-blink-features=AutomationControlled')  # 谷歌88版以上防止被检测
# chrome_options.add_argument('blink-settings=imagesEnabled=false') #不加载图片, 提升速度

chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36')

# chrome_options.add_argument('--headless') # 浏览器不提供可视化页面. linux下如果系统不支持可视化不加这条会启动失败,可视化带ui的正常使用,方便调试

driver = webdriver.Chrome(options=chrome_options)  # 将chromedriver放到Python安装目录Scripts文件夹下

options = webdriver.ChromeOptions()

def connect_url(target_url,req_headers):
    con_continnue = True
    while con_continnue:
        try:
            res_ = requests.get(target_url,headers=req_headers)
            if res_ is not None:
                con_continnue = False
            else:
                time.sleep(5)
                res_ = requests.get(target_url,headers=req_headers)
        except Exception as e:
            print("链接,出异常了！")
    return res_

# Grayscale image
def recognize_captcha(file_path):
    img = Image.open(file_path).convert('L')

    ret,img = cv.threshold(np.array(img), 125, 255, cv.THRESH_BINARY)
    # img = cv.morphologyEx(np.array(img),cv.MORPH_CLOSE,np.ones(shape=(6,6)))

    img = Image.fromarray(img.astype(np.uint8))
    res = pytesseract.image_to_string(img)
    res = res.replace(' ','').replace('\n','')
    print(res)
    return res

def check_pic_res():
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'lxml')
    # target_table  = soup.find_all('ant-modal-content')
    target_soup = soup.select('body > div.team > form > div > div > div.form-group > div.col-sm-7.col-xs-7.tradition')
    target_table = target_soup[0]
    target_str = target_table.contents[0]
    import re
    rd_str = re.findall(r"randomStr=(.*?)&",str(target_str))[0]
    timestamp = re.findall(r'''timestamp=(.*?)"''',str(target_str))[0]
    req_headers = {
                    'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                    'Connection':'keep-alive',
                    'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
                    }
    target_url = f"https://poll.cnfic.com.cn/financier/captcha/getCaptchaImg?randomStr={rd_str}&timestamp={timestamp}"
    res = connect_url(target_url,req_headers)
    res = res.content
    file_path = r"C:\Users\kaiyu\Desktop\miller\test_cath_data\test_1" + ".png"
    playFile = open(file_path, 'wb')
    playFile.write(res)
    playFile.close()

    pic_res = recognize_captcha(file_path)
    print(pic_res)
    return pic_res


driver.get('https://poll.cnfic.com.cn/vote2022/index.html')  # 此处不要再放登录的网址，可以用未登录的首页

driver.refresh()

target_path_16 = '''/html/body/div[2]/form/div/div/div[1]/div[16]/div[2]/div/input'''
target_a_btn = driver.find_element(by=By.XPATH, value=target_path_16)
time.sleep(1)
target_a_btn.click()
print('16 is choosed')

target_path_14 = '''/html/body/div[2]/form/div/div/div[1]/div[14]/div[2]/div/input'''
target_b_btn = driver.find_element(by=By.XPATH, value=target_path_14)
time.sleep(1)
target_b_btn.click()
print('14 is choosed')

pp_html = driver.page_source

pic_path = '''/html/body/div[2]/form/div/div/div[2]/div[2]/img'''
target_pic_btn = driver.find_element(by=By.XPATH, value=pic_path)
target_pic_btn.click()
print('pit tick')

pic_res = check_pic_res()

if '=?' in pic_res:
    do_next = 0
    target_str = pic_res[:-2]
    target_str = target_str.replace('=','')
    ret_res = str(eval(target_str))
    user_input = driver.find_element(by=By.XPATH, value='/html/body/div[2]/form/div/div/div[2]/div[1]/input')
    user_input.send_keys(ret_res)

entry = '''/html/body/div[2]/form/div/div/div[2]/div[4]/button'''
target_3_btn = driver.find_element(by=By.XPATH, value=entry)
target_3_btn.click()

last = '''/html/body/div[2]/form/div/div/div[3]/div/div/div[3]/a'''
target_3_btn = driver.find_element(by=By.XPATH, value=last)
target_3_btn.click()


# cv.imshow('input image', src)
# recognize_text(src)
# print(src.shape)
# cv.waitKey(0)
# cv.destroyAllWindows()










print(1)
# 内容部分
