import seaborn as sns
import pyecharts as chart
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from pyecharts import Bar


df = pd.DataFrame([[-1,-2,3,4,5,6,7,8,9,10]])
df.columns = ['A50','RUS','HS300','d','e','f','g','h','i','j']

# def autolabel(rects):
#     """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3点垂直偏移
#                     textcoords="offset points",
#                     ha='center', va='bottom')

def global_stock_index_chart(df):
    color_dict = {}
    for col in df.columns :
        color_dict[col] ='b'
        if col in ['A50','HS300']:
            color_dict[col] ='r'

    sns.set_style(rc={'font.sans-serif':"Microsoft Yahei"})
    chart = sns.barplot(df,palette=color_dict,width=0.5,)
    chart.axhline(0, color="k", clip_on=False)
    chart.tick_params(bottom=False,direction = 'in',width =1)
    sns.despine(bottom = True)

    for col_num in range(len(df.columns)):
        col_name = df.columns[col_num]
        v_i = df[col_name].values[0]
        if v_i >0:
            chart.text(col_num,v_i+0.5,str(v_i)+' %',ha='center')
        else:
            chart.text(col_num,v_i-0.6,str(v_i)+' %',ha='center')
    plt.show()

def get_turnover_rate(df):

    sns.set_style('white',rc={'font.sans-serif':"Microsoft Yahei"})
    labels = ['第一项', '第二项','第三项']
    no_1 = [1]
    no_2 = [2]
    no_3 = [1]
    no_4 = [2]
    no_5 = [1]
    no_6 = [2]

    x = np.arange(len(labels))  # 标签位置
    width = 0.1  # 柱状图的宽度，可以根据自己的需求和审美来改

    fig, ax = plt.subplots()

    # 进入循环
    rects1 = ax.bar(x - width*2, [1,2,3], width, label='第一周')
    rects2 = ax.bar(x - width+0.01, [1,2,3], width, label='第二周')
    rects3 = ax.bar(x + 0.02, [1,2,3], width, label='第三周')
    rects4 = ax.bar(x + width+ 0.03, [1,2,3], width, label='第四周')
    rects5 = ax.bar(x + width*2 + 0.04, [1,2,3], width, label='第五周')
    rects6 = ax.bar(x + width*2 + 0.15, [1,2,3], width, label='第六周')
    # 循环退出
    # 为y轴、标题和x轴等添加一些文本。
    # ax.set_ylabel('换手率', fontsize=16)
    # ax.set_xlabel('X轴', fontsize=16)
    ax.set_title('换手率簇柱图(%)')
    ax.set_xticks(x)
    # ax.set_yticks()
    ax.yaxis.grid(True, color ="black")
    ax.set_xticklabels(labels)
    sns.despine(left = True)

    fig.tight_layout()
    plt.legend(bbox_to_anchor=(0.5,-0.25),loc=8, borderaxespad=1,ncol = 6,frameon=False,fancybox=True)
    plt.show()

def get_trading_volume(df):

    sns.set_style('white',rc={'font.sans-serif':"Microsoft Yahei"})
    labels = ['第一项', '第二项','第三项']
    no_1 = [1]
    no_2 = [2]
    no_3 = [1]
    no_4 = [2]
    no_5 = [1]
    no_6 = [2]

    x = np.arange(len(labels))  # 标签位置
    width = 0.1  # 柱状图的宽度，可以根据自己的需求和审美来改

    fig, ax = plt.subplots()

    # 进入循环
    rects1 = ax.bar(x - width*2, [1,2,3], width, label='第一周')
    rects2 = ax.bar(x - width+0.01, [1,2,3], width, label='第二周')
    rects3 = ax.bar(x + 0.02, [1,2,3], width, label='第三周')
    rects4 = ax.bar(x + width+ 0.03, [1,2,3], width, label='第四周')
    rects5 = ax.bar(x + width*2 + 0.04, [1,2,3], width, label='第五周')
    rects6 = ax.bar(x + width*2 + 0.15, [1,2,3], width, label='第六周')
    # 循环退出
    # 为y轴、标题和x轴等添加一些文本。
    # ax.set_ylabel('换手率', fontsize=16)
    # ax.set_xlabel('X轴', fontsize=16)
    ax.set_title('换手率簇柱图(%)')
    ax.set_xticks(x)
    # ax.set_yticks()
    ax.yaxis.grid(True, color ="black",width = 0.5)
    ax.set_xticklabels(labels)
    sns.despine(left = True)

    fig.tight_layout()
    plt.legend(bbox_to_anchor=(0.5,-0.25),loc=8, borderaxespad=1,ncol = 6,frameon=False,fancybox=True)
    plt.show()

def get_finance_chart(df):

    sns.set_style('white',rc={'font.sans-serif':"Microsoft Yahei"})
    mv = [100,200,300,100,500,800,700]
    mv.sort(reverse=True)
    finance = [1,2,3,4,5,6,7]
    date_lst = pd.date_range('20221001','20221007').astype('str')

    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.plot(date_lst,mv,label='两融余额占A股流通市值(%)',color='red',lw=3)
    plt.legend(bbox_to_anchor=(0.2,-0.4),loc=8,borderaxespad=1,ncol = 6,frameon=False,fancybox=True,fontsize = 10)
    #横坐标显示的设置一定要在建立双坐标轴之前
    plt.xticks(date_lst,rotation=45)
    ax1.yaxis.grid(True, color ="black")
    ax2=ax1.twinx()
    ax2.plot(date_lst,finance,label='融资融券余额(沪深两市)',color='blue')
    ax2.tick_params(right=False)
    plt.fill_between(x=date_lst, y1=1, y2=finance, facecolor='blue',alpha =0.5)
    sns.despine(left = True)
    plt.legend(bbox_to_anchor=(0.8,-0.4),loc=8,borderaxespad=1,ncol = 6,frameon=False,fancybox=True,fontsize = 10)
    plt.show()

def get_finance_chart(df):

    sns.set_style('white',rc={'font.sans-serif':"Microsoft Yahei"})
    mv = [100,200,300,100,500,800,700]
    mv.sort(reverse=True)
    finance = [1.5,2,3,2,3,2,3]
    date_lst = pd.date_range('20221001','20221007').astype('str')

    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.plot(date_lst,mv,label='两融余额占A股流通市值(%)',color='red',lw=3)
    plt.legend(bbox_to_anchor=(0.2,-0.4),loc=8,borderaxespad=1,ncol = 6,frameon=False,fancybox=True,fontsize = 10)
    #横坐标显示的设置一定要在建立双坐标轴之前
    plt.xticks(date_lst,rotation=45)
    ax1.yaxis.grid(True, color ="black")
    ax2=ax1.twinx()
    ax2.plot(date_lst,finance,label='融资融券余额(沪深两市)',color='blue')
    ax2.tick_params(right=False)
    plt.fill_between(x=date_lst, y1=1, y2=finance, facecolor='blue',alpha =0.5)
    sns.despine(left = True)
    plt.legend(bbox_to_anchor=(0.75,-0.4),loc=8,borderaxespad=1,ncol = 6,frameon=False,fancybox=True,fontsize = 10)
    plt.show() 

def get_fund_volume(date_lst,combine_fund_vol,stock_fund_vol):

    sns.set_style(rc={'font.sans-serif':"Microsoft Yahei"})
    sns.lineplot(x = date_lst,y=stock_fund_vol,label = '股票基金发行份额')
    sns.lineplot(x = date_lst,y=combine_fund_vol,label = '混合型基金发行份额')
    plt.xticks(date_lst,rotation=90)
    plt.legend(bbox_to_anchor=(0.5,-0.5),loc=8, borderaxespad=1,ncol = 6,frameon=False,fancybox=True)
    plt.show()