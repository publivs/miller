import seaborn as sns
import pyecharts as chart
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# from pyecharts import Bar


df = pd.DataFrame([[-1,-2,3,4,5,6,7,8,9,10]])
df.columns = ['A50','RUS','HS300','d','e','f','g','h','i','j']




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
            chart.text(col_num,v_i+0.5,str(v_i).join('%'),ha='center')
        else:
            chart.text(col_num,v_i-0.5,str(v_i),ha='center')
    plt.show()

def get_trading_volume(df):
    def autolabel(rects):
        """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

    labels = ['第一项', '第二项']
    a = [4.0, 3.8]
    b = [26.9, 48.1]
    c = [55.6, 63.0]
    d = [59.3, 81.5]
    e = [89, 90]    

    x = np.arange(len(labels))  # 标签位置
    width = 0.1  # 柱状图的宽度，可以根据自己的需求和审美来改

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width*2, a, width, label='a')
    rects2 = ax.bar(x - width+0.01, b, width, label='b')
    rects3 = ax.bar(x + 0.02, c, width, label='c')
    rects4 = ax.bar(x + width+ 0.03, d, width, label='d')
    rects5 = ax.bar(x + width*2 + 0.04, e, width, label='e')

    # 为y轴、标题和x轴等添加一些文本。
    ax.set_ylabel('Y轴', fontsize=16)
    ax.set_xlabel('X轴', fontsize=16)
    ax.set_title('这里是标题')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()

    plt.show()
