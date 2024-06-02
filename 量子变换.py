import math
import matplotlib.pyplot as plt
import numpy as np
import akshare as ts


# 函数定义
def self_period(data, period):
    f = np.zeros(period)
    count = np.zeros(period)
    for i in range(len(data)):
        f[i%period] += data[i]
        count[i%period] += 1
    f = f/count
    for i in range(len(data)):
        data[i] -= f[i%period]
    return f
days = 5
def sim2self(x):
    data = x.copy()
    f2 = self_period(data, 2)
    f5 = self_period(data, 5)
    f11 = self_period(data, 11)
    f21 = self_period(data, 21)
    f31 = self_period(data, 31)
    f61 = self_period(data, 61)
    y = np.zeros(len(data)+days)
    for i in range(len(data)+days):
        y[i] = f2[i%2] + f5[i%5] + f11[i%11] + f21[i%21] + f31[i%31] + f61[i%61]
    return y
stock ="002457"
# 通过股票代码获取股票数据,这里没有指定开始及结束日期
df = ts.stock_zh_a_hist(symbol=stock , period='daily', start_date='20231001')
# 通过板块名称获取板块数据,这里没有指定开始及结束日期
#df = ts.stock_board_industry_index_ths(symbol="半导体及元件")
info = ts.stock_individual_info_em(symbol=stock)

# 数据准备
closes = np.array(df['收盘'].astype('float32')) 
opens = np.array(df['开盘'].astype('float32')) 
highs = np.array(df['最高'].astype('float32')) 
lows = np.array(df['最低'].astype('float32')) 

# 设置画布大小和标题
fig = plt.figure(figsize=(12,6))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig.suptitle(info.loc[5,'value'] + '一周预测', fontsize=14)#

# 添加蜡烛图子图
ax1 = fig.add_subplot(111)

# 设置x轴和y轴标签和范围
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.set_xlim(-1,len(closes)+days)
ax1.set_ylim(min(lows)*0.99, max(highs)*1.01)

# 绘制蜡烛图，红涨绿跌
for i in range(len(closes)):
    if closes[i] >= opens[i]:
        ax1.bar(i, closes[i]-opens[i], bottom=opens[i], color='r', width=0.618, align='center')
        ax1.bar(i, highs[i]-lows[i], bottom=lows[i], color='r', width=0.1, align='center')
    else:
        ax1.bar(i, opens[i]-closes[i], bottom=closes[i], color='g', width=0.618, align='center')
        ax1.bar(i, highs[i]-lows[i], bottom=lows[i], color='g', width=0.1, align='center')

# 绘制预测结果，用褪色柱形表示
closep = sim2self(closes[1:]/closes[:-1])
closep[-1-days] = closes[-1]
for i in range(days):
    closep[-days+i] *= closep[-days-1+i]
closep = closep[-days:]
openp = sim2self(opens[1:]/closes[:-1])[-days:] * closep
highp = sim2self(highs[1:]/closes[:-1])[-days:] * closep
lowp = sim2self(lows[1:]/closes[:-1])[-days:] * closep
for i in range(len(closep)):
    if closep[i] >= openp[i]:
        ax1.bar(i+len(closes), closep[i]-openp[i], bottom=openp[i], color='coral', width=0.618, align='center')
        ax1.bar(i+len(closes), highp[i]-lowp[i], bottom=lowp[i], color='coral', width=0.1, align='center')
    else:
        ax1.bar(i+len(closes), openp[i]-closep[i], bottom=closep[i], color='lime', width=0.618, align='center')
        ax1.bar(i+len(closes), highp[i]-lowp[i], bottom=lowp[i], color='lime', width=0.1, align='center')

# 显示图像
plt.show()
plt.savefig('C:/Users/Wu/Downloads/week.jpeg')
