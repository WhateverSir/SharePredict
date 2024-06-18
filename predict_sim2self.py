import matplotlib.pyplot as plt
import numpy as np
import akshare as aks
import argparse
# 定义命令行参数  
parser = argparse.ArgumentParser(description='Predict stock data based on sim2self.')  
parser.add_argument('--stock', type=str, default='601127', help='Stock code to predict.')  
parser.add_argument('--args.days', type=int, default=5, help='The number of args.days to predict.')  
  
# 解析命令行参数  
args = parser.parse_args()
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

def sim2self(x):
    data = x.copy()
    f7 = self_period(data, 7)    
    f15 = self_period(data, 15)
    f22 = self_period(data, 22)
    f31 = self_period(data, 31)
    f43 = self_period(data, 43)
    f61 = self_period(data, 61)
    y = np.zeros(len(data)+args.days)
    for i in range(len(data)+args.days):
        y[i] = f7[i%7] + f22[i%22] + f15[i%15] + f31[i%31] + f43[i%43] + f61[i%61]
    return y

# 通过股票代码获取股票数据
df = aks.stock_zh_a_hist(symbol=args.stock , period='daily', start_date='20231001')
# 通过股票代码获取股票信息
info = aks.stock_individual_info_em(symbol=args.stock)

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
ax1.set_xlim(len(closes)-101, len(closes)+args.days)
ax1.set_ylim(min(lows)*0.99, max(highs)*1.01)

# 绘制蜡烛图，红涨绿跌
for i in range(len(closes)-100, len(closes)):
    if closes[i] >= opens[i]:
        ax1.bar(i, closes[i]-opens[i], bottom=opens[i], color='r', width=0.618, align='center')
        ax1.bar(i, highs[i]-lows[i], bottom=lows[i], color='r', width=0.1, align='center')
    else:
        ax1.bar(i, opens[i]-closes[i], bottom=closes[i], color='g', width=0.618, align='center')
        ax1.bar(i, highs[i]-lows[i], bottom=lows[i], color='g', width=0.1, align='center')

# 绘制预测结果，用褪色柱形表示
closep = sim2self(closes[1:]/closes[:-1])
closep[-1-args.days] = closes[-1]
for i in range(args.days):
    closep[-args.days+i] *= closep[-args.days-1+i]
closep = closep[-args.days:]
openp = sim2self(opens[1:]/closes[:-1])[-args.days:] * closep
highp = sim2self(highs[1:]/closes[:-1])[-args.days:] * closep
lowp = sim2self(lows[1:]/closes[:-1])[-args.days:] * closep
for i in range(len(closep)):
    if closep[i] >= openp[i]:
        ax1.bar(i+len(closes), closep[i]-openp[i], bottom=openp[i], color='coral', width=0.618, align='center')
        ax1.bar(i+len(closes), highp[i]-lowp[i], bottom=lowp[i], color='coral', width=0.1, align='center')
    else:
        ax1.bar(i+len(closes), openp[i]-closep[i], bottom=closep[i], color='lime', width=0.618, align='center')
        ax1.bar(i+len(closes), highp[i]-lowp[i], bottom=lowp[i], color='lime', width=0.1, align='center')

# 显示图像&保存图像
plt.show()
plt.savefig('C:/Users/Wu/Downloads/week.jpeg')
