import math
import matplotlib.pyplot as plt
import numpy as np
import akshare as aks


# 函数定义
def f(lenth, period):
    temp=np.zeros(lenth) + 1
    if(period == 0):return temp
    for i in range(lenth):
        temp[i] = (1-2*((i/period)%2))*(2.0*(i%period)/period-1.0)
    return temp
def loss(array1, array2):
    tempsum=0.0
    for i in range(len(array1)):
        tempsum += (array1[i]-array2[i])*(array1[i]-array2[i])
    return tempsum
def getarray(coeff, lenth):
    temparray=np.zeros(lenth)
    for i in range(len(coeff)):
        temparray += coeff[i]*f(lenth, i)
    return temparray
def f1(lenth, period, flag=True):
    f=np.zeros(lenth)
    if(flag):
        for i in range(lenth):
            f[i] = math.sin(math.pi*i/period)
    else:
        for i in range(lenth):
            f[i] = math.cos(math.pi*i/period)
    return f
def fftchange(data, period, flag=True):
    a=0.0
    if(period <= 0):return sum(data)/len(data)
    if(flag):
        for i in range(len(data)):
            a += data[i]*math.sin(math.pi*i/period)
    else:
        for i in range(len(data)):
            a += data[i]*math.cos(math.pi*i/period)
    return a/len(data)
days = 5
def guihua(data):
    # 泰勒近似
    y = np.zeros(len(data)+days) + data.mean()
    k = 0.0
    data -= data.mean()
    for i in range(len(data)):
        k += i*data[i]
    k = k - len(data)*(len(data)-1)*data.mean()/2
    k = k /(len(data)*(len(data)-1)*((2*len(data)-1)/6-(len(data)-1)/4))
    y += k*range(len(data)+days)
    data -= k*range(len(data))
    # 傅里叶近似
    index = 1
    while(index<20):
        y += f1(len(data)+days, int(len(data)/(2*index)))*fftchange(data, int(len(data)/(2*index)))
        data -= f1(len(data), int(len(data)/(2*index)))*fftchange(data, int(len(data)/(2*index)))
        y += f1(len(data)+days, int(len(data)/(2*index)),flag=False)*fftchange(data, int(len(data)/(2*index)),flag=False)
        data -= f1(len(data), int(len(data)/(2*index)),flag=False)*fftchange(data, int(len(data)/(2*index)),flag=False)
        index+=1
    # 方波近似
    coefficient = np.zeros(50)
    step = 1.0
    while(step > 0.01):
        for i in range(len(coefficient)):
            result = loss(getarray(coefficient, len(data)), data)
            coefficient[i] += step
            newresult = loss(getarray(coefficient, len(data)), data)
            if(newresult < result):
                while(newresult < result):
                    result = newresult
                    coefficient[i] += step
                    newresult = loss(getarray(coefficient, len(data)), data)
                coefficient[i] -= step
            else:
                coefficient[i] -= 2*step
                newresult = loss(getarray(coefficient, len(data)), data)
                while(newresult < result):
                    result = newresult
                    coefficient[i] -= step
                    newresult = loss(getarray(coefficient, len(data)), data)
                coefficient[i] += step
        step *= 0.36788
    # while(index > 0):
    #     y += sum(f(len(datatemp), i)*datatemp)/len(datatemp)*f(len(data)+days, i)
    #     data -= sum(f(len(datatemp), i)*datatemp)/len(datatemp)*f(len(data), i)
    #     index -= 1
    return y + getarray(coefficient, len(data)+days)
stock = "sz002371"
# 通过股票代码获取股票数据,这里没有指定开始及结束日期
df = aks.stock_zh_a_daily(stock)
# 通过板块名称获取板块数据,这里没有指定开始及结束日期
#df = aks.stock_board_industry_index_ths(symbol="半导体及元件")
info = aks.stock_individual_info_em(symbol=stock[2:])

# 数据准备
closes = np.array(df.close.astype('float32')[-100:])
opens = np.array(df.open.astype('float32')[-100:])
highs = np.array(df.high.astype('float32')[-100:])
lows = np.array(df.low.astype('float32')[-100:])

# 设置画布大小和标题
fig = plt.figure(figsize=(12,6))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig.suptitle(info.loc[5,'value'] + '一周预测', fontsize=14)

# 添加蜡烛图子图
ax1 = fig.add_subplot(111)

# 设置x轴和y轴标签和范围
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.set_xlim(-1,len(closes)+days)
ax1.set_ylim(min(closes)*0.95, max(closes)*1.05)

# 绘制蜡烛图，红涨绿跌
for i in range(len(closes)):
    if closes[i] >= opens[i]:
        ax1.bar(i, closes[i]-opens[i], bottom=opens[i], color='r', width=0.618, align='center')
        ax1.bar(i, highs[i]-lows[i], bottom=lows[i], color='r', width=0.1, align='center')
    else:
        ax1.bar(i, opens[i]-closes[i], bottom=closes[i], color='g', width=0.618, align='center')
        ax1.bar(i, highs[i]-lows[i], bottom=lows[i], color='g', width=0.1, align='center')

# 绘制预测结果，用褪色柱形表示
closep = guihua(closes)[-5:]
openp = guihua(opens)[-5:]
highp = guihua(highs)[-5:]
lowp = guihua(lows)[-5:]
for i in range(len(closep)):
    if closep[i] >= openp[i]:
        ax1.bar(i+len(closes), closep[i]-openp[i], bottom=openp[i], color='coral', width=0.618, align='center')
        ax1.bar(i+len(closes), highp[i]-lowp[i], bottom=lowp[i], color='coral', width=0.1, align='center')
    else:
        ax1.bar(i+len(closes), openp[i]-closep[i], bottom=closep[i], color='lime', width=0.618, align='center')
        ax1.bar(i+len(closes), highp[i]-lowp[i], bottom=lowp[i], color='lime', width=0.1, align='center')

# 显示图像
plt.show()
