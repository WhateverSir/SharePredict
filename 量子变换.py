import math
import matplotlib.pyplot as plt
import numpy as np
import akshare as ts


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
# 通过股票代码获取股票数据,这里没有指定开始及结束日期
df = ts.stock_zh_a_daily("sz002371")
# 数据准备
dataset = df.close

# 将整型变为float
dataset = dataset.astype('float32')

# 获取最近100天的涨跌幅数据并计算收盘价
datatemp=np.zeros(100)
for i in range(100):
    datatemp[i] =(dataset[len(dataset)-100+i]-dataset[len(dataset)-101+i])/dataset[len(dataset)-101+i]
    
# 将收盘价按时间顺序排列，用于画蜡烛图
closes = dataset[-100:]
closes.reverse()

# 设置画布大小和标题
fig = plt.figure(figsize=(12,6))
fig.suptitle('Stock Candlestick Chart', fontsize=14)

# 添加蜡烛图子图
ax1 = fig.add_subplot(111)

# 设置x轴和y轴标签和范围
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.set_xlim(-1,len(closes))
ax1.set_ylim(min(closes)*0.9, max(closes)*1.1)

# 绘制蜡烛图，红涨绿跌
for i in range(len(closes)):
    if closes[i] > closes[i-1]:
        ax1.bar(i, closes[i]-closes[i-1], bottom=closes[i-1], color='red', width=0.5, align='center')
        ax1.bar(i, 0.01, bottom=closes[i], color='red', width=0.5, align='center')
    else:
        ax1.bar(i, closes[i]-closes[i-1], bottom=closes[i], color='green', width=0.5, align='center')
        ax1.bar(i, 0.01, bottom=closes[i-1], color='green', width=0.5, align='center')

# 绘制预测结果，用黄色柱形表示
predicted = guihua(datatemp)[-5:]
for i in range(len(predicted)):
    ax1.bar(i+len(closes), predicted[i]-closes[-1], bottom=closes[-1], color='yellow', width=0.5, align='center')

# 显示图像
plt.show()
