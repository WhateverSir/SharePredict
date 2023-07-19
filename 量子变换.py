import math
import matplotlib.pyplot as plt
import numpy as np
import tushare as ts


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
df = ts.get_k_data("300888")
# 数据准备
dataset = df.close
# 将整型变为float
dataset = dataset.astype('float32')
datatemp=np.zeros(100)
for i in range(100):
    datatemp[i] =(dataset[len(dataset)-100+i]-dataset[len(dataset)-101+i])/dataset[len(dataset)-101+i]

# 画图 
plt.figure(1)
plt.plot(datatemp)
predicted = guihua(datatemp)
plt.bar(range(len(predicted)), predicted, width=0.582, color='orange')#plot(predicted)#
plt.legend(["real","predict"], loc='upper left')
plt.show()