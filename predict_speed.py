import matplotlib.pyplot as plt  
import numpy as np  
import akshare as aks  
import argparse  
import pandas as pd  
# 初始化命令行参数解析器  
parser = argparse.ArgumentParser(description='基于速度和加速度的股票交易模拟')  
parser.add_argument('--stock', type=str, default='601127', required=True, help='请输入股票代码')  
args = parser.parse_args()
  
# 通过股票代码获取股票数据  
df = aks.stock_zh_a_hist(symbol=args.stock, period='daily', start_date='20231001', adjust="qfq")  
# 注意：akshare的API可能会更新，所以请确保使用的函数和参数是最新的  
# 如果akshare没有'adjust'参数或者'qfq'（前复权）不是有效的选项，请根据需要调整  
  
# 数据预处理  
closes = np.array(df['收盘'].dropna().astype('float32'))  
dates = pd.to_datetime(df.index.dropna())  
  
# 计算速度和加速度（这里我们采用简单的差分方法）  
speeds = np.diff(closes)  # 日收益率，即速度  
accelerations = np.diff(speeds)   # 加速度的简单计算（这里假设时间间隔为1天）  
  
# 为了在速度和加速度数组中使用交易信号，我们需要在它们前面添加一个NaN值（或进行其他处理）  
speeds = np.insert(speeds, 0, np.nan)  
accelerations = np.insert(accelerations, 0, np.nan)  
accelerations = np.insert(accelerations, 0, np.nan)  
  
  
# 初始化变量以记录交易和收益  
positions = []  # 记录持仓状态（买入、卖出或未持仓）  
cash = 100000  # 初始资金  
shares = 0  # 初始持股数量  
total_cost = 0  # 总成本  
total_profit = 0  # 总收益  
  
# 模拟交易过程  
for i in range(1, len(closes)):  
    if speeds[i-1]<0 and speeds[i]>0 and cash >= closes[i]:  
        # 执行买入操作  
        shares = cash // closes[i]  # 用全部现金买入股票  
        cash -= shares * closes[i]  
        total_cost += shares * closes[i]  
        positions.append(('buy', i, shares, closes[i]))  
    elif speeds[i-1]>0 and speeds[i]<0 and shares > 0:  
        # 执行卖出操作  
        cash += shares * closes[i] 
        total_profit += shares * (closes[i] - positions[-1][3])  # 计算卖出时的收益  
        positions.append(('sell', i, shares, closes[i]))  
        shares = 0 
  
# 计算最终收益和收益率  
final_value = cash + (shares * closes[-1] if shares > 0 else 0)  
total_return = final_value - 100000  # 总收益（包括现金和剩余股票的价值）  
return_rate = total_return / 100000 * 100  # 收益率  
  
# 打印交易信息、收益和收益率  
print("交易记录:")  
for pos in positions:  
    print(f"{pos[0]}于第{pos[1]}天以{pos[3]:.2f}元的价格进行了{pos[2]}股的交易")  
print(f"\n总收益: {total_profit:.2f}元")  
print(f"最终资产价值: {final_value:.2f}元")  
print(f"收益率: {return_rate:.2f}%")  
plt.plot(closes)
# 注意：此代码为模拟交易，并未实际执行买卖操作。  
# 在实际应用中，请确保您的交易策略经过充分回测，并考虑交易成本、滑点等因素。
