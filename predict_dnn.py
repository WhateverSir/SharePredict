import numpy as np  
import akshare as aks
import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader  
import argparse  
import os  
# 定义命令行参数  
parser = argparse.ArgumentParser(description='Predict stock data based DNN.')  
parser.add_argument('--stock', type=str, default='601127', help='Stock code to predict.')  
parser.add_argument('--seq_length', type=int, default=100, help='The number of days to predict.')  
parser.add_argument('--train_model', type=str, default="D:/dataSet/models/DNN_share.pt", help='the model file to load')  
  
# 解析命令行参数  
args = parser.parse_args()

# 数据预处理  

to do
