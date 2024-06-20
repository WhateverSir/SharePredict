import numpy as np  
import akshare as aks
import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader  
import argparse  
# 定义命令行参数  
parser = argparse.ArgumentParser(description='Predict stock data based DNN.')  
parser.add_argument('--stock', type=str, default='601127', help='Stock code to predict.')  
parser.add_argument('--seq_length', type=int, default=100, help='The number of days to predict.')  
  
# 解析命令行参数  
args = parser.parse_args()

to do
