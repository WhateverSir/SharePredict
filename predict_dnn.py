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
# 数据预处理  
stock_list = ['601127', '002436', '600363', '601816', '600446', '002410', '000338', '600588', '603195', '603290', '603995', '300598', '002371', '688111', '000977', '300450', '688006', '688088', '688002', '688005', '688029', '603658', '002603', '688068', '000963', '688012', '688008', '600741', '002959', '688036', '300630', '603160', '300433', '601601', '600660', '300616', '601012', '601668', '600030', '600009', '601318', '600104', '000423', '601939', '601398', '000002', '002236', '002003', '002304', '601633', '300003', '002615', '000333', '000063', '601888', '600519', '600563', '600518', '002507', '002352', '002468', '300033', '300059', '300750', '600066', '600299', '600305', '600567', '600585', '601138', '002597', '600352', '600309', '603986', '603605', '603515', '000860', '002032', '002027', '000858', '000651', '000568', '603027', '603579', '603806', '603288', '002594', '600298', '600332', '002466', '002230', '000895', '600570', '600887', '002661', '002244', '002275', '300673', '000538', '603899', '603517', '002120', '002415', '002294', '600315', '603868', '603156']
class MyDataset(Dataset):  
    def __init__(self, stock_list):
        self.x, self.y, name_list= [], [] ,[] 
        for stock in stock_list:
            info = aks.stock_individual_info_em(symbol=stock)
            name_list.append(info.loc[5,'value'])    
            df = aks.stock_zh_a_hist(symbol=stock, period='daily', start_date='20231101') 
            change = np.array(df['涨跌幅'].astype('float32')) / 10  #标准化
            exchange = np.array(df['成交量'].astype('float32')) / info.loc[7,'value'] #标准化
            for i in range(len(change) - args.seq_length - 1):
                combined_input = np.stack([change[i:i+args.seq_length], exchange[i:i+args.seq_length]], axis=0)
                self.x.append(combined_input)  
                self.y.append(change[i+args.seq_length]) 
        print("Data loaded, length is ", len(self.x))  
    def __len__(self):  
        return len(self.x)  
  
    def __getitem__(self, idx):  
        return self.x[idx], self.y[idx]
  
# 定义DNN模型  
class DNN(nn.Module):  
    def __init__(self, input_size=args.seq_length, hidden_layer_size=64, num_layers=2):  
        super(DNN, self).__init__() 
        layers = []  
        layers.append(nn.Linear(input_size, hidden_layer_size))  # 因为x的形状是(batch, 2, 100)
        for i in range(num_layers-1):  
            layers.append(nn.ReLU())  # 添加激活函数，比如ReLU  
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))  
        layers.append(nn.ReLU())  # 最后一层之前也需要激活函数  
        layers.append(nn.Linear(hidden_layer_size, 1, bias=False))  # 输出层，对应y的形状(batch, 1)  
        self.network = nn.Sequential(*layers)  # 使用nn.Sequential包装所有层  
  
    def forward(self, x):  
        y = self.network(x)
        y = torch.prod(y, dim=1, keepdim=True)
        return torch.tanh(y.squeeze(1)) 
     
# 实例化 Dataset  
dataset = MyDataset(stock_list)  
  
# 实例化 DataLoader  
batch_size = 32  # 您可以根据需要调整这个值  
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)    
# 实例化模型、损失函数和优化器  
model = DNN() 
if os.path.exists(args.train_model): 
    model.load_state_dict(torch.load(args.train_model),strict=False)
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
model.train()
# 训练循环（简单示例）  
num_epochs, best_loss = 100, 100  
for epoch in range(num_epochs):  
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src, tgt = batch
        src, tgt = src.to(torch.float32), tgt.to(torch.float32)
        optimizer.zero_grad()
        output = model(src)
        # n = output.shape[-1]
        loss = criterion(output, tgt.unsqueeze(1))#bce_loss_w(output, tgt)#
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()  
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), args.train_model)
    # 可以在这里添加验证步骤和打印损失等  
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')
# 加载模型，预测
model.eval()
info = aks.stock_individual_info_em(symbol=args.stock) 
name = info.loc[5,'value']
df = aks.stock_zh_a_hist(symbol=args.stock, period='daily', start_date='20231101') 
change = np.array(df['涨跌幅'].astype('float32')) / 10  #标准化
exchange = np.array(df['成交量'].astype('float32')) / info.loc[7,'value'] #标准化
test_input = np.stack([change[-args.seq_length:], exchange[-args.seq_length:]], axis=0)
test_output = model(torch.tensor(test_input).unsqueeze(0)).squeeze()
print(f'Stock {name}[code:{args.stock}] tomorrow price change: {test_output.detach().numpy()*10}%')
