

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import mode
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class TimeSeriesDataset(Dataset):
    def __init__(self, label_dir, seq_len):
        self.data = []
        self.labels = []
        self.seq_len = seq_len
        # 读取数据
        for label in os.listdir(label_dir):
            label_path = os.path.join(label_dir, label)
            if not label_path.endswith('.DS_Store'):
                label_value = 1 if label == '1' else 0  # 标签：1文件夹为正样本，0文件夹为负样本
                for file in os.listdir(label_path):
                    if file.endswith('.xlsx'):
                        file_path = os.path.join(label_path, file)
                        df = pd.read_excel(file_path, header=None)
                        
                        # 取第二行和第三行作为特征值：Y1和Y2
                        Y1 = df.iloc[1, 1:seq_len].values.astype('float32')  # Y1：第二行
                        Y2 = df.iloc[2, 1:seq_len].values.astype('float32')  # Y2：第三行
                        
                        # 每个时间步有两个特征（Y1, Y2）
                        X = np.vstack((Y1, Y2)).T  # 转置后，每行表示一个时间点的特征
                        
                        self.data.append(X)
                        self.labels.append(label_value)  # 0为负样本，1为正样本
          
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取样本数据，确保返回的格式为 [seq_len, input_size]
        X = self.data[idx]
        y = self.labels[idx]
        
        # 转换为torch.tensor，并确保X的形状是 [seq_len, input_size]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 只使用最后一个时间步的输出
        return F.softmax(output, dim=1)  # 添加softmax层，dim=1表示按行处理




# 指定目标文件夹路径
directory = "LSTM model data split"

# 获取所有文件夹路径
all_path = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

for my_path in all_path: 
    train_label_dir = my_path+'/train'       
    test_data_dir = my_path+'/test'
    test_labels = []  # 用于存储测试集的标签
    # 获取当前系统时间
    current_time = datetime.now()
    print(f'Satrt training{train_label_dir}.....................{current_time}', )

    for loop in range(0,1):#多次loop取均值
        # seq_len5-20超参数
        input_size = 2  # 每个时间步的特征数：Y1和Y2
        hidden_size = 16  # LSTM隐层大小
        output_size = 2  # 二分类问题：0和1
        batch_size =8
        epochs = 200
        learning_rate = 0.001

        # 初始化metrics DataFrame
        metrics_df = pd.DataFrame(columns=["seq_len", "accuracy", "precision", "recall"])

        # 设置使用的时间数据长度seq_len
        for seq_len in range(5, 340, 10):
            if(seq_len >= 20):
                #seq_len20-30超参数
                hidden_size = 16  # LSTM隐层大小
                batch_size =8
                learning_rate = 0.003
            if(seq_len >= 30):
                #seq_len30-50超参数
                hidden_size = 16  # LSTM隐层大小
                batch_size =16
                learning_rate = 0.003
            if(seq_len >= 50):
                #seq_len50+超参数
                hidden_size = 32  # LSTM隐层大小
                batch_size =32
                learning_rate = 0.005
            all_accuracy = []
            all_precision = []
            all_recall = []
            export_prediction_list = []  #存放每个测试文件在不同专家模型的预测list

            for run in range(5):  # 对于每个seq_len运行5个模型
                print(f"Model {run+1} Running for seq_len={seq_len} in hid{hidden_size} bat{batch_size} learn{learning_rate}")

                # 数据加载
                train_dataset = TimeSeriesDataset(train_label_dir, seq_len)

                # 数据拆分：80%用于训练，20%用于验证
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_data, val_data = random_split(train_dataset, [train_size, val_size])

                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

                # 初始化LSTM模型
                model = LSTMClassifier(input_size, hidden_size, output_size)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                # 训练和验证
                train_losses = []
                val_losses = []

                for epoch in range(epochs):
                    model.train()
                    epoch_loss = 0
                    for X, y in train_loader:
                        X = X  # 增加一个维度，作为输入特征 [batch_size, seq_len, input_size]
                        y = y

                        optimizer.zero_grad()
                        outputs = model(X)
                        loss = criterion(outputs, y)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                    avg_loss = epoch_loss / len(train_loader)
                    train_losses.append(avg_loss)

                    # 验证集损失
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for X, y in val_loader:
                            X = X  # 增加特征维度
                            y = y
                            outputs = model(X)
                            loss = criterion(outputs, y)
                            val_loss += loss.item()

                    avg_val_loss = val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)

                    #print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    #             # 绘制训练和验证损失曲线
    #             plt.plot(train_losses, label='Train Loss')
    #             plt.plot(val_losses, label='Val Loss')
    #             plt.xlabel('Epoch')
    #             plt.ylabel('Loss')
    #             plt.title(f'Training and Validation Loss Curve (seq_len={seq_len})')
    #             plt.legend()
    #             plt.savefig(f'fig/loss_curve_seq_len_{seq_len}_run_{run+1}_batch{batch_size}.png')  # 保存图片到fig文件夹
    #             plt.close()

                test_labels = []  # 用于存储测试集的标签
                test_predictions = []  # 用于存储模型的预测结果

                for file in os.listdir(test_data_dir):
                    if file.endswith('.xlsx'):
                        file_path = os.path.join(test_data_dir, file)
                        df = pd.read_excel(file_path, header=None)


                        # X是每个时序的两个特征：Y1和Y2
                        Y1 = df.iloc[1, 1:seq_len].values.astype('float32')
                        Y2 = df.iloc[2, 1:seq_len].values.astype('float32')
                        X = np.vstack((Y1, Y2)).T  # 转置后，每行表示一个时间点的特征

                        label = int(df.iloc[3, 1])  # 获取真值：第四行第二列的值

                        test_labels.append(label)

                        # 模型预测
                        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # 增加batch维度
                        with torch.no_grad():
                            output = model(X_tensor)
                            prediction = torch.argmax(output, dim=1).item()  # 选择最大概率的类别
                            test_predictions.append(prediction)
                            if(prediction!=label):print(f'Model {run+1} wrong predict',file_path)

                #本次模型结果加入模型专家库
                export_prediction_list.append(test_predictions)    
            #专家模型投票取众数
            #print(export_prediction_list)
            final_prediction = np.array(export_prediction_list)
            final_prediction = mode(final_prediction, axis=0).mode.flatten()
            final_prediction = list(final_prediction)
            print('Experts voting results',final_prediction)
            if(final_prediction!=test_labels):print('Incorrect with labels:',test_labels)


            # 计算准确率、精确率和召回率
            accuracy = accuracy_score(test_labels, final_prediction)
            precision = precision_score(test_labels, final_prediction, average='binary')  # 或者使用 'weighted'
            recall = recall_score(test_labels, final_prediction, average='binary')  # 或者使用 'weighted'


            # 将平均结果添加到metrics_df
            metrics = pd.DataFrame({
                "seq_len": [seq_len],
                'accuracy': [accuracy],
                'precision': [precision],
                'recall': [recall]
            })
            print(metrics)

            metrics_df = pd.concat([metrics_df, metrics], ignore_index=True)

        # 保存metrics到CSV
        metrics_df.to_csv(f'metrics of data {train_label_dir[-7]} with loop-{loop}.csv', index=False)



