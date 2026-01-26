#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于Transformer的瞬时功率预测模型

Transformer使用自注意力机制，能够捕获序列中任意位置之间的依赖关系。

输入特征序列 (每个时刻7个特征):
- height: 高度 [m]
- VS: 竖直速度 [m/s]
- GS: 地速 [m/s]
- wind_speed: 风速 [m/s]
- temperature: 温度 [°C]
- humidity: 湿度 [%]
- wind_angle: 风向夹角 [度]

输出序列:
- 瞬时功率序列 [W]
"""

import pandas as pd
import numpy as np
import pickle
import os
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {device}")


# ==================== Transformer 模型定义 ====================

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer功率预测模型"""
    def __init__(self, input_size=7, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, max_len=500):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, src_key_padding_mask=None):
        # x: [batch, seq_len, input_size]
        
        # 输入嵌入
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 输出
        output = self.output_layer(x)
        
        return output.squeeze(-1)  # [batch, seq_len]


# ==================== 数据处理 ====================

class FlightDataset(Dataset):
    """飞行轨迹数据集"""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )


def collate_fn(batch):
    """批次数据整理函数"""
    sequences, targets = zip(*batch)
    
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    # 创建mask (True表示padding位置，需要被忽略)
    mask = torch.ones(len(sequences), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = False
    
    # 创建有效位置mask用于损失计算
    valid_mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        valid_mask[i, :length] = True
    
    return padded_sequences, padded_targets, mask, valid_mask, lengths


class DataProcessor:
    """数据处理器"""
    def __init__(self, min_seq_len=20, max_seq_len=500):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.feature_cols = ['Height', 'VS (m/s)', 'GS (m/s)', 'Wind Speed', 
                            'Temperature', 'Humidity', 'wind_angle']
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
    
    def load_data(self, data_dirs):
        """加载数据"""
        all_data = []
        for data_dir in data_dirs:
            trajectory_path = os.path.join(data_dir, "flightTrajectory.xlsx")
            if os.path.exists(trajectory_path):
                print(f"[INFO] 加载: {trajectory_path}")
                df = pd.read_excel(trajectory_path)
                n_orders = df['Order ID'].nunique()
                print(f"  - 数据量: {len(df)} 条记录, {n_orders} 个航次")
                all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"[INFO] 总数据量: {len(combined_df)} 条记录, {combined_df['Order ID'].nunique()} 个航次")
        
        return combined_df
    
    def preprocess(self, df):
        """数据预处理"""
        print("[INFO] 数据预处理...")
        
        # 计算功率 (W) = 电压(V) × 电流(A)
        df['Power'] = (df['Voltage'] / 1000.0) * (df['Current'] / 1000.0)
        
        # 计算风向夹角
        df['wind_angle'] = np.abs(df['Wind Direct'] - df['Course'])
        df['wind_angle'] = df['wind_angle'].apply(lambda x: x if x <= 180 else 360 - x)
        
        # 处理缺失值
        df = df.dropna(subset=self.feature_cols + ['Power'])
        
        # 过滤异常值
        df = df[df['Power'] > 0]
        df = df[df['Power'] < 15000]
        
        print(f"[INFO] 预处理后数据量: {len(df)} 条记录")
        
        return df
    
    def create_sequences(self, df, test_size=0.2):
        """按航次创建序列"""
        print("[INFO] 创建序列数据...")
        
        sequences = []
        targets = []
        order_ids = []
        
        grouped = df.groupby('Order ID')
        
        for order_id, group in grouped:
            group = group.sort_values('Time Stamp')
            seq_len = len(group)
            
            if seq_len < self.min_seq_len:
                continue
            
            if seq_len > self.max_seq_len:
                group = group.head(self.max_seq_len)
            
            features = group[self.feature_cols].values
            power = group['Power'].values
            
            sequences.append(features)
            targets.append(power)
            order_ids.append(order_id)
        
        print(f"[INFO] 有效序列数量: {len(sequences)}")
        print(f"[INFO] 序列长度范围: {min(len(s) for s in sequences)} - {max(len(s) for s in sequences)}")
        
        # 划分训练集和验证集
        n_samples = len(sequences)
        n_test = int(n_samples * test_size)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[n_test:]
        test_indices = indices[:n_test]
        
        train_sequences = [sequences[i] for i in train_indices]
        train_targets = [targets[i] for i in train_indices]
        test_sequences = [sequences[i] for i in test_indices]
        test_targets = [targets[i] for i in test_indices]
        
        print(f"[INFO] 训练集: {len(train_sequences)} 个航次")
        print(f"[INFO] 验证集: {len(test_sequences)} 个航次")
        
        return train_sequences, train_targets, test_sequences, test_targets
    
    def fit_scalers(self, sequences, targets):
        """拟合标准化器"""
        all_features = np.vstack(sequences)
        all_targets = np.concatenate(targets).reshape(-1, 1)
        
        self.feature_scaler.fit(all_features)
        self.target_scaler.fit(all_targets)
    
    def transform(self, sequences, targets):
        """标准化数据"""
        scaled_sequences = []
        scaled_targets = []
        
        for seq, tgt in zip(sequences, targets):
            scaled_seq = self.feature_scaler.transform(seq)
            scaled_tgt = self.target_scaler.transform(tgt.reshape(-1, 1)).flatten()
            scaled_sequences.append(scaled_seq)
            scaled_targets.append(scaled_tgt)
        
        return scaled_sequences, scaled_targets


# ==================== 训练器 ====================

class TransformerTrainer:
    """Transformer模型训练器"""
    def __init__(self, model, feature_scaler, target_scaler, device):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.0001, 
              patience=10, save_path='result/power_transformer_model.pth'):
        """训练模型"""
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_seq, batch_tgt, padding_mask, valid_mask, lengths in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_tgt = batch_tgt.to(self.device)
                padding_mask = padding_mask.to(self.device)
                valid_mask = valid_mask.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_seq, src_key_padding_mask=padding_mask)
                
                loss = criterion(outputs, batch_tgt)
                masked_loss = (loss * valid_mask).sum() / valid_mask.sum()
                
                masked_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += masked_loss.item() * valid_mask.sum().item()
                train_samples += valid_mask.sum().item()
            
            scheduler.step()
            avg_train_loss = train_loss / train_samples
            self.train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for batch_seq, batch_tgt, padding_mask, valid_mask, lengths in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_tgt = batch_tgt.to(self.device)
                    padding_mask = padding_mask.to(self.device)
                    valid_mask = valid_mask.to(self.device)
                    
                    outputs = self.model(batch_seq, src_key_padding_mask=padding_mask)
                    loss = criterion(outputs, batch_tgt)
                    masked_loss = (loss * valid_mask).sum()
                    
                    val_loss += masked_loss.item()
                    val_samples += valid_mask.sum().item()
            
            avg_val_loss = val_loss / val_samples
            self.val_losses.append(avg_val_loss)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model(save_path)
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f} | Best: {best_val_loss:.6f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"\n[INFO] 早停：在第 {epoch+1} 轮停止训练")
                break
        
        # 加载最佳模型
        self.load_model(save_path)
        
        return best_val_loss
    
    def predict(self, data_loader):
        """预测"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_seq, batch_tgt, padding_mask, valid_mask, lengths in data_loader:
                batch_seq = batch_seq.to(self.device)
                padding_mask = padding_mask.to(self.device)
                outputs = self.model(batch_seq, src_key_padding_mask=padding_mask)
                
                for i, length in enumerate(lengths):
                    pred = outputs[i, :length].cpu().numpy()
                    true = batch_tgt[i, :length].numpy()
                    
                    pred = self.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                    true = self.target_scaler.inverse_transform(true.reshape(-1, 1)).flatten()
                    
                    all_predictions.append(pred)
                    all_targets.append(true)
        
        return all_predictions, all_targets
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)
        
        scaler_path = path.replace('.pth', '_scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }, f)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        scaler_path = path.replace('.pth', '_scalers.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.feature_scaler = scalers['feature_scaler']
                self.target_scaler = scalers['target_scaler']


def calculate_metrics(predictions, targets):
    """计算评估指标"""
    all_pred = np.concatenate(predictions)
    all_true = np.concatenate(targets)
    
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    mae = mean_absolute_error(all_true, all_pred)
    r2 = r2_score(all_true, all_pred)
    
    mask = all_true != 0
    mape = np.mean(np.abs((all_true[mask] - all_pred[mask]) / all_true[mask])) * 100
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


def main():
    """主函数"""
    print("="*60)
    print("Transformer 瞬时功率预测模型训练")
    print("="*60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ===== 1. 数据加载与预处理 =====
    data_dirs = [
        "Drone_energy_dataset/UAS04028624",
        "Drone_energy_dataset/UAS04028648"
    ]
    
    processor = DataProcessor(min_seq_len=20, max_seq_len=500)
    df = processor.load_data(data_dirs)
    df = processor.preprocess(df)
    
    train_sequences, train_targets, test_sequences, test_targets = processor.create_sequences(df)
    
    processor.fit_scalers(train_sequences, train_targets)
    train_sequences_scaled, train_targets_scaled = processor.transform(train_sequences, train_targets)
    test_sequences_scaled, test_targets_scaled = processor.transform(test_sequences, test_targets)
    
    train_dataset = FlightDataset(train_sequences_scaled, train_targets_scaled)
    test_dataset = FlightDataset(test_sequences_scaled, test_targets_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # ===== 2. 创建模型 =====
    # 统一参数配置（与LSTM/GRU/Bi-LSTM保持一致以便公平对比）
    # Transformer的d_model=256对应RNN的hidden_size=256
    # num_layers=3对应RNN的num_layers=3
    model = TransformerModel(
        input_size=7,
        d_model=256,         # 模型维度（统一为256，对应RNN的hidden_size）
        nhead=8,             # 注意力头数
        num_layers=3,        # 编码器层数（统一为3）
        dim_feedforward=1024,# 前馈网络维度（4倍d_model）
        dropout=0.2,         # Dropout比例（统一为0.2）
        max_len=500
    ).to(device)
    
    print(f"\n[INFO] 模型结构:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 总参数量: {total_params:,}")
    
    # ===== 3. 训练模型 =====
    trainer = TransformerTrainer(model, processor.feature_scaler, processor.target_scaler, device)
    
    print("\n" + "="*60)
    print("开始训练 Transformer 模型")
    print("="*60)
    
    # 统一训练参数
    best_loss = trainer.train(
        train_loader, test_loader,
        epochs=100,          # 统一为100轮
        lr=0.001,            # 统一学习率
        patience=20,         # 统一早停耐心值
        save_path='result/power_transformer_model.pth'
    )
    
    print(f"\n[INFO] 训练完成，最佳验证损失: {best_loss:.6f}")
    
    # ===== 4. 评估模型 =====
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    predictions, targets = trainer.predict(test_loader)
    metrics = calculate_metrics(predictions, targets)
    
    print(f"\n测试集评估指标:")
    print(f"  RMSE: {metrics['RMSE']:.4f} W")
    print(f"  MAE:  {metrics['MAE']:.4f} W")
    print(f"  R2:   {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # 保存评估结果
    results_df = pd.DataFrame([metrics])
    results_df.to_csv('result/transformer_evaluation.csv', index=False)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"模型文件: result/power_transformer_model.pth")


if __name__ == "__main__":
    main()
