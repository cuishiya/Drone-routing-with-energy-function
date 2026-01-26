#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于GRU的Seq2Seq瞬时功率预测模型

GRU (Gated Recurrent Unit) 相比LSTM结构更简单，参数更少，训练更快。

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


# ==================== GRU Seq2Seq 模型定义 ====================

class GRUEncoder(nn.Module):
    """GRU编码器"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
    
    def forward(self, x):
        outputs, hidden = self.gru(x)
        return outputs, hidden


class GRUDecoder(nn.Module):
    """GRU解码器"""
    def __init__(self, hidden_size, output_size=1, num_layers=2, dropout=0.2, bidirectional=True):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        decoder_input_size = hidden_size * self.num_directions
        
        self.gru = nn.GRU(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, encoder_outputs, hidden):
        decoder_outputs, _ = self.gru(encoder_outputs, hidden)
        outputs = self.fc(decoder_outputs)
        return outputs.squeeze(-1)


class GRUSeq2Seq(nn.Module):
    """GRU Seq2Seq 完整模型"""
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, 
                 dropout=0.2, bidirectional=True):
        super(GRUSeq2Seq, self).__init__()
        
        self.encoder = GRUEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.decoder = GRUDecoder(
            hidden_size=hidden_size,
            output_size=1,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    
    def forward(self, x):
        encoder_outputs, hidden = self.encoder(x)
        outputs = self.decoder(encoder_outputs, hidden)
        return outputs


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
    
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return padded_sequences, padded_targets, mask, lengths


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

class GRUTrainer:
    """GRU模型训练器"""
    def __init__(self, model, feature_scaler, target_scaler, device):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, 
              patience=10, save_path='result/power_gru_model.pth'):
        """训练模型"""
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=5, verbose=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_seq, batch_tgt, mask, lengths in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_tgt = batch_tgt.to(self.device)
                mask = mask.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_seq)
                
                loss = criterion(outputs, batch_tgt)
                masked_loss = (loss * mask).sum() / mask.sum()
                
                masked_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += masked_loss.item() * mask.sum().item()
                train_samples += mask.sum().item()
            
            avg_train_loss = train_loss / train_samples
            self.train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for batch_seq, batch_tgt, mask, lengths in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_tgt = batch_tgt.to(self.device)
                    mask = mask.to(self.device)
                    
                    outputs = self.model(batch_seq)
                    loss = criterion(outputs, batch_tgt)
                    masked_loss = (loss * mask).sum()
                    
                    val_loss += masked_loss.item()
                    val_samples += mask.sum().item()
            
            avg_val_loss = val_loss / val_samples
            self.val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
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
            for batch_seq, batch_tgt, mask, lengths in data_loader:
                batch_seq = batch_seq.to(self.device)
                outputs = self.model(batch_seq)
                
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
    print("GRU Seq2Seq 瞬时功率预测模型训练")
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
    # 统一参数配置（与LSTM/Bi-LSTM/Transformer保持一致以便公平对比）
    model = GRUSeq2Seq(
        input_size=7,
        hidden_size=256,     # 隐藏层大小（统一为256）
        num_layers=3,        # GRU层数（统一为3）
        dropout=0.2,         # Dropout比例（统一为0.2）
        bidirectional=True
    ).to(device)
    
    print(f"\n[INFO] 模型结构:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 总参数量: {total_params:,}")
    
    # ===== 3. 训练模型 =====
    trainer = GRUTrainer(model, processor.feature_scaler, processor.target_scaler, device)
    
    print("\n" + "="*60)
    print("开始训练 GRU Seq2Seq 模型")
    print("="*60)
    
    # 统一训练参数
    best_loss = trainer.train(
        train_loader, test_loader,
        epochs=100,          # 统一为100轮
        lr=0.001,            # 统一学习率
        patience=20,         # 统一早停耐心值
        save_path='result/power_gru_model.pth'
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
    results_df.to_csv('result/gru_evaluation.csv', index=False)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"模型文件: result/power_gru_model.pth")


if __name__ == "__main__":
    main()
