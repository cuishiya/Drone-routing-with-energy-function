#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM-Transformer混合模型瞬时功率预测

结合LSTM和Transformer的优势：
- LSTM层：提取时间特征，捕捉短期依赖关系
- Transformer层：通过自注意力机制捕捉长距离依赖关系

输入特征序列 (每个时刻8个特征):
- height: 高度 [m]
- VS: 竖直速度 [m/s]
- GS: 地速 [m/s]
- wind_speed: 风速 [m/s]
- temperature: 温度 [°C]
- humidity: 湿度 [%]
- wind_angle: 风向夹角 [度]
- payload: 载荷 [kg]

输出序列:
- 瞬时功率序列 [W]

模型架构:
1. 输入映射层: Linear(8 -> d_model)
2. LSTM层: 提取短期时序特征
3. Transformer编码器: 捕捉长距离依赖
4. 输出层: Linear(d_model -> 1)
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
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# 切换工作目录到项目根目录，确保相对路径正确
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {device}")


# ==================== LSTM-Transformer 混合模型定义 ====================

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMTransformerModel(nn.Module):
    """
    LSTM-Transformer串行模型
    
    架构: 输入 → LSTM → Transformer → 输出
    """
    def __init__(self, input_size=8, d_model=256, lstm_layers=2,
                 nhead=8, num_transformer_layers=3, dim_feedforward=512, 
                 dropout=0.1, max_len=500):
        super(LSTMTransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 1. 输入映射层
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # 2. 双向LSTM（hidden * 2 = d_model，无需额外映射）
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,  # 双向后刚好等于d_model
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # 3. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # 4. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # 5. 输出层（与单纯Transformer一致）
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, src_key_padding_mask=None):
        """
        前向传播: 输入 → LSTM → Transformer → 输出
        """
        # 1. 输入映射
        x = self.input_embedding(x)  # (batch, seq_len, d_model)
        
        # 2. LSTM提取时序特征
        x, _ = self.lstm(x)  # (batch, seq_len, d_model)  双向: hidden*2 = d_model
        
        # 3. 缩放 + 位置编码
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # 4. Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 5. 输出
        output = self.output_layer(x)
        
        return output.squeeze(-1)  # (batch, seq_len)


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
    """批次数据整理函数（支持变长序列）"""
    sequences, targets = zip(*batch)
    
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    
    # 填充序列
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    # 创建掩码（True表示填充位置，用于Transformer）
    padding_mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_mask[i, length:] = True
    
    return padded_sequences, padded_targets, padding_mask, lengths


class DataProcessor:
    """数据处理器"""
    def __init__(self, min_seq_len=20, max_seq_len=500):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        # 8个特征（包含载荷）
        self.feature_cols = ['Height', 'VS (m/s)', 'GS (m/s)', 'Wind Speed', 
                            'Temperature', 'Humidity', 'wind_angle', 'payload']
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
    
    def load_data(self, data_dirs):
        """加载数据（包含载荷信息）"""
        all_data = []
        for data_dir in data_dirs:
            trajectory_path = os.path.join(data_dir, "flightTrajectory.xlsx")
            record_path = os.path.join(data_dir, "flightRecord.xlsx")
            
            if os.path.exists(trajectory_path):
                print(f"[INFO] 加载轨迹数据: {trajectory_path}")
                df = pd.read_excel(trajectory_path)
                n_orders = df['Order ID'].nunique()
                print(f"  - 数据量: {len(df)} 条记录, {n_orders} 个航次")
                
                # 加载飞行记录以获取载荷信息
                if os.path.exists(record_path):
                    print(f"[INFO] 加载飞行记录: {record_path}")
                    record_df = pd.read_excel(record_path)
                    
                    if 'Payload (kg)' in record_df.columns:
                        payload_map = record_df.set_index('Order ID')['Payload (kg)'].to_dict()
                        df['payload'] = df['Order ID'].map(payload_map)
                        print(f"  - 成功关联载荷信息，有效载荷数据: {df['payload'].notna().sum()} 条")
                    else:
                        print(f"  - 警告: 飞行记录中没有Payload (kg)列")
                        df['payload'] = 0.0
                else:
                    print(f"  - 警告: 未找到飞行记录文件，载荷设为0")
                    df['payload'] = 0.0
                
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
        
        # 处理载荷缺失值（用0填充，表示空载）
        if 'payload' in df.columns:
            payload_missing = df['payload'].isna().sum()
            if payload_missing > 0:
                print(f"[INFO] 载荷缺失值: {payload_missing} 条，用0填充")
                df['payload'] = df['payload'].fillna(0.0)
        
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
        
        print(f"[INFO] 创建了 {len(sequences)} 个序列")
        
        # 划分训练集和测试集
        indices = list(range(len(sequences)))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        train_sequences = [sequences[i] for i in train_idx]
        train_targets = [targets[i] for i in train_idx]
        test_sequences = [sequences[i] for i in test_idx]
        test_targets = [targets[i] for i in test_idx]
        
        print(f"[INFO] 训练集: {len(train_sequences)} 个序列, 测试集: {len(test_sequences)} 个序列")
        
        return train_sequences, train_targets, test_sequences, test_targets
    
    def fit_scalers(self, sequences, targets):
        """拟合标准化器"""
        all_features = np.vstack(sequences)
        all_targets = np.concatenate(targets).reshape(-1, 1)
        
        self.feature_scaler.fit(all_features)
        self.target_scaler.fit(all_targets)
        
        print(f"[INFO] 标准化器已拟合")
    
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

class LSTMTransformerTrainer:
    """LSTM-Transformer模型训练器"""
    
    def __init__(self, model, feature_scaler, target_scaler, device):
        self.model = model.to(device)
        self.device = device
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, 
              patience=15, save_path='result/power_lstm_transformer_model.pth'):
        """训练模型"""
        
        # 使用ADAM优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = nn.MSELoss(reduction='none')
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n[INFO] 开始训练，共 {epochs} 轮")
        print(f"[INFO] 优化器: Adam, 学习率: {lr}")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_seq, batch_tgt, padding_mask, lengths in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_tgt = batch_tgt.to(self.device)
                padding_mask = padding_mask.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(batch_seq, src_key_padding_mask=padding_mask)
                
                # 计算损失（只计算非填充位置）
                loss_matrix = criterion(outputs, batch_tgt)
                valid_mask = ~padding_mask
                loss = (loss_matrix * valid_mask).sum() / valid_mask.sum()
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * valid_mask.sum().item()
                train_samples += valid_mask.sum().item()
            
            avg_train_loss = train_loss / train_samples
            self.train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for batch_seq, batch_tgt, padding_mask, lengths in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_tgt = batch_tgt.to(self.device)
                    padding_mask = padding_mask.to(self.device)
                    
                    outputs = self.model(batch_seq, src_key_padding_mask=padding_mask)
                    
                    loss_matrix = criterion(outputs, batch_tgt)
                    valid_mask = ~padding_mask
                    val_loss += (loss_matrix * valid_mask).sum().item()
                    val_samples += valid_mask.sum().item()
            
            avg_val_loss = val_loss / val_samples
            self.val_losses.append(avg_val_loss)
            
            # 学习率调整
            scheduler.step(avg_val_loss)
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # 保存最佳模型
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, save_path)
                
                # 保存标准化器
                scaler_path = save_path.replace('.pth', '_scalers.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump({
                        'feature_scaler': self.feature_scaler,
                        'target_scaler': self.target_scaler
                    }, f)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[INFO] 早停触发，在第 {epoch+1} 轮停止训练")
                    break
        
        print(f"\n[INFO] 训练完成，最佳验证损失: {best_val_loss:.6f}")
        
        # 绘制训练曲线
        self.plot_training_history()
        
        return best_val_loss
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='训练损失', color='blue')
        plt.plot(self.val_losses, label='验证损失', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('LSTM-Transformer 训练历史')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('result/lstm_transformer_training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[INFO] 训练历史图已保存: result/lstm_transformer_training_history.png")
    
    def predict(self, data_loader):
        """预测"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_seq, batch_tgt, padding_mask, lengths in data_loader:
                batch_seq = batch_seq.to(self.device)
                padding_mask = padding_mask.to(self.device)
                
                outputs = self.model(batch_seq, src_key_padding_mask=padding_mask)
                
                # 提取有效预测（去除填充）
                for i, length in enumerate(lengths):
                    pred = outputs[i, :length].cpu().numpy()
                    tgt = batch_tgt[i, :length].numpy()
                    
                    # 反标准化
                    pred = self.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                    tgt = self.target_scaler.inverse_transform(tgt.reshape(-1, 1)).flatten()
                    
                    all_predictions.extend(pred)
                    all_targets.extend(tgt)
        
        return np.array(all_predictions), np.array(all_targets)


# ==================== 评估指标 ====================

def calculate_metrics(predictions, targets):
    """计算评估指标"""
    # 确保为正值
    predictions = np.maximum(predictions, 0)
    
    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # MAPE (避免除零)
    mask = targets > 0
    mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def plot_predictions(predictions, targets, save_path='result/lstm_transformer_prediction_scatter.png'):
    """绘制预测散点图"""
    plt.figure(figsize=(8, 8))
    
    plt.scatter(targets, predictions, alpha=0.3, s=1, color='red')
    
    # 对角线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2)
    
    # 计算指标
    metrics = calculate_metrics(predictions, targets)
    
    plt.xlabel('实际功率 (W)')
    plt.ylabel('预测功率 (W)')
    plt.title('LSTM-Transformer 预测结果')
    
    # 添加指标文本
    text = f"$R^2$ = {metrics['R2']:.4f}\nRMSE = {metrics['RMSE']:.1f} W\nMAE = {metrics['MAE']:.1f} W"
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 预测散点图已保存: {save_path}")


# ==================== 主函数 ====================

def main():
    print("="*60)
    print("LSTM-Transformer 混合模型 - 无人机功率预测")
    print("="*60)
    
    # ===== 1. 数据准备 =====
    data_dirs = [
        "Drone_energy_dataset/UAS04028624"
    ]
    
    processor = DataProcessor(min_seq_len=20, max_seq_len=500)
    
    # 加载和预处理数据
    df = processor.load_data(data_dirs)
    df = processor.preprocess(df)
    
    # 创建序列
    train_sequences, train_targets, test_sequences, test_targets = processor.create_sequences(df)
    
    # 拟合标准化器并转换数据
    processor.fit_scalers(train_sequences, train_targets)
    train_sequences_scaled, train_targets_scaled = processor.transform(train_sequences, train_targets)
    test_sequences_scaled, test_targets_scaled = processor.transform(test_sequences, test_targets)
    
    # 创建数据加载器
    train_dataset = FlightDataset(train_sequences_scaled, train_targets_scaled)
    test_dataset = FlightDataset(test_sequences_scaled, test_targets_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # ===== 2. 创建模型 =====
    # LSTM-Transformer串行模型
    model = LSTMTransformerModel(
        input_size=8,                    # 8个输入特征
        d_model=256,                     # 模型维度（LSTM双向输出也是256）
        lstm_layers=2,                   # LSTM层数
        nhead=8,                         # 多头注意力头数
        num_transformer_layers=3,        # Transformer编码器层数
        dim_feedforward=512,             # 前馈网络维度
        dropout=0.1,                     # 丢弃率
        max_len=500
    )
    
    print(f"\n[INFO] 模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 总参数量: {total_params:,}")
    
    # ===== 3. 训练模型 =====
    trainer = LSTMTransformerTrainer(
        model=model,
        feature_scaler=processor.feature_scaler,
        target_scaler=processor.target_scaler,
        device=device
    )
    
    print("\n" + "="*60)
    print("开始训练 LSTM-Transformer 模型")
    print("="*60)
    
    # 训练参数：ADAM优化器，学习率0.001
    best_loss = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=100,                      # 增加训练轮数
        lr=0.001,                        # 学习率: 0.001
        patience=10,                     # 早停耐心值
        save_path='result/power_lstm_transformer_model.pth'
    )
    
    # ===== 4. 评估模型 =====
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    predictions, targets = trainer.predict(test_loader)
    metrics = calculate_metrics(predictions, targets)
    
    print(f"\n测试集评估结果:")
    print(f"  RMSE: {metrics['RMSE']:.2f} W")
    print(f"  MAE:  {metrics['MAE']:.2f} W")
    print(f"  R²:   {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # 绘制预测散点图
    plot_predictions(predictions, targets)
    
    # 保存评估结果
    results_df = pd.DataFrame([metrics])
    results_df.to_csv('result/lstm_transformer_evaluation.csv', index=False)
    print(f"[INFO] 评估结果已保存: result/lstm_transformer_evaluation.csv")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"模型文件: result/power_lstm_transformer_model.pth")
    print(f"训练历史: result/lstm_transformer_training_history.png")
    print(f"散点图:   result/lstm_transformer_prediction_scatter.png")


if __name__ == "__main__":
    main()
