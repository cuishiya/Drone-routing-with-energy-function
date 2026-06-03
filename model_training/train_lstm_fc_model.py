#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM-FC 瞬时功率预测模型（对照实验基线模型）

简单的 LSTM + 全连接层结构，作为对照实验的基线模型。
与其他复杂模型（Bi-LSTM、LSTM-Transformer等）对比，验证复杂架构的有效性。

模型架构:
- 单向 LSTM 层提取时序特征
- 全连接层输出功率预测
- 无注意力机制、无双向结构、无 Transformer 模块

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

# 切换工作目录到项目根目录，确保相对路径正确
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {device}")


# ==================== 数据集类 ====================

class FlightSequenceDataset(Dataset):
    """
    飞行序列数据集
    将每个航次(Order ID)的轨迹数据作为一个完整序列
    """
    def __init__(self, sequences, targets):
        """
        Args:
            sequences: 输入特征序列列表，每个元素shape为 (seq_len, num_features)
            targets: 目标功率序列列表，每个元素shape为 (seq_len,)
        """
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
    """
    自定义批处理函数，处理变长序列
    使用padding将同一批次的序列填充到相同长度
    """
    sequences, targets = zip(*batch)
    
    # 获取批次中最长序列的长度
    max_len = max(seq.shape[0] for seq in sequences)
    num_features = sequences[0].shape[1]
    
    # 创建填充后的张量
    batch_size = len(sequences)
    padded_sequences = torch.zeros(batch_size, max_len, num_features)
    padded_targets = torch.zeros(batch_size, max_len)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, (seq, tgt) in enumerate(zip(sequences, targets)):
        length = seq.shape[0]
        padded_sequences[i, :length, :] = seq
        padded_targets[i, :length] = tgt
        lengths[i] = length
    
    return padded_sequences, padded_targets, lengths


# ==================== LSTM-FC 模型 ====================

class LSTMFCModel(nn.Module):
    """
    LSTM-FC 功率预测模型（基线对照模型）
    
    结构: 输入嵌入 -> 多层LSTM -> LayerNorm -> 深层全连接网络
    - 无注意力机制
    - 无双向结构
    - 无Transformer模块
    
    优化点:
    - 输入嵌入层提升特征维度
    - LayerNorm稳定训练
    - 更深的全连接网络增强特征提取
    - 残差连接帮助梯度流动
    
    输入: (batch_size, seq_len, input_size)
    输出: (batch_size, seq_len) 每个时刻的功率预测
    """
    def __init__(self, input_size=8, hidden_size=256, num_layers=3, dropout=0.2):
        super(LSTMFCModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入嵌入层：提升特征维度
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # 多层 LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 全连接输出网络（两层）
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 残差连接的投影层（用于将输入嵌入与LSTM输出相加）
        self.residual_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            outputs: (batch_size, seq_len) - 预测的功率序列
        """
        # 输入嵌入
        embedded = self.input_embedding(x)  # (batch, seq_len, hidden_size)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_size)
        
        # 残差连接 + LayerNorm
        residual = self.residual_proj(embedded)
        lstm_out = self.layer_norm(lstm_out + residual)
        
        # 全连接层输出
        outputs = self.fc(lstm_out)  # (batch, seq_len, 1)
        return outputs.squeeze(-1)  # (batch, seq_len)


# ==================== 数据处理 ====================

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, data_dirs=None, min_seq_len=10, max_seq_len=500):
        """
        Args:
            data_dirs: 数据目录列表
            min_seq_len: 最小序列长度（过滤过短的航次）
            max_seq_len: 最大序列长度（截断过长的航次）
        """
        if data_dirs is None:
            data_dirs = [
                "Drone_energy_dataset/UAS04028624",
                "Drone_energy_dataset/UAS04028648"
            ]
        self.data_dirs = data_dirs
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        
        # 特征列名（8个特征，包含载荷）
        self.feature_cols = ['Height', 'VS (m/s)', 'GS (m/s)', 'Wind Speed', 
                            'Temperature', 'Humidity', 'wind_angle', 'payload']
        
        # 标准化器
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def load_data(self):
        """加载所有飞行轨迹数据（包含载荷信息）"""
        all_data = []
        
        for data_dir in self.data_dirs:
            trajectory_path = os.path.join(data_dir, "flightTrajectory.xlsx")
            record_path = os.path.join(data_dir, "flightRecord.xlsx")
            
            if os.path.exists(trajectory_path):
                print(f"[INFO] 加载轨迹数据: {trajectory_path}")
                df = pd.read_excel(trajectory_path)
                print(f"  - 数据量: {len(df)} 条记录, {df['Order ID'].nunique()} 个航次")
                
                # 加载飞行记录以获取载荷信息
                if os.path.exists(record_path):
                    print(f"[INFO] 加载飞行记录: {record_path}")
                    record_df = pd.read_excel(record_path)
                    
                    # 通过Order ID关联载荷信息
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
        
        if not all_data:
            raise ValueError("未找到任何飞行轨迹数据文件")
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"[INFO] 总数据量: {len(combined_df)} 条记录, {combined_df['Order ID'].nunique()} 个航次")
        
        return combined_df
    
    def preprocess(self, df):
        """数据预处理"""
        print("[INFO] 数据预处理...")
        
        # 计算功率 (W) = 电压(mV->V) × 电流(mA->A)
        df['Power'] = (df['Voltage'] / 1000.0) * (df['Current'] / 1000.0)
        
        # 计算风向夹角 (相对风向)
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
        df = df[df['Power'] > 0]  # 功率必须为正
        df = df[df['Power'] < 15000]  # 过滤异常高功率（无人机最大功率约12kW）
        
        print(f"[INFO] 预处理后数据量: {len(df)} 条记录")
        
        return df
    
    def create_sequences(self, df, test_size=0.2):
        """
        按航次(Order ID)创建序列
        
        Returns:
            train_sequences, test_sequences: 训练和测试序列
        """
        print("[INFO] 创建序列数据...")
        
        sequences = []
        targets = []
        order_ids = []
        
        # 按Order ID分组
        grouped = df.groupby('Order ID')
        
        for order_id, group in grouped:
            # 按时间戳排序
            group = group.sort_values('Time Stamp')
            
            seq_len = len(group)
            
            # 过滤过短的序列
            if seq_len < self.min_seq_len:
                continue
            
            # 截断过长的序列
            if seq_len > self.max_seq_len:
                group = group.head(self.max_seq_len)
            
            # 提取特征和目标
            features = group[self.feature_cols].values
            power = group['Power'].values
            
            sequences.append(features)
            targets.append(power)
            order_ids.append(order_id)
        
        print(f"[INFO] 有效序列数量: {len(sequences)}")
        print(f"[INFO] 序列长度范围: {min(len(s) for s in sequences)} - {max(len(s) for s in sequences)}")
        
        # 划分训练集和测试集（按航次划分，保持序列完整性）
        n_samples = len(sequences)
        indices = np.random.permutation(n_samples)
        n_test = int(n_samples * test_size)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        train_sequences = [sequences[i] for i in train_indices]
        train_targets = [targets[i] for i in train_indices]
        test_sequences = [sequences[i] for i in test_indices]
        test_targets = [targets[i] for i in test_indices]
        
        print(f"[INFO] 训练集: {len(train_sequences)} 个航次")
        print(f"[INFO] 测试集: {len(test_sequences)} 个航次")
        
        # 拟合标准化器（使用训练集）
        all_train_features = np.vstack(train_sequences)
        all_train_targets = np.concatenate(train_targets).reshape(-1, 1)
        
        self.feature_scaler.fit(all_train_features)
        self.target_scaler.fit(all_train_targets)
        
        # 标准化序列
        train_sequences = [self.feature_scaler.transform(seq) for seq in train_sequences]
        test_sequences = [self.feature_scaler.transform(seq) for seq in test_sequences]
        train_targets = [self.target_scaler.transform(t.reshape(-1, 1)).flatten() for t in train_targets]
        test_targets = [self.target_scaler.transform(t.reshape(-1, 1)).flatten() for t in test_targets]
        
        return train_sequences, test_sequences, train_targets, test_targets


# ==================== 训练器 ====================

class LSTMFCTrainer:
    """LSTM-FC 模型训练器"""
    
    def __init__(self, model, device, feature_scaler, target_scaler):
        self.model = model.to(device)
        self.device = device
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, 
              patience=20, save_path='result/power_lstm_fc_model.pth'):
        """训练模型"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\n" + "="*60)
        print("开始训练 LSTM-FC 模型（基线对照模型）")
        print("="*60)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (sequences, targets, lengths) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(sequences)
                
                # 计算损失（只计算有效长度部分）
                mask = torch.zeros_like(targets, dtype=torch.bool)
                for i, length in enumerate(lengths):
                    mask[i, :length] = True
                
                loss = criterion(outputs[mask], targets[mask])
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            val_loss = self.evaluate(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model(save_path)
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Best: {best_val_loss:.6f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"\n[INFO] 早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 加载最佳模型
        self.load_model(save_path)
        print(f"\n[INFO] 训练完成，最佳验证损失: {best_val_loss:.6f}")
        
        return best_val_loss
    
    def evaluate(self, data_loader, criterion=None):
        """评估模型"""
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets, lengths in data_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                
                # 计算损失（只计算有效长度部分）
                mask = torch.zeros_like(targets, dtype=torch.bool)
                for i, length in enumerate(lengths):
                    mask[i, :length] = True
                
                loss = criterion(outputs[mask], targets[mask])
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def predict(self, data_loader):
        """预测并返回真实值和预测值"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets, lengths in data_loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                
                # 提取有效长度部分
                for i, length in enumerate(lengths):
                    pred = outputs[i, :length].cpu().numpy()
                    true = targets[i, :length].numpy()
                    
                    # 反标准化
                    pred = self.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                    true = self.target_scaler.inverse_transform(true.reshape(-1, 1)).flatten()
                    
                    all_predictions.append(pred)
                    all_targets.append(true)
        
        return all_predictions, all_targets
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 保存模型权重
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)
        
        # 单独保存标准化器（使用pickle）
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
        
        # 加载标准化器
        scaler_path = path.replace('.pth', '_scalers.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.feature_scaler = scalers['feature_scaler']
                self.target_scaler = scalers['target_scaler']


# ==================== 评估与可视化 ====================

def calculate_metrics(predictions, targets):
    """计算评估指标"""
    # 展平所有序列
    all_pred = np.concatenate(predictions)
    all_true = np.concatenate(targets)
    
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    mae = mean_absolute_error(all_true, all_pred)
    r2 = r2_score(all_true, all_pred)
    
    # 计算MAPE（避免除零）
    mask = all_true != 0
    mape = np.mean(np.abs((all_true[mask] - all_pred[mask]) / all_true[mask])) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def plot_training_history(train_losses, val_losses, save_path='result/lstm_fc_training_history.png'):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('LSTM-FC 训练历史（基线对照模型）', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 训练历史图已保存: {save_path}")


def plot_prediction_scatter(predictions, targets, save_path='result/lstm_fc_prediction_scatter.png'):
    """绘制预测值vs真实值散点图"""
    all_pred = np.concatenate(predictions)
    all_true = np.concatenate(targets)
    
    # 计算指标
    r2 = r2_score(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    
    plt.figure(figsize=(8, 8))
    
    # 散点图
    plt.scatter(all_true, all_pred, alpha=0.3, s=5, c='#4A90D9')
    
    # 对角线
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
    
    plt.xlabel('真实功率 (W)', fontsize=12)
    plt.ylabel('预测功率 (W)', fontsize=12)
    plt.title(f'LSTM-FC 预测结果（基线对照模型）\nR² = {r2:.4f}, RMSE = {rmse:.2f} W', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置相同的坐标轴范围
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 散点图已保存: {save_path}")


def plot_sequence_comparison(predictions, targets, n_samples=3, 
                            save_path='result/lstm_fc_sequence_comparison.png'):
    """绘制序列预测对比图"""
    n_samples = min(n_samples, len(predictions))
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    # 随机选择样本
    indices = np.random.choice(len(predictions), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        pred = predictions[idx]
        true = targets[idx]
        time_steps = np.arange(len(pred))
        
        ax.plot(time_steps, true, 'b-', label='真实功率', linewidth=1.5, alpha=0.8)
        ax.plot(time_steps, pred, 'r--', label='预测功率', linewidth=1.5, alpha=0.8)
        
        # 计算该序列的指标
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        
        ax.set_xlabel('时间步', fontsize=10)
        ax.set_ylabel('功率 (W)', fontsize=10)
        ax.set_title(f'航次 {idx+1}: R² = {r2:.4f}, RMSE = {rmse:.2f} W', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 序列对比图已保存: {save_path}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("="*60)
    print("LSTM-FC 瞬时功率预测模型训练（基线对照模型）")
    print("="*60)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # ===== 1. 数据处理 =====
    processor = DataProcessor(
        min_seq_len=20,   # 最小序列长度
        max_seq_len=500   # 最大序列长度
    )
    
    # 加载和预处理数据
    df = processor.load_data()
    df = processor.preprocess(df)
    
    # 创建序列
    train_sequences, test_sequences, train_targets, test_targets = processor.create_sequences(df)
    
    # 创建数据集和数据加载器
    train_dataset = FlightSequenceDataset(train_sequences, train_targets)
    test_dataset = FlightSequenceDataset(test_sequences, test_targets)
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, 
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, 
        collate_fn=collate_fn, num_workers=0
    )
    
    # ===== 2. 创建模型 =====
    # 统一参数配置（与其他模型保持一致以便公平对比）
    model = LSTMFCModel(
        input_size=8,        # 8个输入特征（包含载荷）
        hidden_size=256,     # 隐藏层大小（统一为256）
        num_layers=3,        # LSTM层数（统一为3）
        dropout=0.2          # Dropout比例（统一为0.2）
    )
    
    print(f"\n[INFO] 模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 总参数量: {total_params:,}")
    
    # ===== 3. 训练模型 =====
    trainer = LSTMFCTrainer(
        model=model,
        device=device,
        feature_scaler=processor.feature_scaler,
        target_scaler=processor.target_scaler
    )
    
    # 统一训练参数
    best_loss = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=100,          # 统一为100轮
        lr=0.001,            # 统一学习率
        patience=20,         # 统一早停耐心值
        save_path='result/power_lstm_fc_model.pth'
    )
    
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
    
    # ===== 5. 可视化 =====
    print("\n[INFO] 生成可视化图表...")
    
    plot_training_history(
        trainer.train_losses, 
        trainer.val_losses,
        save_path='result/lstm_fc_training_history.png'
    )
    
    plot_prediction_scatter(
        predictions, 
        targets,
        save_path='result/lstm_fc_prediction_scatter.png'
    )
    
    plot_sequence_comparison(
        predictions, 
        targets,
        n_samples=5,
        save_path='result/lstm_fc_sequence_comparison.png'
    )
    
    # 保存评估结果
    results_df = pd.DataFrame([metrics])
    results_df.to_csv('result/lstm_fc_evaluation.csv', index=False)
    print(f"[INFO] 评估结果已保存: result/lstm_fc_evaluation.csv")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"模型文件: result/power_lstm_fc_model.pth")
    print(f"训练历史: result/lstm_fc_training_history.png")
    print(f"散点图:   result/lstm_fc_prediction_scatter.png")
    print(f"序列对比: result/lstm_fc_sequence_comparison.png")


if __name__ == "__main__":
    main()
