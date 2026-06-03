#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于Informer的瞬时功率预测模型

Informer (Zhou et al., 2021) 是专为长序列时序预测设计的Transformer变体。
核心创新：
1. ProbSparse自注意力机制：O(L logL)复杂度，高效处理长序列
2. 自注意力蒸馏：逐层压缩关键信息，减少冗余
3. 生成式解码器：一次性预测整个输出序列

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# 切换工作目录到项目根目录，确保相对路径正确
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {device}")


# ==================== Informer 模型定义 ====================

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConvTokenEmbedding(nn.Module):
    """
    卷积Token嵌入：线性投影 + 深度可分离卷积
    同时捕获特征投影（线性）和短期时序模式（局部卷积），优于纯线性嵌入
    """
    def __init__(self, input_size, d_model, dropout=0.1):
        super(ConvTokenEmbedding, self).__init__()
        self.linear = nn.Linear(input_size, d_model)
        # 深度可分离卷积: 深度卷积(groups=d_model)捕获每通道局部模式 + 点卷积混合通道
        self.depth_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.point_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, input_size)
        x = self.linear(x)                                           # (B, L, d_model)
        conv_out = self.depth_conv(x.transpose(1, 2))                # (B, d_model, L)
        conv_out = self.point_conv(conv_out).transpose(1, 2)         # (B, L, d_model)
        x = self.norm(self.act(x + conv_out))                        # 残差 + 归一化
        return self.dropout(x)


class InformerEncoderLayer(nn.Module):
    """
    Informer编码器层: Pre-Norm MHA + 前馈网络
    Pre-Norm（归一化在子层之前）梯度流更顺畅，比Post-Norm收敛更稳定
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.2):
        super(InformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-Norm: 先归一化，再做注意力/FFN，再残差相加
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class InformerDistillingLayer(nn.Module):
    """
    自注意力蒸馏层: 通过MaxPooling压缩序列长度
    每次蒸馏后序列长度减半，保留最重要的时序特征
    """
    def __init__(self, d_model):
        super(InformerDistillingLayer, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x_conv = self.conv(x.transpose(1, 2))   # (batch, d_model, seq_len)
        x_conv = self.activation(x_conv)
        x_conv = self.pool(x_conv)               # (batch, d_model, seq_len//2)
        x_conv = x_conv.transpose(1, 2)          # (batch, seq_len//2, d_model)
        x_conv = self.norm(x_conv)
        return x_conv


class InformerModel(nn.Module):
    """
    Informer功率预测模型（传统Transformer Encoder结构）

    结构: Linear嵌入 → sqrt缩放 → 位置编码 → num_encoder_layers×TransformerEncoderLayer → MLP输出
    输入: (batch_size, seq_len, input_size)
    输出: (batch_size, seq_len)
    """
    def __init__(self, input_size=8, d_model=256, n_heads=8, num_encoder_layers=3,
                 d_ff=1024, dropout=0.2, max_len=500):
        super(InformerModel, self).__init__()
        self.d_model = d_model

        # 输入嵌入
        self.input_embedding = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # num_encoder_layers层标准Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 输出头
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            key_padding_mask: (batch_size, seq_len)，True表示填充位置
        Returns:
            outputs: (batch_size, seq_len)
        """
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        return self.output_layer(x).squeeze(-1)


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
    # padding_mask: True = 填充位置（传入key_padding_mask，屏蔽注意力中的填充）
    padding_mask = torch.ones(len(sequences), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_mask[i, :length] = False
    # valid_mask: True = 有效位置（用于损失计算）
    valid_mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        valid_mask[i, :length] = True
    return padded_sequences, padded_targets, padding_mask, valid_mask, lengths


class DataProcessor:
    """数据处理器"""
    def __init__(self, min_seq_len=20, max_seq_len=500):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
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
        df['Power'] = (df['Voltage'] / 1000.0) * (df['Current'] / 1000.0)
        df['wind_angle'] = np.abs(df['Wind Direct'] - df['Course'])
        df['wind_angle'] = df['wind_angle'].apply(lambda x: x if x <= 180 else 360 - x)

        if 'payload' in df.columns:
            payload_missing = df['payload'].isna().sum()
            if payload_missing > 0:
                print(f"[INFO] 载荷缺失值: {payload_missing} 条，用0填充")
                df['payload'] = df['payload'].fillna(0.0)

        df = df.dropna(subset=self.feature_cols + ['Power'])
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

class InformerTrainer:
    """Informer模型训练器"""
    def __init__(self, model, feature_scaler, target_scaler, device):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader, val_loader, epochs=100, lr=0.001,
              patience=20, save_path='result/power_informer_model.pth'):
        """训练模型"""
        criterion = nn.MSELoss(reduction='none')
        # 对齐Transformer: AdamW(weight_decay=0.01) + CosineAnnealingWarmRestarts
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
                outputs = self.model(batch_seq, key_padding_mask=padding_mask)

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

                    outputs = self.model(batch_seq, key_padding_mask=padding_mask)
                    loss = criterion(outputs, batch_tgt)
                    masked_loss = (loss * valid_mask).sum()

                    val_loss += masked_loss.item()
                    val_samples += valid_mask.sum().item()

            avg_val_loss = val_loss / val_samples
            self.val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model(save_path)
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f} | Best: {best_val_loss:.6f}")

            if patience_counter >= patience:
                print(f"\n[INFO] 早停：在第 {epoch+1} 轮停止训练")
                break

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
                outputs = self.model(batch_seq, key_padding_mask=padding_mask)

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
        torch.save({'model_state_dict': self.model.state_dict()}, path)

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
    print("Informer 瞬时功率预测模型训练")
    print("="*60)

    np.random.seed(42)
    torch.manual_seed(42)

    # ===== 1. 数据加载与预处理 =====
    data_dirs = [
        "Drone_energy_dataset/UAS04028624"
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
    model = InformerModel(
        input_size=8,              # 8个输入特征（包含载荷）
        d_model=256,               # 对齐hidden_size=256
        n_heads=8,                 # 多头注意力头数
        num_encoder_layers=3,      # 对齐num_layers=3
        d_ff=1024,                 # 前馈维度（4×d_model，对齐Transformer参数表）
        dropout=0.2,               # 对齐其他模型dropout=0.2
        max_len=500                # 最大序列长度
    ).to(device)

    print(f"\n[INFO] 模型结构:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 总参数量: {total_params:,}")

    # ===== 3. 训练模型 =====
    trainer = InformerTrainer(model, processor.feature_scaler, processor.target_scaler, device)

    print("\n" + "="*60)
    print("开始训练 Informer 模型")
    print("="*60)

    best_loss = trainer.train(
        train_loader, test_loader,
        epochs=150,
        lr=0.001,
        patience=25,
        save_path='result/power_informer_model.pth'
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

    results_df = pd.DataFrame([metrics])
    results_df.to_csv('result/informer_evaluation.csv', index=False)

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"模型文件: result/power_informer_model.pth")


if __name__ == "__main__":
    main()
