#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCN (Temporal Convolutional Network) 瞬时功率预测模型

TCN 通过因果膨胀卷积（Causal Dilated Convolution）和残差连接捕获长距离时序依赖，
同时保持 O(1) 的并行性，避免 RNN 的序列瓶颈。

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

模型架构: TCN (Temporal Convolutional Network)
- 多个残差块，每块包含两层因果膨胀卷积
- 膨胀率按 2^i 指数增长，感受野随层数指数扩张
- 权重归一化 + ReLU 激活 + Dropout 正则化
- 全连接输出层
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
from torch.nn.utils import weight_norm

# 切换工作目录到项目根目录，确保相对路径正确
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {device}")


# ==================== TCN 模型定义（改进版） ====================

class Chomp1d(nn.Module):
    """裁剪因果卷积多余的填充，保证因果性（不使用未来信息）"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ImprovedTCNBlock(nn.Module):
    """
    改进 TCN 残差块: GELU + 因果膨胀卷积 + 残差连接

    改进点:
    - GELU 激活函数: 比 ReLU 更平滑，避免死神经元
    - 所有块通道数一致（统一由输入嵌入层处理维度转换）
    """
    def __init__(self, n_channels, kernel_size, dilation, dropout=0.2):
        super(ImprovedTCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            n_channels, n_channels, kernel_size, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_channels, n_channels, kernel_size, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: (batch, n_channels, seq_len)
        res = x
        x = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        x = self.drop2(self.act2(self.chomp2(self.conv2(x))))
        return x + res  # 残差连接


class TCNModel(nn.Module):
    """
    改进版 TCN 功率预测模型

    改进架构:
    1. 输入嵌入层 (Linear): 将特征维度从 input_size 统一投影到 num_channels，
       将特征变换与时序建模分离
    2. num_levels 个改进 TCN 块 (GELU 激活，膨胀率 2^i): 按顺序传递，只使用
       最终块的输出作为特征表示（避免 skip 平均稀释有效信息）
    3. 输出层: LayerNorm + 全连接

    感受野 = 1 + (kernel_size - 1) * 2 * (2^num_levels - 1)
    """
    def __init__(self, input_size=8, num_channels=128, kernel_size=3,
                 num_levels=8, dropout=0.2):
        super(TCNModel, self).__init__()

        # 输入嵌入层：统一将 input_size 维特征投影到 num_channels
        self.input_proj = nn.Linear(input_size, num_channels)

        # 改进 TCN 块（所有块通道数一致，顺序传递）
        self.blocks = nn.ModuleList([
            ImprovedTCNBlock(num_channels, kernel_size, 2 ** i, dropout)
            for i in range(num_levels)
        ])

        # 输出层
        self.output_norm = nn.LayerNorm(num_channels)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            outputs: (batch_size, seq_len)
        """
        x = self.input_proj(x)     # (batch, seq_len, num_channels)
        x = x.transpose(1, 2)      # (batch, num_channels, seq_len)

        for block in self.blocks:
            x = block(x)           # 顺序传递，只保留最终块输出

        out = x.transpose(1, 2)    # (batch, seq_len, num_channels)
        out = self.output_norm(out)
        return self.fc(out).squeeze(-1)  # (batch, seq_len)


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

class TCNTrainer:
    """TCN 模型训练器"""
    def __init__(self, model, feature_scaler, target_scaler, device):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train(self, train_loader, val_loader, epochs=100, lr=0.001,
              patience=20, save_path='result/power_tcn_model.pth'):
        """训练模型"""
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

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


def plot_training_curves(train_losses, val_losses, save_path='result/tcn_training_curves.png'):
    """绘制训练曲线"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=1.5)
    ax.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=1.5)
    ax.set_xlabel('轮次', fontsize=12)
    ax.set_ylabel('MSE 损失', fontsize=12)
    ax.set_title('TCN 模型训练曲线', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[INFO] 训练曲线已保存: {save_path}")


def main():
    """主函数"""
    print("="*60)
    print("TCN 瞬时功率预测模型训练")
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
    # 感受野 = 1 + (3-1)*2*(2^8 - 1) = 1 + 2*2*255 = 1021 个时间步
    model = TCNModel(
        input_size=8,        # 8个输入特征（包含载荷）
        num_channels=128,    # 每个 TCN 块的通道数
        kernel_size=3,       # 卷积核大小
        num_levels=8,        # TCN 块数量（膨胀率：1,2,4,8,16,32,64,128）
        dropout=0.2          # Dropout 比例
    ).to(device)

    print(f"\n[INFO] 模型结构:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 总参数量: {total_params:,}")

    # 计算 TCN 感受野
    kernel_size = 3
    num_levels = 8
    receptive_field = 1 + (kernel_size - 1) * 2 * (2 ** num_levels - 1)
    print(f"[INFO] TCN 感受野: {receptive_field} 个时间步")

    # ===== 3. 训练模型 =====
    trainer = TCNTrainer(model, processor.feature_scaler, processor.target_scaler, device)

    print("\n" + "="*60)
    print("开始训练 TCN 模型")
    print("="*60)

    best_loss = trainer.train(
        train_loader, test_loader,
        epochs=100,          # 统一为100轮
        lr=0.001,            # 统一学习率
        patience=20,         # 统一早停耐心值
        save_path='result/power_tcn_model.pth'
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
    results_df.to_csv('result/tcn_evaluation.csv', index=False)

    # 绘制训练曲线
    plot_training_curves(trainer.train_losses, trainer.val_losses)

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"模型文件: result/power_tcn_model.pth")
    print(f"评估结果: result/tcn_evaluation.csv")
    print(f"训练曲线: result/tcn_training_curves.png")


if __name__ == "__main__":
    main()
