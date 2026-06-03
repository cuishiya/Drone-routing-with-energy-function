#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一评估所有时序功率预测模型

支持的模型:
- LSTM Seq2Seq
- GRU Seq2Seq
- Bi-LSTM
- Transformer
- LSTM-Transformer
- LSTM-FC（基线对照模型）
- Informer（稀疏自注意力长序列预测）

评估指标:
- RMSE: 均方根误差
- MAE: 平均绝对误差
- R²: 决定系数
- MAPE: 平均绝对百分比误差
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
from torch.nn.utils import weight_norm as _weight_norm

# 切换工作目录到项目根目录，确保相对路径正确
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {device}")


# ==================== 标准模型定义 ====================

# 标准 LSTM
class LSTMModel(nn.Module):
    """标准LSTM功率预测模型"""
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)


# LSTM-FC（基线对照模型）
class LSTMFCModel(nn.Module):
    """LSTM-FC功率预测模型（基线对照模型）- 优化版"""
    def __init__(self, input_size=8, hidden_size=256, num_layers=3, dropout=0.2):
        super(LSTMFCModel, self).__init__()
        self.hidden_size = hidden_size
        
        # 输入嵌入层
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # 多层 LSTM
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 全连接输出网络（两层）
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 残差连接投影层
        self.residual_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        embedded = self.input_embedding(x)
        lstm_out, _ = self.lstm(embedded)
        residual = self.residual_proj(embedded)
        lstm_out = self.layer_norm(lstm_out + residual)
        return self.fc(lstm_out).squeeze(-1)


# 标准 GRU
class GRUModel(nn.Module):
    """标准GRU功率预测模型"""
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out).squeeze(-1)


# 标准 Bi-LSTM
class BiLSTMModel(nn.Module):
    """标准双向LSTM功率预测模型"""
    def __init__(self, input_size=8, hidden_size=128, num_layers=3, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 双向所以 * 2
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out).squeeze(-1)


# Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size=8, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, max_len=500):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, src_key_padding_mask=None):
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_layer(x).squeeze(-1)


# LSTM-Transformer 串行模型
class LSTMTransformerModel(nn.Module):
    """LSTM-Transformer串行模型: 输入 → LSTM → Transformer → 输出"""
    def __init__(self, input_size=8, d_model=256, lstm_layers=2,
                 nhead=8, num_transformer_layers=3, dim_feedforward=512, 
                 dropout=0.1, max_len=500):
        super(LSTMTransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_size, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model // 2, num_layers=lstm_layers,
                           batch_first=True, dropout=dropout if lstm_layers > 1 else 0, bidirectional=True)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, src_key_padding_mask=None):
        x = self.input_embedding(x)
        x, _ = self.lstm(x)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_layer(x).squeeze(-1)


# ==================== TCN 模型定义（改进版） ====================


class _Chomp1d(nn.Module):
    """裁剪因果卷积多余的填充，保证因果性（不使用未来信息）"""
    def __init__(self, chomp_size):
        super(_Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class _ImprovedTCNBlock(nn.Module):
    """
    改进 TCN 残差块: GELU + 因果膨胀卷积 + 残差连接

    改进点:
    - GELU 激活: 比 ReLU 更平滑，避免死神经元
    - 所有块通道数一致（统一由输入嵌入层处理维度转换）
    """
    def __init__(self, n_channels, kernel_size, dilation, dropout=0.2):
        super(_ImprovedTCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = _weight_norm(nn.Conv1d(
            n_channels, n_channels, kernel_size, padding=padding, dilation=dilation
        ))
        self.chomp1 = _Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = _weight_norm(nn.Conv1d(
            n_channels, n_channels, kernel_size, padding=padding, dilation=dilation
        ))
        self.chomp2 = _Chomp1d(padding)
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
    1. 输入嵌内层 (Linear): 将特征维度从 input_size 统一投影到 num_channels，
       将特征变换与时序建模分离
    2. num_levels 个改进 TCN 块 (GELU 激活，膨胀率 2^i): 按顺序传递，
       只使用最终块输出作为特征表示
    3. 输出层: LayerNorm + 全连接

    感受野 = 1 + (kernel_size - 1) * 2 * (2^num_levels - 1)
    """
    def __init__(self, input_size=8, num_channels=128, kernel_size=3,
                 num_levels=8, dropout=0.2):
        super(TCNModel, self).__init__()

        # 输入嵌内层：统一将 input_size 维特征投影到 num_channels
        self.input_proj = nn.Linear(input_size, num_channels)

        # 改进 TCN 块（所有块通道数一致，顺序传递）
        self.blocks = nn.ModuleList([
            _ImprovedTCNBlock(num_channels, kernel_size, 2 ** i, dropout)
            for i in range(num_levels)
        ])

        # 输出层
        self.output_norm = nn.LayerNorm(num_channels)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)     # (batch, seq_len, num_channels)
        x = x.transpose(1, 2)      # (batch, num_channels, seq_len)

        for block in self.blocks:
            x = block(x)           # 顺序传递，只保留最终块输出

        out = x.transpose(1, 2)    # (batch, seq_len, num_channels)
        out = self.output_norm(out)
        return self.fc(out).squeeze(-1)  # (batch, seq_len)


# ==================== Informer 模型定义 ====================

class InformerEncoderLayer(nn.Module):
    """Informer编码器层: Pre-Norm MHA + 前馈网络（梯度更顺畅，收敛更稳定）"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.2):
        super(InformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class InformerDistillingLayer(nn.Module):
    """自注意力蒸馏层: 通过MaxPooling压缩序列长度"""
    def __init__(self, d_model):
        super(InformerDistillingLayer, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_conv = self.conv(x.transpose(1, 2))
        x_conv = self.activation(x_conv)
        x_conv = self.pool(x_conv)
        x_conv = x_conv.transpose(1, 2)
        return self.norm(x_conv)


class ConvTokenEmbedding(nn.Module):
    """卷积Token嵌入：线性投影 + 深度可分离卷积，融合特征投影和短期时序模式"""
    def __init__(self, input_size, d_model, dropout=0.1):
        super(ConvTokenEmbedding, self).__init__()
        self.linear = nn.Linear(input_size, d_model)
        self.depth_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.point_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        conv_out = self.depth_conv(x.transpose(1, 2))
        conv_out = self.point_conv(conv_out).transpose(1, 2)
        return self.dropout(self.norm(self.act(x + conv_out)))


class InformerModel(nn.Module):
    """Informer功率预测模型（传统Transformer Encoder结构）"""
    def __init__(self, input_size=8, d_model=256, n_heads=8, num_encoder_layers=3,
                 d_ff=1024, dropout=0.2, max_len=500):
        super(InformerModel, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.output_layer(x).squeeze(-1)


# ==================== 模型评估器 ====================

class ModelEvaluator:
    """统一模型评估器"""
    
    def __init__(self):
        self.device = device
        self.feature_cols = ['Height', 'VS (m/s)', 'GS (m/s)', 'Wind Speed', 
                            'Temperature', 'Humidity', 'wind_angle', 'payload']
        
        # 模型配置
        self.model_configs = {
            'LSTM': {
                'path': 'result/power_lstm_model.pth',
                'class': LSTMModel,
                'params': {'input_size': 8, 'hidden_size': 256, 'num_layers': 3, 'dropout': 0.2}
            },
            'TCN': {
                'path': 'result/power_tcn_model.pth',
                'class': TCNModel,
                'params': {'input_size': 8, 'num_channels': 128, 'kernel_size': 3,
                           'num_levels': 8, 'dropout': 0.2}
            },
            'GRU': {
                'path': 'result/power_gru_model.pth',
                'class': GRUModel,
                'params': {'input_size': 8, 'hidden_size': 256, 'num_layers': 3, 'dropout': 0.2}
            },
            'Bi-LSTM': {
                'path': 'result/power_bilstm_model.pth',
                'class': BiLSTMModel,
                'params': {'input_size': 8, 'hidden_size': 256, 'num_layers': 3, 'dropout': 0.2}
            },
            'LSTM-FC': {
                'path': 'result/power_lstm_fc_model.pth',
                'class': LSTMFCModel,
                'params': {'input_size': 8, 'hidden_size': 256, 'num_layers': 3, 'dropout': 0.2}
            },
            'Transformer': {
                'path': 'result/power_transformer_model.pth',
                'class': TransformerModel,
                'params': {'input_size': 8, 'd_model': 256, 'nhead': 8, 'num_layers': 3, 'dim_feedforward': 1024, 'dropout': 0.2, 'max_len': 500}
            },
            'Informer': {
                'path': 'result/power_informer_model.pth',
                'class': InformerModel,
                'params': {'input_size': 8, 'd_model': 256, 'n_heads': 8, 'num_encoder_layers': 3,
                           'd_ff': 1024, 'dropout': 0.2, 'max_len': 500}
            },
            'LSTM-Transformer': {
                'path': 'result/power_lstm_transformer_model.pth',
                'class': LSTMTransformerModel,
                'params': {'input_size': 8, 'd_model': 256, 'lstm_layers': 2, 
                          'nhead': 8, 'num_transformer_layers': 3, 'dim_feedforward': 512, 'dropout': 0.1, 'max_len': 500}
            }
        }
        
        self.models = {}
        self.scalers = {}
    
    def load_models(self):
        """加载所有模型"""
        for name, config in self.model_configs.items():
            if os.path.exists(config['path']):
                try:
                    model = config['class'](**config['params'])
                    checkpoint = torch.load(config['path'], map_location=self.device, weights_only=True)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    model.eval()
                    self.models[name] = model
                    
                    # 加载标准化器
                    scaler_path = config['path'].replace('.pth', '_scalers.pkl')
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            self.scalers[name] = pickle.load(f)
                    
                    print(f"[OK] {name} 模型加载成功")
                except Exception as e:
                    print(f"[ERROR] {name} 模型加载失败: {e}")
            else:
                print(f"[WARNING] {name} 模型文件不存在: {config['path']}")
    
    def load_test_data(self, test_path=None, min_seq_len=20, max_seq_len=500):
        """加载测试数据（包含载荷信息）"""
        if test_path is None:
            test_path = "Drone_energy_dataset/test_data/flightTrajectory.xlsx"
        
        # 获取对应的飞行记录路径
        test_dir = os.path.dirname(test_path)
        record_path = os.path.join(test_dir, "flightRecord.xlsx")
        
        print(f"[INFO] 加载测试数据: {test_path}")
        df = pd.read_excel(test_path)
        
        # 加载载荷信息
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
        
        # 预处理
        df['Power'] = (df['Voltage'] / 1000.0) * (df['Current'] / 1000.0)
        df['wind_angle'] = np.abs(df['Wind Direct'] - df['Course'])
        df['wind_angle'] = df['wind_angle'].apply(lambda x: x if x <= 180 else 360 - x)
        
        # 处理载荷缺失值
        df['payload'] = df['payload'].fillna(0.0)
        
        # 过滤
        df = df.dropna(subset=self.feature_cols + ['Power'])
        df = df[df['Power'] > 0]
        df = df[df['Power'] < 15000]
        
        # 按航次分组创建序列
        sequences = []
        targets = []
        order_ids = []
        
        grouped = df.groupby('Order ID')
        for order_id, group in grouped:
            group = group.sort_values('Time Stamp')
            seq_len = len(group)
            
            if seq_len < min_seq_len:
                continue
            if seq_len > max_seq_len:
                group = group.head(max_seq_len)
            
            features = group[self.feature_cols].values
            power = group['Power'].values
            
            sequences.append(features)
            targets.append(power)
            order_ids.append(order_id)
        
        print(f"[INFO] 加载了 {len(sequences)} 个航次序列")
        
        return sequences, targets, order_ids
    
    def predict_sequence(self, model_name, feature_sequence):
        """预测单个序列"""
        model = self.models.get(model_name)
        scalers = self.scalers.get(model_name, {})
        
        if model is None:
            return None
        
        # 标准化
        feature_scaler = scalers.get('feature_scaler')
        target_scaler = scalers.get('target_scaler')
        
        if feature_scaler is not None:
            feature_sequence = feature_scaler.transform(feature_sequence)
        
        # 预测
        with torch.no_grad():
            input_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0).to(self.device)
            output = model(input_tensor)
            power_sequence = output.squeeze(0).cpu().numpy()
        
        # 反标准化
        if target_scaler is not None:
            power_sequence = target_scaler.inverse_transform(power_sequence.reshape(-1, 1)).flatten()
        
        return np.maximum(0.0, power_sequence)
    
    def evaluate_model(self, model_name, sequences, targets):
        """评估单个模型"""
        if model_name not in self.models:
            return None, None, None
        
        all_predictions = []
        all_targets = []
        
        for seq, tgt in zip(sequences, targets):
            pred = self.predict_sequence(model_name, seq)
            if pred is not None:
                all_predictions.append(pred)
                all_targets.append(tgt)
        
        if not all_predictions:
            return None, None, None
        
        # 计算指标
        all_pred_flat = np.concatenate(all_predictions)
        all_true_flat = np.concatenate(all_targets)
        
        rmse = np.sqrt(mean_squared_error(all_true_flat, all_pred_flat))
        mae = mean_absolute_error(all_true_flat, all_pred_flat)
        r2 = r2_score(all_true_flat, all_pred_flat)
        
        mask = all_true_flat != 0
        mape = np.mean(np.abs((all_true_flat[mask] - all_pred_flat[mask]) / all_true_flat[mask])) * 100
        
        metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
        return metrics, all_predictions, all_targets
    
    def evaluate_all(self, sequences, targets):
        """评估所有模型"""
        results = {}
        predictions_dict = {}
        
        for model_name in self.models.keys():
            print(f"[INFO] 评估 {model_name} 模型...")
            metrics, preds, trues = self.evaluate_model(model_name, sequences, targets)
            if metrics is not None:
                results[model_name] = metrics
                predictions_dict[model_name] = (preds, trues)
        
        return results, predictions_dict


# ==================== 可视化函数 ====================

def plot_metrics_comparison(results, predictions_dict, time_interval=1.0, save_path='result/models_comparison.png'):
    """
    绘制模型指标对比图
    
    分两部分展示：
    1. 逐点功率预测指标：评估模型对每个时刻瞬时功率的预测精度
    2. 航次总能耗指标：评估模型对整个航次能耗（功率积分）的预测精度
    """
    if not results:
        print("[WARNING] 没有可用的评估结果")
        return
    
    models = list(results.keys())
    # 使用参考图片的配色风格（支持6个模型）
    color_map = {'LSTM': '#4A90D9', 'GRU': '#7BC47F', 'Bi-LSTM': '#F5A962',
                 'Transformer': '#E57373', 'LSTM-Transformer': '#9C27B0', 'LSTM-FC': '#00BCD4',
                 'Informer': '#20B2AA', 'TCN': '#FF8C00'}
    colors = [color_map.get(m, '#888888') for m in models]
    
    # 计算航次能耗指标
    energy_metrics = {}
    for model_name, (preds, trues) in predictions_dict.items():
        true_energies = []
        pred_energies = []
        for pred_seq, true_seq in zip(preds, trues):
            true_energy = np.sum(true_seq) * (time_interval / 3600.0)
            pred_energy = np.sum(pred_seq) * (time_interval / 3600.0)
            true_energies.append(true_energy)
            pred_energies.append(pred_energy)
        
        true_energies = np.array(true_energies)
        pred_energies = np.array(pred_energies)
        
        # 过滤掉实际能耗大于425 Wh的异常航次
        valid_mask = true_energies <= 425
        true_energies = true_energies[valid_mask]
        pred_energies = pred_energies[valid_mask]
        
        energy_rmse = np.sqrt(mean_squared_error(true_energies, pred_energies))
        energy_mae = mean_absolute_error(true_energies, pred_energies)
        energy_r2 = r2_score(true_energies, pred_energies)
        
        energy_metrics[model_name] = {'RMSE': energy_rmse, 'MAE': energy_mae, 'R2': energy_r2}
    
    # 创建2行3列的图表
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # ===== 第一行：逐点功率预测指标 =====
    power_metrics = ['RMSE', 'MAE', 'R2']
    power_units = ['W', 'W', '']
    power_labels = ['(a)', '(b)', '(c)']
    
    for idx, (metric, unit, label) in enumerate(zip(power_metrics, power_units, power_labels)):
        ax = axes[0, idx]
        values = [results[m][metric] for m in models]
        
        # 设置Y轴范围，不从0开始以突显差异
        if metric == 'R2':
            y_min = min(values) - 0.02
            y_max = max(values) + 0.02
        else:
            val_range = max(values) - min(values)
            y_min = min(values) - val_range * 0.3
            y_max = max(values) + val_range * 0.3
        
        bars = ax.bar(models, values, color=colors, edgecolor='none', width=0.6)
        
        # 在柱子上方显示数值（黑色）
        for bar, val in zip(bars, values):
            if metric == 'R2':
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (y_max-y_min)*0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=11, color='black')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (y_max-y_min)*0.02, 
                       f'{val:.1f}', ha='center', va='bottom', fontsize=11, color='black')
        
        # 使用LaTeX格式显示R²
        if metric == 'R2':
            ylabel = r'$R^2$ Score'
        else:
            ylabel = f'{metric} ({unit})'
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold', loc='left')
        ax.set_ylim(y_min, y_max + (y_max-y_min)*0.15)  # 留出数值显示空间
        ax.tick_params(axis='x', rotation=15)
        
        # 简洁的边框样式
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # ===== 第二行：航次总能耗指标 =====
    energy_units = ['Wh', 'Wh', '']
    energy_labels = ['(d)', '(e)', '(f)']
    
    for idx, (metric, unit, label) in enumerate(zip(power_metrics, energy_units, energy_labels)):
        ax = axes[1, idx]
        values = [energy_metrics[m][metric] for m in models]
        
        # 设置Y轴范围，不从0开始以突显差异
        if metric == 'R2':
            y_min = min(values) - 0.02
            y_max = max(values) + 0.02
        else:
            val_range = max(values) - min(values)
            y_min = min(values) - val_range * 0.3
            y_max = max(values) + val_range * 0.3
        
        bars = ax.bar(models, values, color=colors, edgecolor='none', width=0.6)
        
        # 在柱子上方显示数值（黑色）
        for bar, val in zip(bars, values):
            if metric == 'R2':
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (y_max-y_min)*0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=11, color='black')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (y_max-y_min)*0.02, 
                       f'{val:.2f}', ha='center', va='bottom', fontsize=11, color='black')
        
        # 使用LaTeX格式显示R²
        if metric == 'R2':
            ylabel = r'$R^2$ Score'
        else:
            ylabel = f'{metric} ({unit})'
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold', loc='left')
        ax.set_ylim(y_min, y_max + (y_max-y_min)*0.15)  # 留出数值显示空间
        ax.tick_params(axis='x', rotation=15)
        
        # 简洁的边框样式
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 添加行标题（放在图表外侧）
    fig.text(0.5, 0.95, '逐点功率预测指标 (评估每个时刻瞬时功率的预测精度)', 
             ha='center', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.47, '航次总能耗指标 (评估整个航次能耗的预测精度)', 
             ha='center', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[INFO] 模型对比图已保存: {save_path}")


def plot_energy_comparison(predictions_dict, time_interval=1.0, save_path='result/scatter_comparison.png'):
    """
    绘制散点图对比（同时展示逐点功率和航次能耗）
    
    布局：每行3个模型，每组模型（3个）占两行（功率+能耗）
    - 第1行：模型1-3的逐点功率散点图
    - 第2行：模型1-3的航次能耗散点图
    - 第3行：模型4-6的逐点功率散点图
    - 第4行：模型4-6的航次能耗散点图
    """
    if not predictions_dict:
        return
    
    n_models = len(predictions_dict)
    n_cols = 3  # 每行3个模型
    n_model_groups = (n_models + n_cols - 1) // n_cols  # 模型分组数（每组3个模型）
    
    # 总行数 = 模型组数 × 2（每组占两行：功率+能耗）
    n_rows = n_model_groups * 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    
    # 确保axes是2D数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = {'LSTM': '#4A90D9', 'GRU': '#7BC47F', 'Bi-LSTM': '#F5A962', 'Transformer': '#E57373',
              'LSTM-Transformer': '#9C27B0', 'LSTM-FC': '#00BCD4', 'TCN': '#FF8C00', 'Informer': '#20B2AA'}

    model_items = list(predictions_dict.items())
    
    for idx, (model_name, (preds, trues)) in enumerate(model_items):
        color = colors.get(model_name, '#4A90D9')
        
        # 计算当前模型在网格中的位置
        model_group = idx // n_cols  # 模型所在的组（0或1）
        model_col = idx % n_cols     # 模型所在的列（0、1、2）
        
        # 功率行 = 组号 × 2，能耗行 = 组号 × 2 + 1
        power_row = model_group * 2
        energy_row = model_group * 2 + 1
        
        # ===== 功率散点图 =====
        ax_power = axes[power_row, model_col]
        
        all_pred = np.concatenate(preds)
        all_true = np.concatenate(trues)
        
        # 计算逐点功率指标
        power_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
        power_mae = mean_absolute_error(all_true, all_pred)
        power_r2 = r2_score(all_true, all_pred)
        
        # 随机采样以避免过多点
        n_points = len(all_true)
        if n_points > 5000:
            sample_idx = np.random.choice(n_points, 5000, replace=False)
            plot_true = all_true[sample_idx]
            plot_pred = all_pred[sample_idx]
        else:
            plot_true = all_true
            plot_pred = all_pred
        
        ax_power.scatter(plot_true, plot_pred, c=color, alpha=0.3, s=6, edgecolors='none')
        
        # 对角线
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        ax_power.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=1.5)
        
        # 添加指标
        textstr = f'$R^2$ = {power_r2:.4f}\nRMSE = {power_rmse:.1f} W\nMAE = {power_mae:.1f} W'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax_power.text(0.05, 0.95, textstr, transform=ax_power.transAxes, fontsize=11,
                     verticalalignment='top', horizontalalignment='left', bbox=props)
        
        ax_power.set_xlabel('实际功率 (W)', fontsize=12)
        ax_power.set_ylabel('预测功率 (W)', fontsize=12)
        ax_power.set_title(f'({chr(97+idx)}) {model_name}', fontsize=13, fontweight='bold')
        ax_power.grid(True, alpha=0.3)
        
        # ===== 能耗散点图 =====
        ax_energy = axes[energy_row, model_col]
        
        # 计算每个航次的总能耗 (Wh)
        true_energies = []
        pred_energies = []
        for pred_seq, true_seq in zip(preds, trues):
            true_energy = np.sum(true_seq) * (time_interval / 3600.0)
            pred_energy = np.sum(pred_seq) * (time_interval / 3600.0)
            true_energies.append(true_energy)
            pred_energies.append(pred_energy)
        
        true_energies = np.array(true_energies)
        pred_energies = np.array(pred_energies)
        
        # 过滤掉实际能耗大于425 Wh的异常航次
        valid_mask = true_energies <= 425
        true_energies = true_energies[valid_mask]
        pred_energies = pred_energies[valid_mask]
        
        # 计算能耗预测指标
        energy_rmse = np.sqrt(mean_squared_error(true_energies, pred_energies))
        energy_mae = mean_absolute_error(true_energies, pred_energies)
        energy_r2 = r2_score(true_energies, pred_energies)
        
        ax_energy.scatter(true_energies, pred_energies, c=color, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
        
        # 对角线
        min_val = min(true_energies.min(), pred_energies.min())
        max_val = max(true_energies.max(), pred_energies.max())
        ax_energy.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=1.5)
        
        # 添加指标
        textstr = f'$R^2$ = {energy_r2:.4f}\nRMSE = {energy_rmse:.2f} Wh\nMAE = {energy_mae:.2f} Wh'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax_energy.text(0.05, 0.95, textstr, transform=ax_energy.transAxes, fontsize=11,
                      verticalalignment='top', horizontalalignment='left', bbox=props)
        
        ax_energy.set_xlabel('实际能耗 (Wh)', fontsize=12)
        ax_energy.set_ylabel('预测能耗 (Wh)', fontsize=12)
        ax_energy.set_title(f'({chr(103+idx)}) {model_name}', fontsize=13, fontweight='bold')
        ax_energy.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for group in range(n_model_groups):
        for col in range(n_cols):
            model_idx = group * n_cols + col
            if model_idx >= n_models:
                power_row = group * 2
                energy_row = group * 2 + 1
                axes[power_row, col].set_visible(False)
                axes[energy_row, col].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[INFO] 散点图对比已保存: {save_path}")


def plot_sequence_samples(predictions_dict, targets, order_ids, n_samples=3, save_path='result/sequence_samples.png'):
    """
    绘制多个航次的序列预测对比图
    
    布局：每行3个模型，每个航次占两行（模型1-3和模型4-6）
    不显示航次标签，指标标注在底部中间位置（两行排版），图例在右下角
    """
    if not predictions_dict:
        return
    
    n_models = len(predictions_dict)
    n_cols = 3  # 每行3个模型
    n_model_rows = (n_models + n_cols - 1) // n_cols  # 每个航次需要的行数
    
    # 总行数 = 航次数 × 每个航次的行数
    n_rows = n_samples * n_model_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    
    # 确保axes是2D数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = {'LSTM': '#4A90D9', 'GRU': '#7BC47F', 'Bi-LSTM': '#F5A962', 'Transformer': '#E57373',
              'LSTM-Transformer': '#9C27B0', 'LSTM-FC': '#00BCD4', 'TCN': '#FF8C00', 'Informer': '#20B2AA'}
    
    # 选择指定的航次：第1、4、5个（索引0、3、4）
    sample_indices = [0, 3, 4][:min(n_samples, len(targets))]
    
    model_items = list(predictions_dict.items())
    
    for sample_row, sample_idx in enumerate(sample_indices):
        for model_idx, (model_name, (preds, trues)) in enumerate(model_items):
            # 计算当前模型在网格中的位置
            model_row_offset = model_idx // n_cols  # 模型在当前航次中的行偏移（0或1）
            model_col = model_idx % n_cols         # 模型所在的列（0、1、2）
            
            # 实际行号 = 航次起始行 + 模型行偏移
            actual_row = sample_row * n_model_rows + model_row_offset
            
            ax = axes[actual_row, model_col]
            
            true_power = trues[sample_idx]
            pred_power = preds[sample_idx]
            time_axis = np.arange(len(true_power))
            
            # 计算该航次的指标
            seq_rmse = np.sqrt(mean_squared_error(true_power, pred_power))
            seq_r2 = r2_score(true_power, pred_power)
            
            color = colors.get(model_name, '#4A90D9')
            
            ax.plot(time_axis, true_power, 'k-', label='实际值', linewidth=1.2, alpha=0.8)
            ax.plot(time_axis, pred_power, color=color, linestyle='--', label='预测值', linewidth=1.2, alpha=0.8)
            ax.fill_between(time_axis, true_power, pred_power, alpha=0.15, color=color)
            
            # 添加指标标注在底部中间位置（两行排版）
            textstr = f'RMSE={seq_rmse:.1f}W  $R^2$={seq_r2:.3f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.4, 0.08, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='bottom', horizontalalignment='center', bbox=props)
            
            ax.set_xlabel('时间 (s)', fontsize=12)
            ax.set_ylabel('功率 (W)', fontsize=12)
            ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
            
            # 图例保持在右下角
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for sample_row in range(n_samples):
        for model_row_offset in range(n_model_rows):
            for col in range(n_cols):
                model_idx = model_row_offset * n_cols + col
                if model_idx >= n_models:
                    actual_row = sample_row * n_model_rows + model_row_offset
                    axes[actual_row, col].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[INFO] 序列预测样本图已保存: {save_path}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("="*60)
    print("时序功率预测模型统一评估")
    print("="*60)
    
    np.random.seed(42)
    
    # ===== 1. 加载模型 =====
    evaluator = ModelEvaluator()
    evaluator.load_models()
    
    if not evaluator.models:
        print("[ERROR] 没有可用的模型，请先训练模型")
        return
    
    # ===== 2. 加载测试数据 =====
    sequences, targets, order_ids = evaluator.load_test_data()
    
    print(f"[INFO] 测试集: {len(sequences)} 个航次")
    
    # ===== 3. 评估所有模型 =====
    results, predictions_dict = evaluator.evaluate_all(sequences, targets)
    
    # ===== 4. 打印结果 =====
    print("\n" + "="*60)
    print("评估结果汇总")
    print("="*60)
    
    results_list = []
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['RMSE']:.4f} W")
        print(f"  MAE:  {metrics['MAE']:.4f} W")
        print(f"  R2:   {metrics['R2']:.4f}")
        
        results_list.append({
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R2': metrics['R2']
        })
    
    # ===== 5. 可视化 =====
    print("\n[INFO] 生成可视化图表...")

    # 绘图时去掉GRU和Bi-LSTM，保持排版整洁（CSV中仍保留完整结果）
    _exclude = {'GRU', 'Bi-LSTM'}
    plot_results = {k: v for k, v in results.items() if k not in _exclude}
    plot_predictions = {k: v for k, v in predictions_dict.items() if k not in _exclude}

    # 模型对比柱状图（包含逐点功率和航次能耗两类指标）
    plot_metrics_comparison(plot_results, plot_predictions, time_interval=1.0,
                           save_path='result/models_comparison.png')

    if plot_predictions:
        # 散点图对比（逐点功率+航次能耗）
        plot_energy_comparison(plot_predictions, time_interval=1.0,
                              save_path='result/scatter_comparison.png')

        # 多航次序列预测样本图（展示第1、4、5个航次的功率曲线对比）
        plot_sequence_samples(plot_predictions, targets, order_ids, n_samples=3,
                             save_path='result/power_curves_comparison.png')
    
    # ===== 6. 保存结果 =====
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('result/all_models_evaluation.csv', index=False)
    print(f"[INFO] 评估结果已保存: result/all_models_evaluation.csv")
    
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    print("生成的文件:")
    print("  - result/models_comparison.png: 模型指标对比图（逐点功率+航次能耗）")
    print("  - result/scatter_comparison.png: 散点图对比（逐点功率+航次能耗）")
    print("  - result/power_curves_comparison.png: 多航次功率曲线对比图")
    print("  - result/all_models_evaluation.csv: 评估结果汇总")


if __name__ == "__main__":
    main()
