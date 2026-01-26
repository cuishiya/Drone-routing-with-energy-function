#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一评估所有时序功率预测模型

支持的模型:
- LSTM Seq2Seq
- GRU Seq2Seq
- Bi-LSTM
- Transformer

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

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {device}")


# ==================== 模型定义 ====================

# LSTM Seq2Seq
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size=1, num_layers=2, dropout=0.2, bidirectional=True):
        super(LSTMDecoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=hidden_size * num_directions, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * num_directions, output_size)
    
    def forward(self, encoder_outputs, hidden, cell):
        decoder_outputs, _ = self.lstm(encoder_outputs, (hidden, cell))
        return self.fc(decoder_outputs).squeeze(-1)


class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=True):
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.decoder = LSTMDecoder(hidden_size, 1, num_layers, dropout, bidirectional)
    
    def forward(self, x):
        encoder_outputs, hidden, cell = self.encoder(x)
        return self.decoder(encoder_outputs, hidden, cell)


# GRU Seq2Seq
class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
    
    def forward(self, x):
        outputs, hidden = self.gru(x)
        return outputs, hidden


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size=1, num_layers=2, dropout=0.2, bidirectional=True):
        super(GRUDecoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(input_size=hidden_size * num_directions, hidden_size=hidden_size, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * num_directions, output_size)
    
    def forward(self, encoder_outputs, hidden):
        decoder_outputs, _ = self.gru(encoder_outputs, hidden)
        return self.fc(decoder_outputs).squeeze(-1)


class GRUSeq2Seq(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=True):
        super(GRUSeq2Seq, self).__init__()
        self.encoder = GRUEncoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.decoder = GRUDecoder(hidden_size, 1, num_layers, dropout, bidirectional)
    
    def forward(self, x):
        encoder_outputs, hidden = self.encoder(x)
        return self.decoder(encoder_outputs, hidden)


# Bi-LSTM
class BiLSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=3, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        # 注意力机制（与训练脚本保持一致）
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
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
    def __init__(self, input_size=7, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, max_len=500):
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


# ==================== 模型评估器 ====================

class ModelEvaluator:
    """统一模型评估器"""
    
    def __init__(self):
        self.device = device
        self.feature_cols = ['Height', 'VS (m/s)', 'GS (m/s)', 'Wind Speed', 
                            'Temperature', 'Humidity', 'wind_angle']
        
        # 模型配置（统一参数：hidden_size=256, num_layers=3, dropout=0.2）
        self.model_configs = {
            'LSTM': {
                'path': 'result/power_lstm_seq2seq_model.pth',
                'class': LSTMSeq2Seq,
                'params': {'input_size': 7, 'hidden_size': 256, 'num_layers': 3, 'dropout': 0.2, 'bidirectional': True}
            },
            'GRU': {
                'path': 'result/power_gru_model.pth',
                'class': GRUSeq2Seq,
                'params': {'input_size': 7, 'hidden_size': 256, 'num_layers': 3, 'dropout': 0.2, 'bidirectional': True}
            },
            'Bi-LSTM': {
                'path': 'result/power_bilstm_v2_model.pth',
                'class': BiLSTMModel,
                'params': {'input_size': 7, 'hidden_size': 256, 'num_layers': 3, 'dropout': 0.2}
            },
            'Transformer': {
                'path': 'result/power_transformer_model.pth',
                'class': TransformerModel,
                'params': {'input_size': 7, 'd_model': 256, 'nhead': 8, 'num_layers': 3, 'dim_feedforward': 1024, 'dropout': 0.2, 'max_len': 500}
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
        """加载测试数据"""
        if test_path is None:
            test_path = "Drone_energy_dataset/test_data/flightTrajectory.xlsx"
        
        print(f"[INFO] 加载测试数据: {test_path}")
        df = pd.read_excel(test_path)
        
        # 预处理
        df['Power'] = (df['Voltage'] / 1000.0) * (df['Current'] / 1000.0)
        df['wind_angle'] = np.abs(df['Wind Direct'] - df['Course'])
        df['wind_angle'] = df['wind_angle'].apply(lambda x: x if x <= 180 else 360 - x)
        
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
    # 使用参考图片的配色风格
    colors = ['#7BC47F', '#4A90D9', '#F5A962', '#E57373']  # 绿、蓝、橙、红
    
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
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10, color='black')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (y_max-y_min)*0.02, 
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10, color='black')
        
        # 使用LaTeX格式显示R²
        if metric == 'R2':
            ylabel = r'$R^2$ Score'
        else:
            ylabel = f'{metric} ({unit})'
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold', loc='left')
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
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10, color='black')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (y_max-y_min)*0.02, 
                       f'{val:.2f}', ha='center', va='bottom', fontsize=10, color='black')
        
        # 使用LaTeX格式显示R²
        if metric == 'R2':
            ylabel = r'$R^2$ Score'
        else:
            ylabel = f'{metric} ({unit})'
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold', loc='left')
        ax.set_ylim(y_min, y_max + (y_max-y_min)*0.15)  # 留出数值显示空间
        ax.tick_params(axis='x', rotation=15)
        
        # 简洁的边框样式
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 添加行标题（放在图表外侧）
    fig.text(0.5, 0.95, '逐点功率预测指标 (评估每个时刻瞬时功率的预测精度)', 
             ha='center', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.47, '航次总能耗指标 (评估整个航次能耗的预测精度)', 
             ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[INFO] 模型对比图已保存: {save_path}")


def plot_energy_comparison(predictions_dict, time_interval=1.0, save_path='result/scatter_comparison.png'):
    """
    绘制散点图对比（同时展示逐点功率和航次能耗）
    
    上排：逐点功率预测散点图（每个点是一个时刻的功率）
    下排：航次能耗预测散点图（每个点是一个完整航次的总能耗）
    """
    if not predictions_dict:
        return
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(2, n_models, figsize=(4.5 * n_models, 9))
    
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    colors = {'LSTM': '#4A90D9', 'GRU': '#7BC47F', 'Bi-LSTM': '#F5A962', 'Transformer': '#E57373'}
    
    for idx, (model_name, (preds, trues)) in enumerate(predictions_dict.items()):
        color = colors.get(model_name, '#4A90D9')
        
        # ===== 上排：逐点功率散点图 =====
        ax_power = axes[0, idx]
        
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
        ax_power.text(0.05, 0.95, textstr, transform=ax_power.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='left', bbox=props)
        
        ax_power.set_xlabel('Actual Power (W)', fontsize=10)
        ax_power.set_ylabel('Predicted Power (W)', fontsize=10)
        ax_power.set_title(f'({chr(97+idx)}) {model_name}', fontsize=11, fontweight='bold')
        ax_power.grid(True, alpha=0.3)
        
        # ===== 下排：航次能耗散点图 =====
        ax_energy = axes[1, idx]
        
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
        ax_energy.text(0.05, 0.95, textstr, transform=ax_energy.transAxes, fontsize=9,
                      verticalalignment='top', horizontalalignment='left', bbox=props)
        
        ax_energy.set_xlabel('Actual Energy (Wh)', fontsize=10)
        ax_energy.set_ylabel('Predicted Energy (Wh)', fontsize=10)
        ax_energy.set_title(f'({chr(101+idx)})', fontsize=11, fontweight='bold')
        ax_energy.grid(True, alpha=0.3)
    
    # 添加行标题
    fig.text(0.5, 0.97, '逐点功率预测 (每个点=一个时刻的瞬时功率)', ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    fig.text(0.5, 0.48, '航次总能耗预测 (每个点=一个完整航次的总能耗)', ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.35)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[INFO] 散点图对比已保存: {save_path}")


def plot_sequence_samples(predictions_dict, targets, order_ids, n_samples=3, save_path='result/sequence_samples.png'):
    """绘制多个航次的序列预测对比图"""
    if not predictions_dict:
        return
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(n_samples, n_models, figsize=(4.5 * n_models, 3.5 * n_samples))
    
    if n_models == 1:
        axes = axes.reshape(n_samples, 1)
    if n_samples == 1:
        axes = axes.reshape(1, n_models)
    
    colors = {'LSTM': '#4A90D9', 'GRU': '#7BC47F', 'Bi-LSTM': '#F5A962', 'Transformer': '#E57373'}
    
    # 选择不同长度的航次样本
    seq_lengths = [len(t) for t in targets]
    sorted_indices = np.argsort(seq_lengths)
    sample_indices = [
        sorted_indices[len(sorted_indices) // 4],      # 较短
        sorted_indices[len(sorted_indices) // 2],      # 中等
        sorted_indices[6 * len(sorted_indices) // 10]   # 较长
    ][:n_samples]
    
    for row, sample_idx in enumerate(sample_indices):
        for col, (model_name, (preds, trues)) in enumerate(predictions_dict.items()):
            ax = axes[row, col]
            
            true_power = trues[sample_idx]
            pred_power = preds[sample_idx]
            time_axis = np.arange(len(true_power))
            
            # 计算该航次的指标
            seq_rmse = np.sqrt(mean_squared_error(true_power, pred_power))
            seq_r2 = r2_score(true_power, pred_power)
            
            color = colors.get(model_name, '#4A90D9')
            
            ax.plot(time_axis, true_power, 'k-', label='Actual', linewidth=1.2, alpha=0.8)
            ax.plot(time_axis, pred_power, color=color, linestyle='--', label='Predicted', linewidth=1.2, alpha=0.8)
            ax.fill_between(time_axis, true_power, pred_power, alpha=0.15, color=color)
            
            # 添加指标
            textstr = f'RMSE={seq_rmse:.1f}W\n$R^2$={seq_r2:.3f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right', bbox=props)
            
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Power (W)', fontsize=10)
            
            if row == 0:
                ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
            
            if col == 0:
                ax.text(-0.15, 0.5, f'Flight {row+1}\n({len(true_power)}s)', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='center',
                       horizontalalignment='right', fontweight='bold')
            
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
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
    
    # 模型对比柱状图（包含逐点功率和航次能耗两类指标）
    plot_metrics_comparison(results, predictions_dict, time_interval=1.0, 
                           save_path='result/models_comparison.png')
    
    if predictions_dict:
        # 散点图对比（逐点功率+航次能耗）
        plot_energy_comparison(predictions_dict, time_interval=1.0, 
                              save_path='result/scatter_comparison.png')
        
        # 多航次序列预测样本图（展示不同航次的功率曲线对比）
        plot_sequence_samples(predictions_dict, targets, order_ids, n_samples=3,
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
