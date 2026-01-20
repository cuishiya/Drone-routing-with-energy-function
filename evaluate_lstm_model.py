#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM Seq2Seq 瞬时功率预测模型评估脚本

评估指标:
- RMSE: 均方根误差
- MAE: 平均绝对误差
- R²: 决定系数

可视化:
- 功率曲线对比图（真实值 vs 预测值）
- 预测散点图
- 误差分布图
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

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {device}")


# ==================== LSTM Seq2Seq 模型定义 ====================

class LSTMEncoder(nn.Module):
    """LSTM编码器"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, bidirectional=True):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class LSTMDecoder(nn.Module):
    """LSTM解码器"""
    def __init__(self, hidden_size, output_size=1, num_layers=2, dropout=0.2, bidirectional=True):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        decoder_input_size = hidden_size * self.num_directions
        
        self.lstm = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, encoder_outputs, hidden, cell):
        decoder_outputs, _ = self.lstm(encoder_outputs, (hidden, cell))
        outputs = self.fc(decoder_outputs)
        return outputs.squeeze(-1)


class LSTMSeq2Seq(nn.Module):
    """LSTM Seq2Seq 完整模型"""
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, 
                 dropout=0.2, bidirectional=True):
        super(LSTMSeq2Seq, self).__init__()
        
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.decoder = LSTMDecoder(
            hidden_size=hidden_size,
            output_size=1,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    
    def forward(self, x):
        encoder_outputs, hidden, cell = self.encoder(x)
        outputs = self.decoder(encoder_outputs, hidden, cell)
        return outputs


# ==================== 模型评估器 ====================

class LSTMModelEvaluator:
    """LSTM模型评估器"""
    
    def __init__(self, model_path='result/power_lstm_seq2seq_model.pth'):
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.device = device
        
        # 特征列名
        self.feature_cols = ['Height', 'VS (m/s)', 'GS (m/s)', 'Wind Speed', 
                            'Temperature', 'Humidity', 'wind_angle']
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        print(f"[INFO] 加载模型: {self.model_path}")
        
        # 创建模型结构
        self.model = LSTMSeq2Seq(
            input_size=7,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=True
        )
        
        # 加载模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 加载标准化器
        scaler_path = self.model_path.replace('.pth', '_scalers.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.feature_scaler = scalers.get('feature_scaler')
                self.target_scaler = scalers.get('target_scaler')
        
        print("[OK] 模型加载成功")
    
    def load_test_data(self, test_path=None, min_seq_len=20, max_seq_len=500):
        """加载测试数据"""
        if test_path is None:
            test_path = "Drone_energy_dataset/test_data/flightTrajectory.xlsx"
        
        print(f"[INFO] 加载测试数据: {test_path}")
        combined_df = pd.read_excel(test_path)
        
        # 预处理
        combined_df['Power'] = (combined_df['Voltage'] / 1000.0) * (combined_df['Current'] / 1000.0)
        combined_df['wind_angle'] = np.abs(combined_df['Wind Direct'] - combined_df['Course'])
        combined_df['wind_angle'] = combined_df['wind_angle'].apply(lambda x: x if x <= 180 else 360 - x)
        
        # 过滤
        combined_df = combined_df.dropna(subset=self.feature_cols + ['Power'])
        combined_df = combined_df[combined_df['Power'] > 0]
        combined_df = combined_df[combined_df['Power'] < 15000]
        
        # 按航次分组创建序列
        sequences = []
        targets = []
        order_ids = []
        
        grouped = combined_df.groupby('Order ID')
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
    
    def predict_sequence(self, feature_sequence):
        """预测单个序列"""
        # 标准化
        if self.feature_scaler is not None:
            feature_sequence = self.feature_scaler.transform(feature_sequence)
        
        # 转换为张量
        with torch.no_grad():
            input_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            power_sequence = output.squeeze(0).cpu().numpy()
        
        # 反标准化
        if self.target_scaler is not None:
            power_sequence = self.target_scaler.inverse_transform(
                power_sequence.reshape(-1, 1)
            ).flatten()
        
        return np.maximum(0.0, power_sequence)
    
    def evaluate(self, sequences, targets):
        """评估模型"""
        all_predictions = []
        all_targets = []
        
        print("[INFO] 开始评估...")
        for i, (seq, tgt) in enumerate(zip(sequences, targets)):
            pred = self.predict_sequence(seq)
            all_predictions.append(pred)
            all_targets.append(tgt)
            
            if (i + 1) % 100 == 0:
                print(f"  已评估 {i+1}/{len(sequences)} 个航次")
        
        # 展平计算整体指标
        all_pred_flat = np.concatenate(all_predictions)
        all_true_flat = np.concatenate(all_targets)
        
        rmse = np.sqrt(mean_squared_error(all_true_flat, all_pred_flat))
        mae = mean_absolute_error(all_true_flat, all_pred_flat)
        r2 = r2_score(all_true_flat, all_pred_flat)
        
        # 计算MAPE
        mask = all_true_flat != 0
        mape = np.mean(np.abs((all_true_flat[mask] - all_pred_flat[mask]) / all_true_flat[mask])) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics, all_predictions, all_targets


# ==================== 可视化函数 ====================

def plot_power_curve(true_power, pred_power, order_id=None, save_path='result/lstm_power_curve.png'):
    """
    绘制功率曲线对比图
    
    参数:
        true_power: 真实功率序列
        pred_power: 预测功率序列
        order_id: 航次ID
        save_path: 保存路径
    """
    # 计算该序列的指标
    rmse = np.sqrt(mean_squared_error(true_power, pred_power))
    mae = mean_absolute_error(true_power, pred_power)
    r2 = r2_score(true_power, pred_power)
    
    # 创建时间轴（假设采样间隔1秒）
    time_axis = np.arange(len(true_power))
    
    # 设置图形
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制曲线
    ax.plot(time_axis, true_power, 'b-', label='真实功率', linewidth=1.5, alpha=0.8)
    ax.plot(time_axis, pred_power, 'r--', label='预测功率', linewidth=1.5, alpha=0.8)
    
    # 填充误差区域
    ax.fill_between(time_axis, true_power, pred_power, alpha=0.2, color='gray', label='预测误差')
    
    # 设置标签
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('功率 (W)', fontsize=12)
    
    title = 'LSTM Seq2Seq 功率预测曲线对比'
    if order_id:
        title += f'\n航次: {order_id[:30]}...' if len(str(order_id)) > 30 else f'\n航次: {order_id}'
    ax.set_title(title, fontsize=14)
    
    # 添加指标文本框
    textstr = f'RMSE = {rmse:.2f} W\nMAE = {mae:.2f} W\nR² = {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # 图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 网格
    ax.grid(True, alpha=0.3)
    
    # 设置y轴范围
    y_min = min(true_power.min(), pred_power.min()) * 0.9
    y_max = max(true_power.max(), pred_power.max()) * 1.1
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 功率曲线图已保存: {save_path}")


def plot_scatter(all_predictions, all_targets, save_path='result/lstm_evaluation_scatter.png'):
    """绘制预测散点图"""
    all_pred = np.concatenate(all_predictions)
    all_true = np.concatenate(all_targets)
    
    r2 = r2_score(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 散点图（使用密度着色）
    from matplotlib.colors import LogNorm
    h = ax.hist2d(all_true, all_pred, bins=100, cmap='Blues', norm=LogNorm())
    plt.colorbar(h[3], ax=ax, label='样本数量')
    
    # 对角线
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
    
    ax.set_xlabel('真实功率 (W)', fontsize=12)
    ax.set_ylabel('预测功率 (W)', fontsize=12)
    ax.set_title(f'LSTM Seq2Seq 预测散点图\nR² = {r2:.4f}, RMSE = {rmse:.2f} W', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 散点图已保存: {save_path}")


def plot_error_distribution(all_predictions, all_targets, save_path='result/lstm_error_distribution.png'):
    """绘制误差分布图"""
    all_pred = np.concatenate(all_predictions)
    all_true = np.concatenate(all_targets)
    
    errors = all_pred - all_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 误差直方图
    ax1 = axes[0]
    ax1.hist(errors, bins=100, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差')
    ax1.axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2, 
                label=f'平均误差: {np.mean(errors):.2f} W')
    ax1.set_xlabel('预测误差 (W)', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('预测误差分布', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 相对误差直方图
    ax2 = axes[1]
    mask = all_true != 0
    relative_errors = (all_pred[mask] - all_true[mask]) / all_true[mask] * 100
    ax2.hist(relative_errors, bins=100, color='coral', edgecolor='white', alpha=0.7, 
             range=(-50, 50))
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差')
    ax2.axvline(x=np.mean(relative_errors), color='orange', linestyle='-', linewidth=2,
                label=f'平均: {np.mean(relative_errors):.2f}%')
    ax2.set_xlabel('相对误差 (%)', fontsize=12)
    ax2.set_ylabel('频数', fontsize=12)
    ax2.set_title('相对误差分布', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 误差分布图已保存: {save_path}")


def plot_metrics_summary(metrics, save_path='result/lstm_metrics_summary.png'):
    """绘制指标汇总图"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    metric_names = ['RMSE', 'MAE', 'R2', 'MAPE']
    metric_values = [metrics['RMSE'], metrics['MAE'], metrics['R2'], metrics['MAPE']]
    metric_units = ['W', 'W', '', '%']
    colors = ['#4A90D9', '#7BC47F', '#F5A962', '#E57373']
    
    for i, (name, value, unit, color) in enumerate(zip(metric_names, metric_values, metric_units, colors)):
        ax = axes[i]
        ax.bar([name], [value], color=color, edgecolor='black', linewidth=1.5)
        
        # 在柱子上方显示数值
        if name == 'R2':
            ax.text(0, value + 0.02, f'{value:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        else:
            ax.text(0, value + value*0.02, f'{value:.2f}{unit}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylabel(f'{name} ({unit})' if unit else name, fontsize=12)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 设置y轴范围
        if name == 'R2':
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, value * 1.2)
    
    plt.suptitle('LSTM Seq2Seq 模型评估指标', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 指标汇总图已保存: {save_path}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("="*60)
    print("LSTM Seq2Seq 瞬时功率预测模型评估")
    print("="*60)
    
    # 设置随机种子
    np.random.seed(42)
    
    # ===== 1. 加载模型 =====
    evaluator = LSTMModelEvaluator(model_path='result/power_lstm_seq2seq_model.pth')
    
    # ===== 2. 加载测试数据 =====
    test_sequences, test_targets, test_order_ids = evaluator.load_test_data()
    
    print(f"[INFO] 测试集: {len(test_sequences)} 个航次")
    
    # ===== 3. 评估模型 =====
    metrics, predictions, true_values = evaluator.evaluate(test_sequences, test_targets)
    
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"  RMSE: {metrics['RMSE']:.4f} W")
    print(f"  MAE:  {metrics['MAE']:.4f} W")
    print(f"  R2:   {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # ===== 4. 可视化 =====
    print("\n[INFO] 生成可视化图表...")
    
    # 4.1 选择一个航次绘制功率曲线对比
    # 选择一个中等长度的航次
    seq_lengths = [len(s) for s in test_sequences]
    median_len = np.median(seq_lengths)
    best_idx = np.argmin([abs(len(s) - median_len) for s in test_sequences])
    
    plot_power_curve(
        true_power=true_values[best_idx],
        pred_power=predictions[best_idx],
        order_id=test_order_ids[best_idx],
        save_path='result/lstm_power_curve.png'
    )
    
    # 4.2 绘制散点图
    plot_scatter(predictions, true_values, save_path='result/lstm_evaluation_scatter.png')
    
    # 4.3 绘制误差分布图
    plot_error_distribution(predictions, true_values, save_path='result/lstm_error_distribution.png')
    
    # 4.4 绘制指标汇总图
    plot_metrics_summary(metrics, save_path='result/lstm_metrics_summary.png')
    
    # ===== 5. 保存评估结果 =====
    results_df = pd.DataFrame([metrics])
    results_df.to_csv('result/lstm_evaluation_results.csv', index=False)
    print(f"[INFO] 评估结果已保存: result/lstm_evaluation_results.csv")
    
    # 保存详细预测结果
    detailed_results = []
    for i, (pred, true, oid) in enumerate(zip(predictions, true_values, test_order_ids)):
        seq_rmse = np.sqrt(mean_squared_error(true, pred))
        seq_mae = mean_absolute_error(true, pred)
        seq_r2 = r2_score(true, pred)
        detailed_results.append({
            'Order_ID': oid,
            'Sequence_Length': len(pred),
            'RMSE': seq_rmse,
            'MAE': seq_mae,
            'R2': seq_r2
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('result/lstm_detailed_evaluation.csv', index=False)
    print(f"[INFO] 详细评估结果已保存: result/lstm_detailed_evaluation.csv")
    
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    print("生成的文件:")
    print("  - result/lstm_power_curve.png: 功率曲线对比图")
    print("  - result/lstm_evaluation_scatter.png: 预测散点图")
    print("  - result/lstm_error_distribution.png: 误差分布图")
    print("  - result/lstm_metrics_summary.png: 指标汇总图")
    print("  - result/lstm_evaluation_results.csv: 评估指标")
    print("  - result/lstm_detailed_evaluation.csv: 各航次详细评估")


if __name__ == "__main__":
    main()
