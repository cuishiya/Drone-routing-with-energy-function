#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特征分组消融实验

目标：验证"载荷特征"与"气象特征"对功率预测精度的贡献度
模型：统一使用 LSTM-Transformer 融合模型

四组实验：
  1. 基础特征组   — 飞行状态（高度、竖直速度、水平速度、飞行朝向）
  2. 基础+任务属性 — 基础 + 载荷质量
  3. 基础+环境因素 — 基础 + 风速、温度、湿度
  4. 全部特征组   — 上述所有特征

输出文件：
  - result/ablation_feature_groups_comparison.png  对比图
  - result/ablation_feature_groups_results.csv     数值结果
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# 切换工作目录到项目根目录，确保相对路径正确
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {device}")


# ==================== 特征分组定义 ====================

# wind_angle = |Wind Direct - Course|，是风场（环境）特征，不属于纯飞行状态
# 基础组仅保留纯运动学特征：高度、竖直速度、水平速度
FEATURE_GROUPS = {
    '①基础特征组': {
        'features': ['Height', 'VS (m/s)', 'GS (m/s)'],
        'description': '飞行状态：高度、竖直速度、水平速度',
        'color': '#4A90D9'
    },
    '②基础+任务属性': {
        'features': ['Height', 'VS (m/s)', 'GS (m/s)', 'payload'],
        'description': '飞行状态 + 载荷质量',
        'color': '#7BC47F'
    },
    '③基础+环境因素': {
        'features': ['Height', 'VS (m/s)', 'GS (m/s)',
                     'Wind Speed', 'Temperature', 'Humidity', 'wind_angle'],
        'description': '飞行状态 + 风速、温度、湿度、风向夹角（完整气象）',
        'color': '#F5A962'
    },
    '④全部特征组': {
        'features': ['Height', 'VS (m/s)', 'GS (m/s)',
                     'payload', 'Wind Speed', 'Temperature', 'Humidity', 'wind_angle'],
        'description': '所有特征',
        'color': '#E57373'
    },
}

# 全量特征列（数据加载时需要全部读取，再按组切片）
ALL_FEATURE_COLS = ['Height', 'VS (m/s)', 'GS (m/s)', 'Wind Speed',
                    'Temperature', 'Humidity', 'wind_angle', 'payload']


# ==================== LSTM-Transformer 模型 ====================

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


class LSTMTransformerModel(nn.Module):
    """LSTM-Transformer 串行模型（输入维度由 input_size 动态控制）"""
    def __init__(self, input_size=8, d_model=256, lstm_layers=2,
                 nhead=8, num_transformer_layers=3, dim_feedforward=512,
                 dropout=0.1, max_len=500):
        super(LSTMTransformerModel, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_size, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model // 2,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0, bidirectional=True
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_embedding(x)
        x, _ = self.lstm(x)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_layer(x).squeeze(-1)


# ==================== 数据集 ====================

class FlightDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])


def collate_fn(batch):
    sequences, targets = zip(*batch)
    lengths = [len(s) for s in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    max_len = padded_sequences.size(1)
    padding_mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_mask[i, length:] = True
    return padded_sequences, padded_targets, padding_mask, lengths


# ==================== 数据加载（一次性，全量特征）====================

def load_full_data(data_dirs):
    """加载并预处理全量数据，返回包含所有特征的 DataFrame"""
    all_data = []
    for data_dir in data_dirs:
        trajectory_path = os.path.join(data_dir, 'flightTrajectory.xlsx')
        record_path = os.path.join(data_dir, 'flightRecord.xlsx')
        if not os.path.exists(trajectory_path):
            continue
        print(f"[INFO] 加载轨迹数据: {trajectory_path}")
        df = pd.read_excel(trajectory_path)
        if os.path.exists(record_path):
            record_df = pd.read_excel(record_path)
            if 'Payload (kg)' in record_df.columns:
                payload_map = record_df.set_index('Order ID')['Payload (kg)'].to_dict()
                df['payload'] = df['Order ID'].map(payload_map)
            else:
                df['payload'] = 0.0
        else:
            df['payload'] = 0.0
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    # 功率计算
    combined['Power'] = (combined['Voltage'] / 1000.0) * (combined['Current'] / 1000.0)

    # 风向夹角
    combined['wind_angle'] = np.abs(combined['Wind Direct'] - combined['Course'])
    combined['wind_angle'] = combined['wind_angle'].apply(lambda x: x if x <= 180 else 360 - x)

    # 填充载荷缺失值
    combined['payload'] = combined['payload'].fillna(0.0)

    # 过滤：保留所有特征和功率均有效的记录
    combined = combined.dropna(subset=ALL_FEATURE_COLS + ['Power'])
    combined = combined[(combined['Power'] > 0) & (combined['Power'] < 15000)]

    print(f"[INFO] 预处理后: {len(combined)} 条记录, {combined['Order ID'].nunique()} 个航次")
    return combined


def get_order_split(df, min_seq_len=20, test_size=0.2):
    """对 df 中有效航次做 80/20 随机划分（random_state=42），返回 (train_ids, val_ids)"""
    valid_orders = [
        oid for oid, g in df.groupby('Order ID') if len(g) >= min_seq_len
    ]
    train_ids, val_ids = train_test_split(valid_orders, test_size=test_size, random_state=42)
    print(f"[INFO] 80/20 划分 — 训练: {len(train_ids)} 个航次, 验证: {len(val_ids)} 个航次")
    return train_ids, val_ids


def make_seq_by_ids(df, order_ids, feature_cols, min_seq_len=20, max_seq_len=500):
    """按指定 order_ids 子集构建序列"""
    sequences, targets = [], []
    id_set = set(order_ids)
    for order_id, group in df.groupby('Order ID'):
        if order_id not in id_set:
            continue
        group = group.sort_values('Time Stamp')
        if len(group) < min_seq_len:
            continue
        if len(group) > max_seq_len:
            group = group.head(max_seq_len)
        sequences.append(group[feature_cols].values)
        targets.append(group['Power'].values)
    return sequences, targets


def make_all_sequences(df, feature_cols, min_seq_len=20, max_seq_len=500):
    """将 df 中所有符合条件的航次转换为序列（不做划分）"""
    sequences, targets = [], []
    for order_id, group in df.groupby('Order ID'):
        group = group.sort_values('Time Stamp')
        if len(group) < min_seq_len:
            continue
        if len(group) > max_seq_len:
            group = group.head(max_seq_len)
        sequences.append(group[feature_cols].values)
        targets.append(group['Power'].values)
    return sequences, targets


# ==================== 训练阶段 ====================

def train_group(group_name, feature_cols, train_seq, train_tgt, val_seq, val_tgt,
                save_dir='result'):
    """训练单组模型并保存权重和 Scaler，返回 (save_path, feature_scaler, target_scaler)"""
    print(f"\n{'='*55}")
    print(f"  [训练] {group_name}  ({len(feature_cols)} 个特征)")
    print(f"  特征: {feature_cols}")
    print('='*55)

    # 标准化（每组独立 Scaler，仅在训练集上 fit）
    feature_scaler = StandardScaler()
    target_scaler  = StandardScaler()
    feature_scaler.fit(np.vstack(train_seq))
    target_scaler.fit(np.concatenate(train_tgt).reshape(-1, 1))

    def scale(seqs, tgts):
        return (
            [feature_scaler.transform(s) for s in seqs],
            [target_scaler.transform(t.reshape(-1, 1)).flatten() for t in tgts]
        )

    train_seq_s, train_tgt_s = scale(train_seq, train_tgt)
    val_seq_s,   val_tgt_s   = scale(val_seq,   val_tgt)

    train_loader = DataLoader(FlightDataset(train_seq_s, train_tgt_s),
                              batch_size=32, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(FlightDataset(val_seq_s,   val_tgt_s),
                              batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = LSTMTransformerModel(
        input_size=len(feature_cols), d_model=256, lstm_layers=2,
        nhead=8, num_transformer_layers=3, dim_feedforward=512,
        dropout=0.1, max_len=500
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss(reduction='none')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'ablation_group{group_name[0]}.pth')

    for epoch in range(50):
        model.train()
        train_loss, train_n = 0.0, 0
        for batch_seq, batch_tgt, pad_mask, lengths in train_loader:
            batch_seq, batch_tgt, pad_mask = (
                batch_seq.to(device), batch_tgt.to(device), pad_mask.to(device))
            optimizer.zero_grad()
            out = model(batch_seq, src_key_padding_mask=pad_mask)
            valid = ~pad_mask
            loss  = (criterion(out, batch_tgt) * valid).sum() / valid.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * valid.sum().item()
            train_n    += valid.sum().item()

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for batch_seq, batch_tgt, pad_mask, lengths in val_loader:
                batch_seq, batch_tgt, pad_mask = (
                    batch_seq.to(device), batch_tgt.to(device), pad_mask.to(device))
                out   = model(batch_seq, src_key_padding_mask=pad_mask)
                valid = ~pad_mask
                val_loss += (criterion(out, batch_tgt) * valid).sum().item()
                val_n    += valid.sum().item()

        avg_val = val_loss / val_n
        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/50] train={train_loss/train_n:.6f}  val={avg_val:.6f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_scaler':   feature_scaler,
        'target_scaler':    target_scaler,
        'feature_cols':     feature_cols,
    }, save_path)
    print(f"  ✔ 训练完成（50 epoch），模型已保存: {save_path}")
    return save_path


# ==================== 评估阶段 ====================

def evaluate_group(group_name, save_path, test_seq, test_tgt):
    """加载已保存模型，在测试集上评估，返回 metrics dict"""
    print(f"  [评估] {group_name} ...")

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    feature_scaler = ckpt['feature_scaler']
    target_scaler  = ckpt['target_scaler']
    feature_cols   = ckpt['feature_cols']

    model = LSTMTransformerModel(
        input_size=len(feature_cols), d_model=256, lstm_layers=2,
        nhead=8, num_transformer_layers=3, dim_feedforward=512,
        dropout=0.1, max_len=500
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 标准化测试集（用训练集 fit 的 scaler）
    test_seq_s = [feature_scaler.transform(s) for s in test_seq]
    test_tgt_s = [target_scaler.transform(t.reshape(-1, 1)).flatten() for t in test_tgt]

    test_loader = DataLoader(FlightDataset(test_seq_s, test_tgt_s),
                             batch_size=32, shuffle=False, collate_fn=collate_fn)

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_seq, batch_tgt, pad_mask, lengths in test_loader:
            batch_seq, pad_mask = batch_seq.to(device), pad_mask.to(device)
            out = model(batch_seq, src_key_padding_mask=pad_mask)
            for i, length in enumerate(lengths):
                pred = target_scaler.inverse_transform(
                    out[i, :length].cpu().numpy().reshape(-1, 1)).flatten()
                true = target_scaler.inverse_transform(
                    batch_tgt[i, :length].numpy().reshape(-1, 1)).flatten()
                all_pred.extend(pred)
                all_true.extend(true)

    all_pred = np.maximum(np.array(all_pred), 0)
    all_true = np.array(all_true)

    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    mae  = mean_absolute_error(all_true, all_pred)
    r2   = r2_score(all_true, all_pred)
    mask = all_true > 0
    mape = np.mean(np.abs((all_true[mask] - all_pred[mask]) / all_true[mask])) * 100

    print(f"      RMSE={rmse:.2f}W  MAE={mae:.2f}W  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


# ==================== 可视化 ====================

def plot_ablation_results(results, save_path='result/ablation_feature_groups_comparison.png'):
    """绘制消融实验对比图：指标柱状图 + 贡献度分析"""
    groups  = list(results.keys())
    colors  = [FEATURE_GROUPS[g]['color'] for g in groups]
    rmse_vals = [results[g]['RMSE'] for g in groups]
    mae_vals  = [results[g]['MAE']  for g in groups]
    r2_vals   = [results[g]['R2']   for g in groups]

    # 贡献度（相对基础组 RMSE 的降幅百分比）
    base_rmse = rmse_vals[0]
    base_r2   = r2_vals[0]
    rmse_improve = [(base_rmse - v) / base_rmse * 100 for v in rmse_vals]
    r2_improve   = [(v - base_r2) / base_r2 * 100 for v in r2_vals]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    short_labels = ['①基础', '②+任务', '③+环境', '④全部']

    def bar_plot(ax, values, ylabel, title, fmt='.1f', unit=''):
        bars = ax.bar(short_labels, values, color=colors, edgecolor='none', width=0.55)
        val_range = max(values) - min(values)
        y_min = min(values) - val_range * 0.35
        y_max = max(values) + val_range * 0.35
        ax.set_ylim(y_min, y_max + val_range * 0.2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + val_range * 0.04,
                    f'{v:{fmt}}{unit}', ha='center', va='bottom', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    bar_plot(axes[0, 0], rmse_vals, 'RMSE (W)', '(a) RMSE 对比', fmt='.1f')
    bar_plot(axes[0, 1], mae_vals,  'MAE (W)',  '(b) MAE 对比',  fmt='.1f')
    bar_plot(axes[0, 2], r2_vals,   r'$R^2$ Score', '(c) R² 对比', fmt='.4f')

    # 第二行：贡献度分析（相对基础组的改善幅度）
    bar_plot(axes[1, 0], rmse_improve, 'RMSE 降幅 (%)', '(d) RMSE 相对基础组降幅', fmt='.1f', unit='%')
    bar_plot(axes[1, 1], r2_improve,   'R² 提升 (%)',   '(e) R² 相对基础组提升',   fmt='.1f', unit='%')

    # 第三子图：特征贡献对比（载荷 vs 气象，分别加入基础组后的提升）
    ax = axes[1, 2]
    payload_contrib = (rmse_vals[0] - rmse_vals[1]) / rmse_vals[0] * 100   # ①→②
    env_contrib     = (rmse_vals[0] - rmse_vals[2]) / rmse_vals[0] * 100   # ①→③
    both_contrib    = (rmse_vals[0] - rmse_vals[3]) / rmse_vals[0] * 100   # ①→④

    contrib_labels  = ['仅增加\n载荷特征', '仅增加\n气象特征', '同时增加\n载荷+气象']
    contrib_values  = [payload_contrib, env_contrib, both_contrib]
    contrib_colors  = ['#7BC47F', '#F5A962', '#E57373']

    bars = ax.bar(contrib_labels, contrib_values, color=contrib_colors, edgecolor='none', width=0.5)
    for bar, v in zip(bars, contrib_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('相对基础组 RMSE 降幅 (%)', fontsize=12, fontweight='bold')
    ax.set_title('(f) 各类特征对 RMSE 的独立贡献', fontsize=13, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    y_max_c = max(contrib_values)
    ax.set_ylim(0, y_max_c * 1.3)

    fig.text(0.5, 0.96, '特征分组消融实验 — LSTM-Transformer 模型', ha='center',
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.38)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[INFO] 消融对比图已保存: {save_path}")


def print_contribution_analysis(results):
    """终端打印贡献度分析"""
    base = results['①基础特征组']
    r_task = results['②基础+任务属性']
    r_env  = results['③基础+环境因素']
    r_all  = results['④全部特征组']

    print("\n" + "="*60)
    print("特征贡献度分析（相对基础特征组的提升）")
    print("="*60)

    def show(name, r):
        rmse_drop = (base['RMSE'] - r['RMSE']) / base['RMSE'] * 100
        mae_drop  = (base['MAE']  - r['MAE'])  / base['MAE']  * 100
        r2_gain   = (r['R2'] - base['R2']) / base['R2'] * 100
        print(f"\n  {name}:")
        print(f"    RMSE  {base['RMSE']:.2f} → {r['RMSE']:.2f} W  (↓{rmse_drop:.1f}%)")
        print(f"    MAE   {base['MAE']:.2f}  → {r['MAE']:.2f} W   (↓{mae_drop:.1f}%)")
        print(f"    R²    {base['R2']:.4f}  → {r['R2']:.4f}      (↑{r2_gain:.1f}%)")

    show("② 仅增加载荷特征（任务属性）", r_task)
    show("③ 仅增加气象特征（环境因素）", r_env)
    show("④ 同时增加载荷 + 气象特征",   r_all)

    # 两类特征的相对贡献（以全部特征组为基准）
    rmse_payload = (base['RMSE'] - r_task['RMSE']) / (base['RMSE'] - r_all['RMSE']) * 100
    rmse_env     = (base['RMSE'] - r_env['RMSE'])  / (base['RMSE'] - r_all['RMSE']) * 100
    print(f"\n  在全部精度提升中的贡献占比（以全部特征为100%基准）:")
    print(f"    载荷特征独立贡献占比: {rmse_payload:.1f}%")
    print(f"    气象特征独立贡献占比: {rmse_env:.1f}%")
    print("="*60)


# ==================== 主函数 ====================

def main():
    print("="*60)
    print("特征分组消融实验 — LSTM-Transformer")
    print("="*60)

    # ===== 1. 加载数据 =====
    print("[INFO] 加载训练数据（UAS04028624）...")
    train_df = load_full_data(['Drone_energy_dataset/UAS04028624'])
    print("[INFO] 加载测试数据（test_data，与训练完全独立）...")
    test_df  = load_full_data(['Drone_energy_dataset/test_data'])

    # ===== 2. 对 UAS04028624 做 80/20 随机划分（与原 LSTM-Transformer 完全一致）=====
    train_ids, val_ids = get_order_split(train_df)

    # ===== 3. 训练阶段：依次训练4组模型 =====
    print("\n" + "="*60)
    print("【训练阶段】依次训练 4 个特征组模型")
    print(f"  训练数据: UAS04028624 — 训练 {len(train_ids)} 个航次 / 验证 {len(val_ids)} 个航次")
    print(f"  验证集仅用于早停，最终测试集为独立的 test_data")
    print("="*60)

    save_paths = {}
    for group_name, group_cfg in FEATURE_GROUPS.items():
        feature_cols = group_cfg['features']
        train_seq, train_tgt = make_seq_by_ids(train_df, train_ids, feature_cols)
        val_seq,   val_tgt   = make_seq_by_ids(train_df, val_ids,   feature_cols)
        save_path = train_group(group_name, feature_cols,
                                train_seq, train_tgt,
                                val_seq,   val_tgt)
        save_paths[group_name] = save_path

    # ===== 4. 评估阶段：统一用 test_data 测试集评估所有已保存模型 =====
    print("\n" + "="*60)
    print("【评估阶段】在 test_data 测试集上评估所有特征组模型")
    print(f"  测试集: {test_df['Order ID'].nunique()} 个航次（完全独立数据，未参与任何训练）")
    print("="*60)

    results = {}
    for group_name, group_cfg in FEATURE_GROUPS.items():
        test_seq, test_tgt = make_all_sequences(test_df, group_cfg['features'])
        metrics = evaluate_group(group_name, save_paths[group_name], test_seq, test_tgt)
        results[group_name] = metrics

    # ===== 5. 汇总结果打印 =====
    print("\n" + "="*60)
    print("测试集评估结果汇总")
    print("="*60)
    print(f"{'特征组':<22} {'#特征':>5} {'RMSE(W)':>10} {'MAE(W)':>10} {'R²':>8} {'MAPE(%)':>10}")
    print("-"*65)
    for gname, m in results.items():
        n_feat = len(FEATURE_GROUPS[gname]['features'])
        print(f"{gname:<22} {n_feat:>5} {m['RMSE']:>10.2f} {m['MAE']:>10.2f} "
              f"{m['R2']:>8.4f} {m['MAPE']:>10.2f}")

    # ===== 6. 贡献度定量分析 =====
    print_contribution_analysis(results)

    # ===== 7. 可视化 =====
    plot_ablation_results(results)

    # ===== 8. 保存 CSV =====
    rows = []
    base = results['①基础特征组']
    for gname, m in results.items():
        rmse_drop = (base['RMSE'] - m['RMSE']) / base['RMSE'] * 100
        r2_gain   = (m['R2'] - base['R2']) / base['R2'] * 100
        rows.append({
            '特征组':       gname,
            '特征数量':     len(FEATURE_GROUPS[gname]['features']),
            'RMSE(W)':      round(m['RMSE'], 4),
            'MAE(W)':       round(m['MAE'],  4),
            'R2':           round(m['R2'],   6),
            'MAPE(%)':      round(m['MAPE'], 4),
            'RMSE降幅(%)':  round(rmse_drop, 2),
            'R²提升(%)':    round(r2_gain,   2),
            '特征描述':     FEATURE_GROUPS[gname]['description'],
        })
    pd.DataFrame(rows).to_csv('result/ablation_feature_groups_results.csv',
                               index=False, encoding='utf-8-sig')
    print("\n[INFO] 数值结果已保存: result/ablation_feature_groups_results.csv")

    print("\n" + "="*60)
    print("消融实验完成！生成文件:")
    print("  result/ablation_feature_groups_comparison.png  — 对比图")
    print("  result/ablation_feature_groups_results.csv     — 数值结果")
    print("  result/ablation_group①②③④.pth               — 各组模型权重")
    print("="*60)


if __name__ == '__main__':
    main()
