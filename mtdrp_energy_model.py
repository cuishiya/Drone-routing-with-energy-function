#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多行程无人机路径问题 (MTDRP) - 时序功率预测模型

基于论文: "Drone routing with energy function: Formulation and exact algorithm"

核心特点:
1. 瞬时功率预测: 基于深度学习模型利用时序特征预测功率序列
2. 多行程支持: 无人机可返回配送中心换电池后继续执行任务
3. 时间窗约束: 每个客户有到达时间窗 [a_i, b_i]
4. 载重约束: 无人机最大载重限制 Q

支持的模型类型 (按性能排序):
- bilstm: 双向LSTM模型 (推荐，R²=0.8287)
- gru: GRU Seq2Seq模型 (R²=0.8247)
- lstm: LSTM Seq2Seq模型 (R²=0.8155)
- transformer: Transformer模型 (R²=0.8104)

模型输入特征序列 (每个时刻7个特征):
- Height: 高度 [m]
- VS: 竖直速度 [m/s]
- GS: 地速 [m/s]
- Wind Speed: 风速 [m/s]
- Temperature: 温度 [°C]
- Humidity: 湿度 [%]
- wind_angle: 风向夹角 [度]

模型输出: 瞬时功率序列 [W]

使用 RLTS-NSGA-II 算法求解
"""

import numpy as np
import math
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pickle

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import warnings
    import math
    PYTORCH_AVAILABLE = True
    # 禁用 cuDNN RNN 内存警告
    warnings.filterwarnings('ignore', message='RNN module weights are not part of single contiguous chunk of memory')
except ImportError:
    print("[WARNING] PyTorch未安装，深度学习模型将不可用")
    PYTORCH_AVAILABLE = False


# ==================== 全局参数配置 ====================
NUM_DRONES: int = 5  # 无人机数量


@dataclass
class DroneParameters:
    """无人机物理参数 - 基于UAS04028624实际参数"""
    W: float = 36.0         # 无人机自重（含电池） [kg]
    m: float = 0.0          # 电池重量已包含在自重中 [kg]
    Q: float = 8.0          # 最大载重 [kg]
    g: float = 9.81         # 重力加速度 [N/kg]
    rho: float = 1.204      # 空气密度 [kg/m^3]
    xi: float = 0.3848      # 旋翼圆盘面积 [m^2]
    h: int = 6              # 旋翼数量
    sigma: float = 1.0      # 电池能量容量 [kWh]
    speed: float = 10.0     # 飞行速度 [m/s]
    drone_id: str = "UAS04028624"  # 无人机ID
    
    @property
    def k(self) -> float:
        """计算能量常数 k = sqrt(g^3 / (2 * rho * xi * h))"""
        return math.sqrt(self.g**3 / (2 * self.rho * self.xi * self.h))
    
    @property
    def k_prime(self) -> float:
        """k' 用于约束方程，包含单位转换"""
        return self.k / 3600000.0


@dataclass
class Customer:
    """客户节点"""
    id: int                 # 客户ID
    x: float                # x坐标
    y: float                # y坐标
    demand: float           # 需求量/载重 [kg]
    earliest_time: float    # 最早服务时间
    latest_time: float      # 最晚服务时间
    service_time: float     # 服务时间 [分钟]


@dataclass
class Depot:
    """配送中心"""
    id: int = 0
    x: float = 5000.0
    y: float = 5000.0
    

@dataclass
class MTDRPInstance:
    """MTDRP问题实例"""
    name: str
    depot: Depot
    customers: List[Customer]
    drone_params: DroneParameters
    num_drones: int
    time_horizon: float = 720.0  # 时间范围 [分钟]
    
    # 预计算的距离矩阵和时间矩阵
    distance_matrix: np.ndarray = field(default=None, repr=False)
    travel_time_matrix: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        """初始化后计算距离矩阵"""
        self._compute_matrices()
    
    def _compute_matrices(self):
        """计算距离矩阵和旅行时间矩阵"""
        n = len(self.customers) + 1  # +1 for depot
        self.distance_matrix = np.zeros((n, n))
        self.travel_time_matrix = np.zeros((n, n))
        
        all_nodes = [(self.depot.x, self.depot.y)] + [(c.x, c.y) for c in self.customers]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = all_nodes[i][0] - all_nodes[j][0]
                    dy = all_nodes[i][1] - all_nodes[j][1]
                    dist = math.sqrt(dx**2 + dy**2)
                    self.distance_matrix[i, j] = dist
                    self.travel_time_matrix[i, j] = (dist / self.drone_params.speed) / 60.0
    
    @property
    def num_customers(self) -> int:
        return len(self.customers)
    
    @property
    def num_nodes(self) -> int:
        return len(self.customers) + 1


def load_instance(filepath: str) -> MTDRPInstance:
    """
    从文件加载MTDRP问题实例
    """
    drone_params = DroneParameters()
    customers = []
    depot = Depot()
    instance_name = os.path.basename(filepath)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    in_customer_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('Customers_data'):
            in_customer_section = True
            continue
        
        if line.endswith('_data') or line.startswith('Num_'):
            in_customer_section = False
            continue
        
        if in_customer_section:
            if line.startswith('id'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                cust_id = int(parts[0])
                if cust_id == 0:
                    depot = Depot(id=0, x=float(parts[4]), y=float(parts[5]))
                else:
                    customer = Customer(
                        id=cust_id,
                        x=float(parts[4]),
                        y=float(parts[5]),
                        demand=float(parts[6]),
                        earliest_time=float(parts[1]),
                        latest_time=float(parts[2]),
                        service_time=float(parts[3])
                    )
                    customers.append(customer)
    
    return MTDRPInstance(
        name=instance_name,
        depot=depot,
        customers=customers,
        drone_params=drone_params,
        num_drones=NUM_DRONES
    )


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


# ==================== Bi-LSTM 模型定义 ====================

class BiLSTMModel(nn.Module):
    """双向LSTM功率预测模型"""
    def __init__(self, input_size=7, hidden_size=128, num_layers=3, dropout=0.3):
        super(BiLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output.squeeze(-1)


# ==================== Transformer 模型定义 ====================

class PositionalEncoding(nn.Module):
    """位置编码"""
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
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = self.output_layer(x)
        return output.squeeze(-1)


# ==================== 功率预测模型基类 ====================

class BasePowerModel:
    """功率预测模型基类"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_names = ['Height', 'VS (m/s)', 'GS (m/s)', 'Wind Speed', 
                             'Temperature', 'Humidity', 'wind_angle']
    
    def _load_scalers(self):
        """加载标准化器"""
        scaler_path = self.model_path.replace('.pth', '_scalers.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.feature_scaler = scalers.get('feature_scaler')
                self.target_scaler = scalers.get('target_scaler')
    
    def predict_power_sequence(self, feature_sequence: np.ndarray) -> np.ndarray:
        """
        预测功率序列
        
        参数:
            feature_sequence: N×7 特征矩阵 [Height, VS, GS, WindSpeed, Temp, Humidity, WindAngle]
            
        返回:
            N维功率序列 [W]
        """
        if self.feature_scaler is not None:
            feature_sequence = self.feature_scaler.transform(feature_sequence)
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            power_sequence = output.squeeze(0).cpu().numpy()
        
        if self.target_scaler is not None:
            power_sequence = self.target_scaler.inverse_transform(
                power_sequence.reshape(-1, 1)
            ).flatten()
        
        return np.maximum(0.0, power_sequence)
    
    def generate_arc_trajectory(self, distance: float, 
                                cruise_height: float = 120.0,
                                cruise_speed: float = 10.0,
                                vertical_speed: float = 3.0,
                                wind_speed: float = 2.0,
                                temperature: float = 25.0,
                                humidity: float = 60.0,
                                wind_angle: float = 90.0,
                                time_interval: float = 1.0) -> np.ndarray:
        """
        生成弧航迹的特征序列
        
        航迹模式: 垂直上升 → 水平巡航 → 垂直下降
        
        阶段1 - 垂直上升:
            - Height: 0 → cruise_height (线性增加)
            - VS: +vertical_speed (正值表示上升)
            - GS: 0 (无水平移动)
            
        阶段2 - 水平巡航:
            - Height: cruise_height (恒定)
            - VS: 0
            - GS: cruise_speed
            
        阶段3 - 垂直下降:
            - Height: cruise_height → 0 (线性减少)
            - VS: -vertical_speed (负值表示下降)
            - GS: 0 (无水平移动)
        
        参数:
            distance: 水平飞行距离 [m]
            cruise_height: 巡航高度 [m]
            cruise_speed: 巡航速度 [m/s]
            vertical_speed: 垂直速度 [m/s]
            wind_speed: 风速 [m/s]
            temperature: 温度 [°C]
            humidity: 湿度 [%]
            wind_angle: 风向夹角 [度]
            time_interval: 采样间隔 [秒]
            
        返回:
            N×7 特征矩阵，每行 [Height, VS, GS, WindSpeed, Temp, Humidity, WindAngle]
        """
        trajectory_points = []
        
        # 阶段1: 垂直上升
        climb_time = cruise_height / vertical_speed  # 上升所需时间 [秒]
        climb_steps = max(1, int(climb_time / time_interval))
        for i in range(climb_steps):
            t = i * time_interval
            height = min(cruise_height, t * vertical_speed)
            trajectory_points.append([
                height,           # Height
                vertical_speed,   # VS (正值=上升)
                0.0,              # GS (无水平移动)
                wind_speed,
                temperature,
                humidity,
                wind_angle
            ])
        
        # 阶段2: 水平巡航
        cruise_time = distance / cruise_speed  # 巡航所需时间 [秒]
        cruise_steps = max(1, int(cruise_time / time_interval))
        for i in range(cruise_steps):
            trajectory_points.append([
                cruise_height,    # Height (恒定)
                0.0,              # VS (无垂直移动)
                cruise_speed,     # GS
                wind_speed,
                temperature,
                humidity,
                wind_angle
            ])
        
        # 阶段3: 垂直下降
        descent_time = cruise_height / vertical_speed  # 下降所需时间 [秒]
        descent_steps = max(1, int(descent_time / time_interval))
        for i in range(descent_steps):
            t = i * time_interval
            height = max(0.0, cruise_height - t * vertical_speed)
            trajectory_points.append([
                height,            # Height
                -vertical_speed,   # VS (负值=下降)
                0.0,               # GS (无水平移动)
                wind_speed,
                temperature,
                humidity,
                wind_angle
            ])
        
        return np.array(trajectory_points)
    
    def predict_arc_energy(self, distance: float,
                          cruise_height: float = 120.0,
                          cruise_speed: float = 10.0,
                          vertical_speed: float = 3.0,
                          wind_speed: float = 2.0,
                          temperature: float = 25.0,
                          humidity: float = 60.0,
                          wind_angle: float = 90.0,
                          time_interval: float = 1.0) -> Tuple[float, float]:
        """
        预测一条弧的总能耗
        
        参数:
            distance: 水平飞行距离 [m]
            其他参数同 generate_arc_trajectory
            
        返回:
            (energy_kwh, total_time_seconds): 总能耗[kWh], 总飞行时间[秒]
        """
        # 生成航迹特征序列
        trajectory = self.generate_arc_trajectory(
            distance=distance,
            cruise_height=cruise_height,
            cruise_speed=cruise_speed,
            vertical_speed=vertical_speed,
            wind_speed=wind_speed,
            temperature=temperature,
            humidity=humidity,
            wind_angle=wind_angle,
            time_interval=time_interval
        )
        
        # 预测功率序列
        power_sequence = self.predict_power_sequence(trajectory)
        
        # 计算总能耗: E = Σ(P_i * Δt) / 3600 / 1000 [kWh]
        total_time_seconds = len(power_sequence) * time_interval
        total_energy_wh = np.sum(power_sequence) * time_interval / 3600.0
        energy_kwh = total_energy_wh / 1000.0
        
        return energy_kwh, total_time_seconds
    
    def calculate_sequence_energy(self, power_sequence: np.ndarray, 
                                  time_interval: float = 1.0) -> float:
        """根据功率序列计算总能耗 [kWh]"""
        total_energy_wh = np.sum(power_sequence) * (time_interval / 3600.0)
        return total_energy_wh / 1000.0


# ==================== LSTM Seq2Seq 功率预测模型 ====================

class LSTMPowerModel(BasePowerModel):
    """基于LSTM Seq2Seq的瞬时功率预测模型"""
    
    def __init__(self, model_path: str = 'result/power_lstm_seq2seq_model.pth'):
        super().__init__(model_path)
        self._load_model()
    
    def _load_model(self):
        """加载训练好的LSTM Seq2Seq模型"""
        if not PYTORCH_AVAILABLE:
            print("[WARNING] PyTorch不可用")
            return
            
        try:
            if os.path.exists(self.model_path):
                # 统一参数配置（与训练脚本保持一致）
                self.model = LSTMSeq2Seq(
                    input_size=7,
                    hidden_size=256,     # 统一为256
                    num_layers=3,        # 统一为3
                    dropout=0.2,
                    bidirectional=True
                )
                
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # 消除 cuDNN 内存警告
                self.model.encoder.lstm.flatten_parameters()
                self.model.decoder.lstm.flatten_parameters()
                
                self._load_scalers()
                print(f"[OK] LSTM Seq2Seq功率模型加载成功: {self.model_path}")
            else:
                print(f"[WARNING] 模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 加载LSTM模型失败: {e}")


# ==================== GRU Seq2Seq 功率预测模型 ====================

class GRUPowerModel(BasePowerModel):
    """基于GRU Seq2Seq的瞬时功率预测模型"""
    
    def __init__(self, model_path: str = 'result/power_gru_model.pth'):
        super().__init__(model_path)
        self._load_model()
    
    def _load_model(self):
        """加载训练好的GRU Seq2Seq模型"""
        if not PYTORCH_AVAILABLE:
            print("[WARNING] PyTorch不可用")
            return
            
        try:
            if os.path.exists(self.model_path):
                # 统一参数配置（与训练脚本保持一致）
                self.model = GRUSeq2Seq(
                    input_size=7,
                    hidden_size=256,     # 统一为256
                    num_layers=3,        # 统一为3
                    dropout=0.2,
                    bidirectional=True
                )
                
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # 消除 cuDNN 内存警告
                self.model.encoder.gru.flatten_parameters()
                self.model.decoder.gru.flatten_parameters()
                
                self._load_scalers()
                print(f"[OK] GRU Seq2Seq功率模型加载成功: {self.model_path}")
            else:
                print(f"[WARNING] 模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 加载GRU模型失败: {e}")


# ==================== Bi-LSTM 功率预测模型 ====================

class BiLSTMPowerModel(BasePowerModel):
    """基于Bi-LSTM的瞬时功率预测模型"""
    
    def __init__(self, model_path: str = 'result/power_bilstm_v2_model.pth'):
        super().__init__(model_path)
        self._load_model()
    
    def _load_model(self):
        """加载训练好的Bi-LSTM模型"""
        if not PYTORCH_AVAILABLE:
            print("[WARNING] PyTorch不可用")
            return
            
        try:
            if os.path.exists(self.model_path):
                # 统一参数配置（与训练脚本保持一致）
                self.model = BiLSTMModel(
                    input_size=7,
                    hidden_size=256,     # 统一为256
                    num_layers=3,
                    dropout=0.2          # 统一为0.2
                )
                
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # 消除 cuDNN 内存警告
                self.model.lstm.flatten_parameters()
                
                self._load_scalers()
                print(f"[OK] Bi-LSTM功率模型加载成功: {self.model_path}")
            else:
                print(f"[WARNING] 模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 加载Bi-LSTM模型失败: {e}")


# ==================== Transformer 功率预测模型 ====================

class TransformerPowerModel(BasePowerModel):
    """基于Transformer的瞬时功率预测模型"""
    
    def __init__(self, model_path: str = 'result/power_transformer_model.pth'):
        super().__init__(model_path)
        self._load_model()
    
    def _load_model(self):
        """加载训练好的Transformer模型"""
        if not PYTORCH_AVAILABLE:
            print("[WARNING] PyTorch不可用")
            return
            
        try:
            if os.path.exists(self.model_path):
                # 统一参数配置（与训练脚本保持一致）
                self.model = TransformerModel(
                    input_size=7,
                    d_model=256,         # 统一为256
                    nhead=8,
                    num_layers=3,        # 统一为3
                    dim_feedforward=1024,# 4倍d_model
                    dropout=0.2,         # 统一为0.2
                    max_len=500
                )
                
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                self._load_scalers()
                print(f"[OK] Transformer功率模型加载成功: {self.model_path}")
            else:
                print(f"[WARNING] 模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 加载Transformer模型失败: {e}")


# ==================== 模型工厂函数 ====================

# 模型配置信息
MODEL_CONFIGS = {
    'lstm': {
        'name': 'LSTM Seq2Seq',
        'class': 'LSTMPowerModel',
        'path': 'result/power_lstm_seq2seq_model.pth',
        'r2': 0.8155,
        'description': '编码器-解码器结构的LSTM模型'
    },
    'gru': {
        'name': 'GRU Seq2Seq',
        'class': 'GRUPowerModel',
        'path': 'result/power_gru_model.pth',
        'r2': 0.8247,
        'description': '编码器-解码器结构的GRU模型'
    },
    'bilstm': {
        'name': 'Bi-LSTM',
        'class': 'BiLSTMPowerModel',
        'path': 'result/power_bilstm_v2_model.pth',
        'r2': 0.8287,
        'description': '双向LSTM模型（推荐）'
    },
    'transformer': {
        'name': 'Transformer',
        'class': 'TransformerPowerModel',
        'path': 'result/power_transformer_model.pth',
        'r2': 0.8104,
        'description': '基于自注意力机制的Transformer模型'
    }
}


def create_power_model(model_type: str = "bilstm", instance: 'MTDRPInstance' = None):
    """
    创建瞬时功率预测模型的工厂函数
    
    参数:
        model_type: 模型类型 ("lstm", "gru", "bilstm", "transformer")
                   默认使用 bilstm（性能最佳）
        instance: MTDRP问题实例（保留参数，当前未使用）
        
    返回:
        瞬时功率预测模型实例
        
    模型性能对比 (test_data测试集):
        | 模型        | R²     | RMSE (W) | MAE (W) |
        |-------------|--------|----------|---------|
        | Bi-LSTM     | 0.8287 | 397.78   | 291.08  |
        | GRU         | 0.8247 | 402.50   | 295.57  |
        | LSTM        | 0.8155 | 412.86   | 305.24  |
        | Transformer | 0.8104 | 418.51   | 292.74  |
    """
    model_type = model_type.lower()
    
    if model_type == "lstm":
        print("[OK] 使用 LSTM Seq2Seq 功率预测模型 (R^2=0.8155)")
        return LSTMPowerModel()
    
    elif model_type == "gru":
        print("[OK] 使用 GRU Seq2Seq 功率预测模型 (R^2=0.8247)")
        return GRUPowerModel()
    
    elif model_type == "bilstm":
        print("[OK] 使用 Bi-LSTM 功率预测模型 (推荐, R^2=0.8287)")
        return BiLSTMPowerModel()
    
    elif model_type == "transformer":
        print("[OK] 使用 Transformer 功率预测模型 (R^2=0.8104)")
        return TransformerPowerModel()
    
    else:
        supported = list(MODEL_CONFIGS.keys())
        raise ValueError(f"不支持的模型类型: {model_type}. 支持: {supported}")


# ==================== 能耗计算辅助函数 ====================

def calculate_trip_energy(power_model: LSTMPowerModel, trajectory_points: List[Dict], 
                         time_interval: float = 1.0) -> float:
    """
    根据轨迹点序列计算总能耗（利用LSTM序列预测）
    
    参数:
        power_model: LSTM功率预测模型
        trajectory_points: 轨迹点列表，每个点包含 {height, VS, GS, wind_speed, temperature, humidity, wind_angle}
        time_interval: 采样时间间隔 [秒]
        
    返回:
        总能耗 [kWh]
    """
    if not trajectory_points:
        return 0.0
    
    # 构建特征序列
    feature_sequence = np.array([
        [
            point.get('height', 100.0),
            point.get('VS', 0.0),
            point.get('GS', 10.0),
            point.get('wind_speed', 2.0),
            point.get('temperature', 25.0),
            point.get('humidity', 60.0),
            point.get('wind_angle', 90.0)
        ]
        for point in trajectory_points
    ])
    
    # 使用LSTM预测功率序列
    power_sequence = power_model.predict_power_sequence(feature_sequence)
    
    # 计算总能耗
    total_energy = power_model.calculate_sequence_energy(power_sequence, time_interval)
    
    return total_energy


def estimate_flight_energy(power_model: LSTMPowerModel, distance: float, speed: float = 10.0,
                          height: float = 100.0, wind_speed: float = 2.0,
                          temperature: float = 25.0, humidity: float = 60.0,
                          wind_angle: float = 90.0, time_interval: float = 1.0) -> float:
    """
    估算一段飞行的能耗（基于LSTM序列预测）
    
    参数:
        power_model: LSTM功率预测模型
        distance: 飞行距离 [m]
        speed: 飞行速度 [m/s]
        height: 飞行高度 [m]
        wind_speed: 风速 [m/s]
        temperature: 温度 [°C]
        humidity: 湿度 [%]
        wind_angle: 风向夹角 [度]
        time_interval: 采样时间间隔 [秒]
        
    返回:
        估算能耗 [kWh]
    """
    # 计算飞行时间和采样点数
    flight_time = distance / speed  # 秒
    num_points = max(1, int(flight_time / time_interval))
    
    # 构建特征序列（假设匀速平飞）
    feature_sequence = np.array([
        [height, 0.0, speed, wind_speed, temperature, humidity, wind_angle]
        for _ in range(num_points)
    ])
    
    # 使用LSTM预测功率序列
    power_sequence = power_model.predict_power_sequence(feature_sequence)
    
    # 计算总能耗
    total_energy = power_model.calculate_sequence_energy(power_sequence, time_interval)
    
    return total_energy


def predict_flight_plan_energy(power_model: LSTMPowerModel, 
                               flight_plan: List[Dict],
                               time_interval: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    根据预设飞行计划预测功率序列和总能耗
    
    参数:
        power_model: LSTM功率预测模型
        flight_plan: 飞行计划列表，每个点包含:
            - height: 高度 [m]
            - VS: 竖直速度 [m/s]
            - GS: 地速 [m/s]
            - wind_speed: 风速 [m/s]
            - temperature: 温度 [°C]
            - humidity: 湿度 [%]
            - wind_angle: 风向夹角 [度]
        time_interval: 采样时间间隔 [秒]
        
    返回:
        (power_sequence, total_energy): 功率序列 [W] 和总能耗 [kWh]
    """
    if not flight_plan:
        return np.array([]), 0.0
    
    # 构建特征序列
    feature_sequence = np.array([
        [
            point.get('height', 100.0),
            point.get('VS', 0.0),
            point.get('GS', 10.0),
            point.get('wind_speed', 2.0),
            point.get('temperature', 25.0),
            point.get('humidity', 60.0),
            point.get('wind_angle', 90.0)
        ]
        for point in flight_plan
    ])
    
    # 使用LSTM预测功率序列
    power_sequence = power_model.predict_power_sequence(feature_sequence)
    
    # 计算总能耗
    total_energy = power_model.calculate_sequence_energy(power_sequence, time_interval)
    
    return power_sequence, total_energy
