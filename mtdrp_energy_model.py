#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多行程无人机路径问题 (MTDRP) - LSTM Seq2Seq 瞬时功率预测模型

基于论文: "Drone routing with energy function: Formulation and exact algorithm"

核心特点:
1. 瞬时功率预测: 基于LSTM Seq2Seq模型利用时序特征预测功率序列
2. 多行程支持: 无人机可返回配送中心换电池后继续执行任务
3. 时间窗约束: 每个客户有到达时间窗 [a_i, b_i]
4. 载重约束: 无人机最大载重限制 Q

模型输入特征序列 (每个时刻7个特征):
- height: 高度 [m]
- VS: 竖直速度 [m/s]
- GS: 地速 [m/s]
- wind_speed: 风速 [m/s]
- temperature: 温度 [°C]
- humidity: 湿度 [%]
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
    PYTORCH_AVAILABLE = True
except ImportError:
    print("[WARNING] PyTorch未安装，LSTM模型将不可用")
    PYTORCH_AVAILABLE = False


# ==================== 全局参数配置 ====================
NUM_DRONES: int = 12  # 无人机数量


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


# ==================== LSTM Seq2Seq 功率预测模型 ====================

class LSTMPowerModel:
    """
    基于LSTM Seq2Seq的瞬时功率预测模型
    
    输入: 特征序列 (seq_len, 7) - height, VS, GS, wind_speed, temperature, humidity, wind_angle
    输出: 功率序列 (seq_len,) [W]
    """
    
    def __init__(self, model_path: str = 'result/power_lstm_seq2seq_model.pth'):
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_names = ['Height', 'VS (m/s)', 'GS (m/s)', 'Wind Speed', 
                             'Temperature', 'Humidity', 'wind_angle']
        self._load_model()
    
    def _load_model(self):
        """加载训练好的LSTM Seq2Seq模型"""
        if not PYTORCH_AVAILABLE:
            print("[WARNING] PyTorch不可用")
            return
            
        try:
            if os.path.exists(self.model_path):
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
                
                # 加载标准化器（单独的pickle文件）
                scaler_path = self.model_path.replace('.pth', '_scalers.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scalers = pickle.load(f)
                        self.feature_scaler = scalers.get('feature_scaler')
                        self.target_scaler = scalers.get('target_scaler')
                
                print(f"[OK] LSTM Seq2Seq功率模型加载成功: {self.model_path}")
            else:
                print(f"[WARNING] 模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 加载LSTM模型失败: {e}")
    
    def predict_power_sequence(self, feature_sequence: np.ndarray) -> np.ndarray:
        """
        预测功率序列
        
        参数:
            feature_sequence: 特征序列 (seq_len, 7)
            
        返回:
            功率序列 (seq_len,) [W]
        """
        if self.model is None:
            # 后备估算：使用简单公式
            GS = feature_sequence[:, 2] if feature_sequence.shape[1] > 2 else 10.0
            VS = feature_sequence[:, 1] if feature_sequence.shape[1] > 1 else 0.0
            return 4500.0 + 50.0 * GS + 100.0 * np.abs(VS)
        
        # 标准化输入
        if self.feature_scaler is not None:
            feature_sequence = self.feature_scaler.transform(feature_sequence)
        
        # 转换为张量
        with torch.no_grad():
            input_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            power_sequence = output.squeeze(0).cpu().numpy()
        
        # 反标准化输出
        if self.target_scaler is not None:
            power_sequence = self.target_scaler.inverse_transform(
                power_sequence.reshape(-1, 1)
            ).flatten()
        
        return np.maximum(0.0, power_sequence)
    
    def predict_power(self, height: float, VS: float, GS: float, 
                     wind_speed: float, temperature: float, humidity: float,
                     wind_angle: float) -> float:
        """
        预测单个时刻的功率（用于兼容旧接口）
        
        注意: LSTM模型设计用于序列预测，单点预测效果可能不如序列预测
        """
        features = np.array([[height, VS, GS, wind_speed, temperature, humidity, wind_angle]])
        power_seq = self.predict_power_sequence(features)
        return float(power_seq[0])
    
    def energy_from_power(self, power: float, duration_seconds: float) -> float:
        """根据功率和时间计算能耗 [kWh]"""
        energy_wh = power * (duration_seconds / 3600.0)
        return energy_wh / 1000.0
    
    def calculate_sequence_energy(self, power_sequence: np.ndarray, 
                                  time_interval: float = 1.0) -> float:
        """
        根据功率序列计算总能耗
        
        参数:
            power_sequence: 功率序列 [W]
            time_interval: 采样时间间隔 [秒]
            
        返回:
            总能耗 [kWh]
        """
        total_energy_wh = np.sum(power_sequence) * (time_interval / 3600.0)
        return total_energy_wh / 1000.0


# ==================== 模型工厂函数 ====================

def create_power_model(model_type: str = "lstm", instance: 'MTDRPInstance' = None):
    """
    创建瞬时功率预测模型的工厂函数
    
    参数:
        model_type: 模型类型 ("lstm")
        instance: MTDRP问题实例（保留参数，当前未使用）
        
    返回:
        瞬时功率预测模型实例
    """
    if model_type.lower() == "lstm":
        print("[OK] 使用LSTMPowerModel (基于LSTM Seq2Seq)")
        return LSTMPowerModel()
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持: 'lstm'")


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
