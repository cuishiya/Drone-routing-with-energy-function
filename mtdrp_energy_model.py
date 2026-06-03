#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多行程无人机路径问题 (MTDRP) - LSTM-Transformer 功率预测模型

基于论文: "Drone routing with energy function: Formulation and exact algorithm"

核心特点:
1. 瞬时功率预测: 基于LSTM-Transformer混合模型利用时序特征预测功率序列
2. 多行程支持: 无人机可返回配送中心换电池后继续执行任务
3. 时间窗约束: 每个客户有到达时间窗 [a_i, b_i]
4. 载重约束: 无人机最大载重限制 Q

模型输入特征序列 (每个时刻8个特征):
- Height: 高度 [m]
- VS: 竖直速度 [m/s]
- GS: 地速 [m/s]
- Wind Speed: 风速 [m/s]
- Temperature: 温度 [°C]
- Humidity: 湿度 [%]
- wind_angle: 风向夹角 [度]
- payload: 当前载重 [kg]

模型输出: 瞬时功率序列 [W]

使用 RLTS-NSGA-II 算法求解
"""

import numpy as np
import math
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
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


# ==================== LSTM-Transformer 混合模型定义 ====================

class PositionalEncoding(nn.Module):
    """正弦位置编码（自适应序列长度）"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        device = x.device
        dtype = x.dtype

        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / self.d_model)
        )

        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        x = x + pe
        return self.dropout(x)


class LSTMTransformerModel(nn.Module):
    """
    LSTM-Transformer 串行混合模型
    
    架构: 输入(8特征) → 线性映射 → 双向LSTM → 位置编码 → Transformer编码器 → 输出
    """
    def __init__(self, input_size=8, d_model=256, lstm_layers=2,
                 nhead=8, num_transformer_layers=3, dim_feedforward=512,
                 dropout=0.1, max_len=500):
        super(LSTMTransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 1. 输入映射层
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # 2. 双向LSTM（hidden * 2 = d_model）
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # 3. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 4. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 5. 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, src_key_padding_mask=None):
        """前向传播: 输入 → LSTM → Transformer → 输出"""
        x = self.input_embedding(x)
        x, _ = self.lstm(x)
        x = x * math.sqrt(self.d_model)
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
        self.device = torch.device('cpu')  # 推理统一使用CPU，避免MTDRP高频调用时GPU-CPU设备不匹配
        self.feature_names = ['Height', 'VS (m/s)', 'GS (m/s)', 'Wind Speed',
                             'Temperature', 'Humidity', 'wind_angle', 'payload']
    
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
            feature_sequence: N×8 特征矩阵 [Height, VS, GS, WindSpeed, Temp, Humidity, WindAngle, Payload]
            
        返回:
            N维功率序列 [W]
        """
        if self.feature_scaler is not None:
            feature_sequence = self.feature_scaler.transform(feature_sequence)
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)  # 始终在CPU上推理
            output = self.model(input_tensor)
            power_sequence = output.squeeze(0).numpy()
        
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
                                payload: float = 0.0,
                                time_interval: float = 1.0) -> np.ndarray:
        """
        生成弧航迹的特征序列
        
        航迹模式: 垂直上升 → 水平巡航 → 垂直下降
        
        参数:
            distance: 水平飞行距离 [m]
            cruise_height: 巡航高度 [m]
            cruise_speed: 巡航速度 [m/s]
            vertical_speed: 垂直速度 [m/s]
            wind_speed: 风速 [m/s]
            temperature: 温度 [°C]
            humidity: 湿度 [%]
            wind_angle: 风向夹角 [度]
            payload: 当前载重 [kg]（整条弧载重不变）
            time_interval: 采样间隔 [秒]
            
        返回:
            N×8 特征矩阵，每行 [Height, VS, GS, WindSpeed, Temp, Humidity, WindAngle, Payload]
        """
        trajectory_points = []
        
        # 阶段1: 垂直上升
        climb_time = cruise_height / vertical_speed
        climb_steps = max(1, int(climb_time / time_interval))
        for i in range(climb_steps):
            t = i * time_interval
            height = min(cruise_height, t * vertical_speed)
            trajectory_points.append([
                height, vertical_speed, 0.0,
                wind_speed, temperature, humidity, wind_angle, payload
            ])
        
        # 阶段2: 水平巡航
        cruise_time = distance / cruise_speed
        cruise_steps = max(1, int(cruise_time / time_interval))
        for i in range(cruise_steps):
            trajectory_points.append([
                cruise_height, 0.0, cruise_speed,
                wind_speed, temperature, humidity, wind_angle, payload
            ])
        
        # 阶段3: 垂直下降
        descent_time = cruise_height / vertical_speed
        descent_steps = max(1, int(descent_time / time_interval))
        for i in range(descent_steps):
            t = i * time_interval
            height = max(0.0, cruise_height - t * vertical_speed)
            trajectory_points.append([
                height, -vertical_speed, 0.0,
                wind_speed, temperature, humidity, wind_angle, payload
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
                          payload: float = 0.0,
                          time_interval: float = 1.0) -> Tuple[float, float]:
        """
        预测一条弧的总能耗
        
        参数:
            distance: 水平飞行距离 [m]
            payload: 当前载重 [kg]
            其他参数同 generate_arc_trajectory
            
        返回:
            (energy_kwh, total_time_seconds): 总能耗[kWh], 总飞行时间[秒]
        """
        cache_key = self._make_cache_key(
            distance,
            cruise_height,
            cruise_speed,
            vertical_speed,
            wind_speed,
            temperature,
            humidity,
            wind_angle,
            payload,
            time_interval,
        )

        if cache_key in self._arc_cache:
            return self._arc_cache[cache_key]

        # 生成航迹特征序列（含 payload 第8列）
        trajectory = self.generate_arc_trajectory(
            distance=distance,
            cruise_height=cruise_height,
            cruise_speed=cruise_speed,
            vertical_speed=vertical_speed,
            wind_speed=wind_speed,
            temperature=temperature,
            humidity=humidity,
            wind_angle=wind_angle,
            payload=payload,
            time_interval=time_interval
        )
        
        # 预测功率序列
        power_sequence = self.predict_power_sequence(trajectory)
        
        # 计算总能耗: E = Σ(P_i * Δt) / 3600 / 1000 [kWh]
        total_time_seconds = len(power_sequence) * time_interval
        total_energy_wh = np.sum(power_sequence) * time_interval / 3600.0
        energy_kwh = total_energy_wh / 1000.0
        
        result = (energy_kwh, total_time_seconds)
        # 缓存计算结果，避免重复推理
        self._arc_cache[cache_key] = result
        return result
    
    def calculate_sequence_energy(self, power_sequence: np.ndarray, 
                                  time_interval: float = 1.0) -> float:
        """根据功率序列计算总能耗 [kWh]"""
        total_energy_wh = np.sum(power_sequence) * (time_interval / 3600.0)
        return total_energy_wh / 1000.0


# ==================== LSTM-Transformer 功率预测模型 ====================

class LSTMTransformerPowerModel(BasePowerModel):
    """基于LSTM-Transformer混合模型的瞬时功率预测模型"""
    
    def __init__(self, model_path: str = 'result/power_lstm_transformer_model.pth'):
        super().__init__(model_path)
        self._arc_cache: Dict[Tuple[float, ...], Tuple[float, float]] = {}
        self._load_model()

    def _make_cache_key(self,
                         distance: float,
                         cruise_height: float,
                         cruise_speed: float,
                         vertical_speed: float,
                         wind_speed: float,
                         temperature: float,
                         humidity: float,
                         wind_angle: float,
                         payload: float,
                         time_interval: float) -> Tuple[float, ...]:
        return (
            round(distance, 2),
            round(cruise_height, 1),
            round(cruise_speed, 2),
            round(vertical_speed, 2),
            round(wind_speed, 2),
            round(temperature, 1),
            round(humidity, 1),
            round(wind_angle, 1),
            round(payload, 2),
            round(time_interval, 2),
        )
    
    def _load_model(self):
        """加载训练好的LSTM-Transformer模型"""
        if not PYTORCH_AVAILABLE:
            print("[WARNING] PyTorch不可用")
            return
            
        try:
            if os.path.exists(self.model_path):
                self.model = LSTMTransformerModel(
                    input_size=8,          # 8个特征（含 payload）
                    d_model=256,
                    lstm_layers=2,
                    nhead=8,
                    num_transformer_layers=3,
                    dim_feedforward=512,
                    dropout=0.1,
                    max_len=500
                )
                
                checkpoint = torch.load(self.model_path, map_location='cpu')
                state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                if unexpected:
                    print(f"[WARN] 忽略未匹配权重: {unexpected}")
                if missing:
                    print(f"[WARN] 未找到权重: {missing}")
                self.model.cpu()
                self.model.eval()
                
                # 消除 cuDNN 内存警告
                self.model.lstm.flatten_parameters()
                
                self._load_scalers()
                print(f"[OK] LSTM-Transformer功率模型加载成功: {self.model_path}")
            else:
                print(f"[WARNING] 模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 加载LSTM-Transformer模型失败: {e}")


# ==================== 模型工厂函数 ====================

def create_power_model(model_type: str = "lstm_transformer", instance: 'MTDRPInstance' = None):
    """
    创建瞬时功率预测模型的工厂函数
    
    当前唯一支持的模型类型: "lstm_transformer"
    输入特征: 8个 (Height, VS, GS, WindSpeed, Temp, Humidity, wind_angle, payload)
    """
    model_type = model_type.lower()
    
    if model_type in ("lstm_transformer", "lstm-transformer"):
        print("[OK] 使用 LSTM-Transformer 功率预测模型")
        return LSTMTransformerPowerModel()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}。当前唯一支持 'lstm_transformer'")


# ==================== 能耗计算辅助函数 ====================

def calculate_trip_energy(power_model: LSTMTransformerPowerModel, trajectory_points: List[Dict],
                         time_interval: float = 1.0) -> float:
    """
    根据轨迹点序列计算总能耗
    
    参数:
        power_model: LSTM-Transformer功率预测模型
        trajectory_points: 轨迹点列表，每个点包含 {height, VS, GS, wind_speed, temperature, humidity, wind_angle, payload}
        time_interval: 采样时间间隔 [秒]
        
    返回:
        总能耗 [kWh]
    """
    if not trajectory_points:
        return 0.0
    
    # 构建特征序列（含 payload 第8列）
    feature_sequence = np.array([
        [
            point.get('height', 100.0),
            point.get('VS', 0.0),
            point.get('GS', 10.0),
            point.get('wind_speed', 2.0),
            point.get('temperature', 25.0),
            point.get('humidity', 60.0),
            point.get('wind_angle', 90.0),
            point.get('payload', 0.0)
        ]
        for point in trajectory_points
    ])
    
    power_sequence = power_model.predict_power_sequence(feature_sequence)
    return power_model.calculate_sequence_energy(power_sequence, time_interval)


def estimate_flight_energy(power_model: LSTMTransformerPowerModel, distance: float,
                          speed: float = 10.0, height: float = 100.0,
                          wind_speed: float = 2.0, temperature: float = 25.0,
                          humidity: float = 60.0, wind_angle: float = 90.0,
                          payload: float = 0.0, time_interval: float = 1.0) -> float:
    """
    估算一段飞行的能耗
    
    参数:
        power_model: LSTM-Transformer功率预测模型
        distance: 飞行距离 [m]
        payload: 载重 [kg]
        其他参数: 速度、高度、风速、温度、湿度、风向夹角
        
    返回:
        估算能耗 [kWh]
    """
    flight_time = distance / speed
    num_points = max(1, int(flight_time / time_interval))
    
    # 构建特征序列（匹匀速平飞，含 payload）
    feature_sequence = np.array([
        [height, 0.0, speed, wind_speed, temperature, humidity, wind_angle, payload]
        for _ in range(num_points)
    ])
    
    power_sequence = power_model.predict_power_sequence(feature_sequence)
    return power_model.calculate_sequence_energy(power_sequence, time_interval)


def predict_flight_plan_energy(power_model: LSTMTransformerPowerModel,
                               flight_plan: List[Dict],
                               time_interval: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    根据预设飞行计划预测功率序列和总能耗
    
    参数:
        power_model: LSTM-Transformer功率预测模型
        flight_plan: 飞行计划列表，每个点包含:
            - height, VS, GS, wind_speed, temperature, humidity, wind_angle, payload
        time_interval: 采样时间间隔 [秒]
        
    返回:
        (power_sequence, total_energy): 功率序列 [W] 和总能耗 [kWh]
    """
    if not flight_plan:
        return np.array([]), 0.0
    
    # 构建特征序列（含 payload 第8列）
    feature_sequence = np.array([
        [
            point.get('height', 100.0),
            point.get('VS', 0.0),
            point.get('GS', 10.0),
            point.get('wind_speed', 2.0),
            point.get('temperature', 25.0),
            point.get('humidity', 60.0),
            point.get('wind_angle', 90.0),
            point.get('payload', 0.0)
        ]
        for point in flight_plan
    ])
    
    # 使用LSTM预测功率序列
    power_sequence = power_model.predict_power_sequence(feature_sequence)
    
    # 计算总能耗
    total_energy = power_model.calculate_sequence_energy(power_sequence, time_interval)
    
    return power_sequence, total_energy


# ==================== 多行程无人机路径建模 ====================

@dataclass
class EvaluationResult:
    """多行程无人机路径评估结果"""
    total_distance: float
    total_energy: float
    total_delay: float
    penalties: Dict[str, float]
    total_penalty: float
    objective_cost: float
    visited_customers: Set[int]
    unvisited_customers: Set[int]
    arc_payloads: Dict[Tuple[int, int], float]
    drone_schedules: Dict[int, List[Tuple[int, float]]]


class MTDRPSolution:
    """多行程无人机路径解的表示"""

    def __init__(self, instance: MTDRPInstance):
        self.instance = instance
        # routes[drone_id] = [trip1, trip2, ...]，trip 为客户ID列表
        self.routes: Dict[int, List[List[int]]] = {k: [] for k in range(instance.num_drones)}

    def add_trip(self, drone_id: int, customers: List[int]):
        """为指定无人机添加一个行程"""
        if not customers:
            return
        self.routes.setdefault(drone_id, [])
        self.routes[drone_id].append(customers)

    def get_all_trips(self) -> List[Tuple[int, int, List[int]]]:
        """返回所有行程 (drone_id, trip_idx, customers)"""
        trips: List[Tuple[int, int, List[int]]] = []
        for drone_id, drone_trips in self.routes.items():
            for trip_idx, customers in enumerate(drone_trips):
                trips.append((drone_id, trip_idx, customers))
        return trips

    def get_visited_customers(self) -> Set[int]:
        """返回所有已访问客户集合"""
        visited: Set[int] = set()
        for drone_trips in self.routes.values():
            for trip in drone_trips:
                visited.update(trip)
        return visited

    def is_complete(self) -> bool:
        """是否已覆盖所有客户"""
        required = {c.id for c in self.instance.customers}
        return self.get_visited_customers() == required


class MTDRPModel:
    """多行程无人机路径问题建模与评估器"""

    def __init__(
        self,
        instance: MTDRPInstance,
        power_model: Optional[LSTMTransformerPowerModel] = None,
        energy_weight: float = 1000.0,
        battery_swap_time: float = 5.0,
        environment: Optional[Dict[str, float]] = None,
    ):
        self.instance = instance
        self.power_model = power_model or LSTMTransformerPowerModel()
        self.energy_weight = energy_weight
        self.battery_swap_time = battery_swap_time
        self.environment = environment or {
            'cruise_height': 120.0,
            'cruise_speed': instance.drone_params.speed,
            'vertical_speed': 3.0,
            'wind_speed': 2.0,
            'temperature': 25.0,
            'humidity': 60.0,
            'wind_angle': 90.0,
        }
        self.customer_map: Dict[int, Customer] = {c.id: c for c in self.instance.customers}
        # 客户ID到矩阵索引映射
        self.node_index: Dict[int, int] = {0: 0}
        for idx, customer in enumerate(self.instance.customers, start=1):
            self.node_index[customer.id] = idx

    def evaluate(self, solution: MTDRPSolution) -> Tuple[float, float, Dict[str, float]]:
        """评估解的目标函数与约束，返回 (目标值, 罚则, 详情)"""
        result = self._evaluate_solution(solution)
        details = {
            'total_energy': result.total_energy,
            'total_distance': result.total_distance,
            'total_delay': result.total_delay,
            'penalties': result.penalties,
            'total_penalty': result.total_penalty,
            'num_trips': sum(len(v) for v in solution.routes.values()),
            'visited_customers': len(result.visited_customers),
            'unvisited_customers': len(result.unvisited_customers),
            'arc_payloads': result.arc_payloads,
            'drone_schedules': result.drone_schedules,
        }
        return result.objective_cost, result.total_penalty, details

    # -------------------- 内部评估逻辑 --------------------

    def _evaluate_solution(self, solution: MTDRPSolution) -> EvaluationResult:
        total_energy = 0.0
        total_distance = 0.0
        total_delay = 0.0
        penalties: Dict[str, float] = {
            'unvisited': 0.0,
            'duplicate': 0.0,
            'capacity': 0.0,
            'energy': 0.0,
            'energy_midway': 0.0,
            'time_window': 0.0,
        }
        visited: Set[int] = set()
        arc_payloads: Dict[Tuple[int, int], float] = {}
        drone_schedules: Dict[int, List[Tuple[int, float]]] = {k: [] for k in solution.routes.keys()}

        for drone_id, trips in solution.routes.items():
            available_time = 0.0
            for trip_idx, trip in enumerate(trips):
                trip_result = self._evaluate_trip(
                    trip=trip,
                    already_visited=visited,
                    start_time=available_time,
                    arc_payloads=arc_payloads,
                )

                total_energy += trip_result['energy']
                total_distance += trip_result['distance']
                total_delay += trip_result['delay']

                for key in penalties:
                    penalties[key] += trip_result['penalties'].get(key, 0.0)

                visited.update(trip)
                end_time = trip_result['end_time']
                drone_schedules.setdefault(drone_id, []).append((trip_idx, end_time))
                available_time = end_time + self.battery_swap_time

        required = {c.id for c in self.instance.customers}
        unvisited = required - visited
        penalties['unvisited'] = len(unvisited) * 10000.0

        total_penalty = sum(penalties.values())
        objective_cost = total_distance + self.energy_weight * total_energy

        return EvaluationResult(
            total_distance=total_distance,
            total_energy=total_energy,
            total_delay=total_delay,
            penalties=penalties,
            total_penalty=total_penalty,
            objective_cost=objective_cost,
            visited_customers=visited,
            unvisited_customers=unvisited,
            arc_payloads=arc_payloads,
            drone_schedules=drone_schedules,
        )

    def _evaluate_trip(
        self,
        trip: List[int],
        already_visited: Set[int],
        start_time: float,
        arc_payloads: Dict[Tuple[int, int], float],
    ) -> Dict[str, float]:
        """评估单条行程，返回局部统计"""
        result = {
            'energy': 0.0,
            'distance': 0.0,
            'delay': 0.0,
            'end_time': start_time,
            'penalties': {},
        }

        if not trip:
            return result

        penalties: Dict[str, float] = {}
        current_node = 0
        current_time = start_time

        current_payload = sum(
            self._get_customer(cust_id).demand for cust_id in trip if self._get_customer(cust_id) is not None
        )

        if current_payload > self.instance.drone_params.Q:
            penalties['capacity'] = (current_payload - self.instance.drone_params.Q) * 1000.0

        cumulative_energy = 0.0

        for cust_id in trip:
            customer = self._get_customer(cust_id)
            if customer is None:
                continue

            if cust_id in already_visited:
                penalties['duplicate'] = penalties.get('duplicate', 0.0) + 5000.0

            arc_payloads[(current_node, cust_id)] = current_payload

            energy, distance, travel_time = self._calculate_arc_energy(current_node, cust_id, current_payload)
            cumulative_energy += energy
            result['distance'] += distance

            if cumulative_energy > self.instance.drone_params.sigma:
                penalties['energy_midway'] = penalties.get('energy_midway', 0.0) + \
                    (cumulative_energy - self.instance.drone_params.sigma) * 5000.0

            arrival_time = current_time + travel_time
            if arrival_time < customer.earliest_time:
                current_time = customer.earliest_time + customer.service_time
            elif arrival_time > customer.latest_time:
                delay = arrival_time - customer.latest_time
                result['delay'] += delay
                penalties['time_window'] = penalties.get('time_window', 0.0) + delay * 100.0
                current_time = arrival_time + customer.service_time
            else:
                current_time = arrival_time + customer.service_time

            result['energy'] += energy
            current_payload -= customer.demand
            current_node = cust_id

        arc_payloads[(current_node, 0)] = current_payload
        energy_back, distance_back, travel_time_back = self._calculate_arc_energy(current_node, 0, current_payload)
        result['energy'] += energy_back
        result['distance'] += distance_back
        cumulative_energy += energy_back
        current_time += travel_time_back

        if cumulative_energy > self.instance.drone_params.sigma:
            penalties['energy'] = penalties.get('energy', 0.0) + \
                (cumulative_energy - self.instance.drone_params.sigma) * 10000.0

        result['end_time'] = current_time
        result['penalties'] = penalties
        return result

    # -------------------- 工具函数 --------------------

    def _get_customer(self, cust_id: int) -> Optional[Customer]:
        return self.customer_map.get(cust_id)

    def _calculate_arc_energy(self, from_node: int, to_node: int, payload: float) -> Tuple[float, float, float]:
        """计算弧 (i,j) 的能耗、距离与飞行时间"""
        from_idx = self.node_index.get(from_node)
        to_idx = self.node_index.get(to_node)
        if from_idx is None or to_idx is None:
            return 0.0, 0.0, 0.0

        distance = float(self.instance.distance_matrix[from_idx, to_idx])
        if distance <= 0:
            return 0.0, 0.0, 0.0

        cruise_height = self.environment.get('cruise_height', 120.0)
        cruise_speed = self.environment.get('cruise_speed', self.instance.drone_params.speed)
        vertical_speed = self.environment.get('vertical_speed', 3.0)
        wind_speed = self.environment.get('wind_speed', 2.0)
        temperature = self.environment.get('temperature', 25.0)
        humidity = self.environment.get('humidity', 60.0)
        wind_angle = self.environment.get('wind_angle', 90.0)

        energy_kwh, total_time_seconds = self.power_model.predict_arc_energy(
            distance=distance,
            cruise_height=cruise_height,
            cruise_speed=cruise_speed,
            vertical_speed=vertical_speed,
            wind_speed=wind_speed,
            temperature=temperature,
            humidity=humidity,
            wind_angle=wind_angle,
            payload=payload,
            time_interval=1.0,
        )

        travel_time_minutes = total_time_seconds / 60.0
        return energy_kwh, distance, travel_time_minutes


def build_mtdrp_model(
    instance: MTDRPInstance,
    model_type: str = "lstm_transformer",
    **kwargs,
) -> MTDRPModel:
    """构建多行程无人机路径模型，默认使用 LSTM-Transformer 能耗预测"""
    power_model = create_power_model(model_type, instance)
    return MTDRPModel(instance=instance, power_model=power_model, **kwargs)
