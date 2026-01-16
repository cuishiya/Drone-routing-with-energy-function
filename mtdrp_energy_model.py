#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多行程无人机路径问题 (MTDRP) - 瞬时功率预测模型

基于论文: "Drone routing with energy function: Formulation and exact algorithm"

核心特点:
1. 瞬时功率预测: 基于飞行状态预测当前功率
2. 多行程支持: 无人机可返回配送中心换电池后继续执行任务
3. 时间窗约束: 每个客户有到达时间窗 [a_i, b_i]
4. 载重约束: 无人机最大载重限制 Q

模型输入特征:
- height: 高度 [m]
- VS: 竖直速度 [m/s]
- GS: 地速 [m/s]
- wind_speed: 风速 [m/s]
- temperature: 温度 [°C]
- humidity: 湿度 [%]
- wind_angle: 风向夹角 [度]

使用 RLTS-NSGA-II 算法求解
"""

import numpy as np
import math
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pickle

# 尝试导入LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("[WARNING] LightGBM未安装，树模型将不可用")
    LIGHTGBM_AVAILABLE = False

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    print("[WARNING] PyTorch未安装，深度学习模型将不可用")
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


# ==================== 物理功率模型 ====================

class PhysicalPowerModel:
    """
    基于物理公式的瞬时功率模型
    
    P(q) = k * (W + m + q)^(3/2)
    """
    
    def __init__(self, drone_params: DroneParameters = None):
        self.params = drone_params or DroneParameters()
        self.k = self.params.k
        
    def predict_power(self, payload: float = 0.0) -> float:
        """
        预测瞬时功率 [W]
        
        P(q) = k * (W + m + q)^(3/2)
        """
        total_weight = self.params.W + self.params.m + payload
        return self.k * (total_weight ** 1.5)
    
    def energy_from_power(self, power: float, duration_seconds: float) -> float:
        """根据功率和时间计算能耗 [kWh]"""
        energy_wh = power * (duration_seconds / 3600.0)
        return energy_wh / 1000.0


# ==================== LightGBM树模型 ====================

class TreePowerModel:
    """
    基于LightGBM的瞬时功率预测模型
    
    输入: height, VS, GS, wind_speed, temperature, humidity, wind_angle, payload
    输出: 瞬时功率 [W]
    """
    
    def __init__(self, model_path: str = 'result/power_lgb_model.txt'):
        self.model_path = model_path
        self.model = None
        self.feature_names = ['height', 'VS', 'GS', 'wind_speed', 'temperature', 'humidity', 'wind_angle', 'payload']
        self._load_model()
    
    def _load_model(self):
        """加载训练好的LightGBM模型"""
        if not LIGHTGBM_AVAILABLE:
            print("[WARNING] LightGBM不可用")
            return
            
        try:
            if os.path.exists(self.model_path):
                self.model = lgb.Booster(model_file=self.model_path)
                print(f"[OK] 瞬时功率树模型加载成功: {self.model_path}")
            else:
                print(f"[WARNING] 模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 加载树模型失败: {e}")
    
    def predict_power(self, height: float, VS: float, GS: float, 
                     wind_speed: float, temperature: float, humidity: float,
                     wind_angle: float, payload: float = 0.0) -> float:
        """预测瞬时功率 [W]"""
        if self.model is None:
            # 后备估算
            return 4500.0 + 50.0 * GS + 100.0 * abs(VS) + 100.0 * payload
        
        features = np.array([[height, VS, GS, wind_speed, temperature, humidity, wind_angle, payload]])
        power = self.model.predict(features)[0]
        return max(0.0, power)
    
    def energy_from_power(self, power: float, duration_seconds: float) -> float:
        """根据功率和时间计算能耗 [kWh]"""
        energy_wh = power * (duration_seconds / 3600.0)
        return energy_wh / 1000.0


# ==================== PyTorch深度学习模型 ====================

class DeepPowerModel:
    """
    基于PyTorch的瞬时功率预测模型
    
    输入: height, VS, GS, wind_speed, temperature, humidity, wind_angle, payload
    输出: 瞬时功率 [W]
    """
    
    def __init__(self, model_path: str = 'result/power_pytorch_model.pth'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = ['height', 'VS', 'GS', 'wind_speed', 'temperature', 'humidity', 'wind_angle', 'payload']
        self._load_model()
    
    def _load_model(self):
        """加载训练好的PyTorch模型"""
        if not PYTORCH_AVAILABLE:
            print("[WARNING] PyTorch不可用")
            return
            
        try:
            if os.path.exists(self.model_path):
                # 定义网络结构
                class PowerNet(nn.Module):
                    def __init__(self, input_size=8, hidden_sizes=[64, 32], dropout_rate=0.1):
                        super(PowerNet, self).__init__()
                        layers = []
                        prev_size = input_size
                        for i, hidden_size in enumerate(hidden_sizes):
                            layers.append(nn.Linear(prev_size, hidden_size))
                            layers.append(nn.ReLU())
                            if i == 0:
                                layers.append(nn.Dropout(dropout_rate))
                            prev_size = hidden_size
                        layers.append(nn.Linear(prev_size, 1))
                        layers.append(nn.ReLU())
                        self.network = nn.Sequential(*layers)
                    
                    def forward(self, x):
                        return self.network(x)
                
                self.model = PowerNet(input_size=8)
                self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                self.model.eval()
                print(f"[OK] 瞬时功率深度学习模型加载成功: {self.model_path}")
                
                # 加载标准化器
                scaler_path = self.model_path.replace('.pth', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f"[OK] 标准化器加载成功: {scaler_path}")
            else:
                print(f"[WARNING] 模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 加载深度学习模型失败: {e}")
    
    def predict_power(self, height: float, VS: float, GS: float, 
                     wind_speed: float, temperature: float, humidity: float,
                     wind_angle: float, payload: float = 0.0) -> float:
        """预测瞬时功率 [W]"""
        if self.model is None:
            return 4500.0 + 50.0 * GS + 100.0 * abs(VS) + 100.0 * payload
        
        features = np.array([[height, VS, GS, wind_speed, temperature, humidity, wind_angle, payload]])
        
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features)
            power = self.model(input_tensor).item()
        
        return max(0.0, power)
    
    def energy_from_power(self, power: float, duration_seconds: float) -> float:
        """根据功率和时间计算能耗 [kWh]"""
        energy_wh = power * (duration_seconds / 3600.0)
        return energy_wh / 1000.0


# ==================== 线性回归模型 ====================

class LinearPowerModel:
    """
    基于线性回归的瞬时功率预测模型
    
    输入: height, VS, GS, wind_speed, temperature, humidity, wind_angle, payload
    输出: 瞬时功率 [W]
    """
    
    def __init__(self, model_path: str = 'result/power_linear_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_names = ['height', 'VS', 'GS', 'wind_speed', 'temperature', 'humidity', 'wind_angle', 'payload']
        self._load_model()
    
    def _load_model(self):
        """加载训练好的线性回归模型"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"[OK] 瞬时功率线性回归模型加载成功: {self.model_path}")
            else:
                print(f"[WARNING] 模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 加载线性回归模型失败: {e}")
    
    def predict_power(self, height: float, VS: float, GS: float, 
                     wind_speed: float, temperature: float, humidity: float,
                     wind_angle: float, payload: float = 0.0) -> float:
        """预测瞬时功率 [W]"""
        if self.model is None:
            return 4500.0 + 50.0 * GS + 100.0 * abs(VS) + 100.0 * payload
        
        features = np.array([[height, VS, GS, wind_speed, temperature, humidity, wind_angle, payload]])
        power = self.model.predict(features)[0]
        return max(0.0, power)
    
    def energy_from_power(self, power: float, duration_seconds: float) -> float:
        """根据功率和时间计算能耗 [kWh]"""
        energy_wh = power * (duration_seconds / 3600.0)
        return energy_wh / 1000.0


# ==================== 模型工厂函数 ====================

def create_power_model(model_type: str = "tree", instance: MTDRPInstance = None):
    """
    创建瞬时功率预测模型的工厂函数
    
    参数:
        model_type: 模型类型 ("physical", "tree", "deep", "linear")
        instance: MTDRP问题实例（仅物理模型需要）
        
    返回:
        瞬时功率预测模型实例
    """
    if model_type.lower() == "physical":
        drone_params = instance.drone_params if instance else DroneParameters()
        print("[OK] 使用PhysicalPowerModel (基于物理公式)")
        return PhysicalPowerModel(drone_params)
    
    elif model_type.lower() == "tree":
        print("[OK] 使用TreePowerModel (基于LightGBM)")
        return TreePowerModel()
    
    elif model_type.lower() == "deep":
        print("[OK] 使用DeepPowerModel (基于PyTorch)")
        return DeepPowerModel()
    
    elif model_type.lower() == "linear":
        print("[OK] 使用LinearPowerModel (基于线性回归)")
        return LinearPowerModel()
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持: 'physical', 'tree', 'deep', 'linear'")


# ==================== 能耗计算辅助函数 ====================

def calculate_trip_energy(power_model, trajectory_points: List[Dict], 
                         time_interval: float = 1.0) -> float:
    """
    根据轨迹点序列计算总能耗
    
    参数:
        power_model: 功率预测模型
        trajectory_points: 轨迹点列表，每个点包含 {height, VS, GS, wind_speed, temperature, humidity, wind_angle}
        time_interval: 采样时间间隔 [秒]
        
    返回:
        总能耗 [kWh]
    """
    total_energy = 0.0
    
    for point in trajectory_points:
        power = power_model.predict_power(
            height=point.get('height', 100.0),
            VS=point.get('VS', 0.0),
            GS=point.get('GS', 10.0),
            wind_speed=point.get('wind_speed', 2.0),
            temperature=point.get('temperature', 25.0),
            humidity=point.get('humidity', 60.0),
            wind_angle=point.get('wind_angle', 90.0)
        )
        energy = power_model.energy_from_power(power, time_interval)
        total_energy += energy
    
    return total_energy


def estimate_flight_energy(power_model, distance: float, speed: float = 10.0,
                          height: float = 100.0, wind_speed: float = 2.0,
                          temperature: float = 25.0, humidity: float = 60.0,
                          wind_angle: float = 90.0) -> float:
    """
    估算一段飞行的能耗（简化版，用于调度规划）
    
    参数:
        power_model: 功率预测模型
        distance: 飞行距离 [m]
        speed: 飞行速度 [m/s]
        其他参数: 环境条件
        
    返回:
        估算能耗 [kWh]
    """
    # 计算飞行时间
    flight_time = distance / speed  # 秒
    
    # 预测平均功率（假设匀速平飞）
    avg_power = power_model.predict_power(
        height=height,
        VS=0.0,  # 平飞
        GS=speed,
        wind_speed=wind_speed,
        temperature=temperature,
        humidity=humidity,
        wind_angle=wind_angle
    )
    
    # 计算能耗
    energy = power_model.energy_from_power(avg_power, flight_time)
    
    return energy
