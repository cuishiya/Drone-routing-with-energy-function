#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多行程无人机路径问题 (MTDRP) - 非线性能量消耗模型

基于论文: "Drone routing with energy function: Formulation and exact algorithm"

核心特点:
1. 非线性能量消耗函数: P(q) = k * (W + m + q)^(3/2)
2. 多行程支持: 无人机可返回配送中心换电池后继续执行任务
3. 时间窗约束: 每个客户有到达时间窗 [a_i, b_i]
4. 载重约束: 无人机最大载重限制 Q

使用 RLTS-NSGA-II 算法求解
"""

import numpy as np
import math
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# 尝试导入PyTorch，如果失败则使用替代方案
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    print("[WARNING] PyTorch未安装，深度学习模型将使用简化实现")
    PYTORCH_AVAILABLE = False


# ==================== 全局参数配置 ====================
NUM_DRONES: int = 12  # 无人机数量


@dataclass
class DroneParameters:
    """无人机物理参数 - 基于UAS04028624实际参数"""
    W: float = 36.0         # 无人机自重（含电池） [kg] - 根据实际数据
    m: float = 0.0          # 电池重量已包含在自重中 [kg]
    Q: float = 8.0          # 最大载重 [kg] - 保持原设定
    g: float = 9.81         # 重力加速度 [N/kg]
    rho: float = 1.204      # 空气密度 [kg/m^3]
    xi: float = 0.3848      # 旋翼圆盘面积 [m^2] - 根据实际数据
    h: int = 6              # 旋翼数量
    sigma: float = 1     # 电池能量容量 [kWh]
    speed: float = 10.0     # 飞行速度 [m/s]
    
    # 实际无人机规格参数 (UAS04028624)
    drone_id: str = "UAS04028624"  # 无人机ID
    
    # 计算得到的常数 k
    @property
    def k(self) -> float:
        """计算能量常数 k = sqrt(g^3 / (2 * rho * xi * h))"""
        return math.sqrt(self.g**3 / (2 * self.rho * self.xi * self.h))
    
    @property
    def k_prime(self) -> float:
        """k' 用于约束方程，包含单位转换 (从瓦特秒到千瓦时)"""
        # 转换因子: 1 kWh = 3.6e6 J = 3.6e6 W·s
        # k' = k / 3600000 (将 W·s 转换为 kWh)
        return self.k / 3600000.0


@dataclass
class Customer:
    """客户节点"""
    id: int                 # 客户ID
    x: float                # x坐标
    y: float                # y坐标
    demand: float           # 需求量/载重 [kg]
    earliest_time: float    # 最早服务时间 (t)
    latest_time: float      # 最晚服务时间 (l_i)
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
        
        # 所有节点: depot (index 0) + customers (index 1 to n-1)
        all_nodes = [(self.depot.x, self.depot.y)] + [(c.x, c.y) for c in self.customers]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = all_nodes[i][0] - all_nodes[j][0]
                    dy = all_nodes[i][1] - all_nodes[j][1]
                    dist = math.sqrt(dx**2 + dy**2)
                    self.distance_matrix[i, j] = dist
                    # 旅行时间 = 距离 / 速度 (转换为分钟)
                    self.travel_time_matrix[i, j] = (dist / self.drone_params.speed) / 60.0
    
    @property
    def num_customers(self) -> int:
        return len(self.customers)
    
    @property
    def num_nodes(self) -> int:
        """总节点数 (depot + customers)"""
        return len(self.customers) + 1


def load_instance(filepath: str) -> MTDRPInstance:
    """
    从文件加载MTDRP问题实例
    
    只从数据文件读取客户信息（坐标、需求、时间窗）和配送中心位置。
    无人机参数使用 DroneParameters 类中定义的默认值。
    无人机数量使用全局变量 NUM_DRONES。
        
    参数:
        filepath: 数据文件路径
        
    返回:
        MTDRPInstance 对象
    """
    # 使用预定义的无人机参数（不从文件读取）
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
        
        # 识别客户数据段
        if line.startswith('Customers_data'):
            in_customer_section = True
            continue
        
        # 遇到其他数据段则退出客户数据解析
        if line.endswith('_data') or line.startswith('Num_'):
            in_customer_section = False
            continue
        
        # 只解析客户数据
        if in_customer_section:
            if line.startswith('id'):  # 跳过表头
                continue
            parts = line.split()
            if len(parts) >= 7:
                cust_id = int(parts[0])
                if cust_id == 0:  # depot (配送中心)
                    depot = Depot(
                        id=0,
                        x=float(parts[4]),
                        y=float(parts[5])
                    )
                else:  # 客户节点
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


class NonlinearEnergyModel:
    """
    非线性能量消耗模型
    
    基于论文公式: P(q_ij) = k * (W + m + q_ij)^(3/2)
    
    其中:
    - k = sqrt(g^3 / (2 * rho * xi * h))
    - W: 机身重量
    - m: 电池重量
    - q_ij: 弧(i,j)上的载重
    """
    
    def __init__(self, drone_params: DroneParameters):
        self.params = drone_params
        self.k = drone_params.k
        self.k_prime = drone_params.k_prime
        
    def power(self, payload: float) -> float:
        """
        计算给定载重下的悬停功率 [W]
        
        P(q) = k * (W + m + q)^(3/2)
        
        参数:
            payload: 载重 [kg]
            
        返回:
            功率 [W]
        """
        total_weight = self.params.W + self.params.m + payload
        return self.k * (total_weight ** 1.5)
    
    def energy_consumption(self, payload: float, end_time: float, start_time: float) -> float:
        """
        计算飞行能量消耗 [kWh]
        
        使用精准公式: E = t * (v + q)^1.5 * sqrt(g^3 / (2 * rho * z * n))
        其中 t = end_time - start_time
        
        参数:
            payload: 载重 [kg]
            end_time: 结束时间 [s]
            start_time: 开始时间 [s]
            
        返回:
            能量消耗 [kWh]
        """
        # 获取飞行时长和载荷
        t = end_time - start_time  # 实际飞行时间
        q = payload
        
        # 计算能耗 (J): e = t * (W + q)^1.5 * sqrt(g^3 / (2 * rho * xi * h))
        # 使用DroneParameters中已定义的参数和常数k
        energy_j = t * ((self.params.W + q) ** 1.5) * self.k
        
        # 转换为 kWh (1 kWh = 3,600,000 J)
        energy_kwh = energy_j / 3600000
        
        return energy_kwh
    
    def energy_consumption_for_arc(self, payload: float, travel_time_min: float) -> float:
        """
        计算弧上的能量消耗 [kWh]
        
        参数:
            payload: 载重 [kg]
            travel_time_min: 旅行时间 [分钟]
            
        返回:
            能量消耗 [kWh]
        """
        power_w = self.power(payload)
        travel_time_s = travel_time_min * 60.0  # 转换为秒
        energy_j = power_w * travel_time_s
        energy_kwh = energy_j / 3600000.0
        return energy_kwh
    
    def max_range_with_payload(self, payload: float) -> float:
        """
        计算给定载重下的最大飞行距离 [m]
        
        参数:
            payload: 载重 [kg]
            
        返回:
            最大飞行距离 [m]
        """
        power_w = self.power(payload)
        max_energy_j = self.params.sigma * 3600000.0  # kWh 转 J
        max_time_s = max_energy_j / power_w
        max_distance = max_time_s * self.params.speed
        return max_distance


class TreeBasedEnergyModel:
    """
    基于LightGBM树模型的能量消耗模型
    
    该模型使用训练好的LightGBM模型来预测无人机的能耗，
    提供与NonlinearEnergyModel相同的接口以便于替换。
    """
    
    def __init__(self, model_path: str = 'result/drone_energy_lgb_model.txt',
                 wind_speed: float = 5.0, wind_angle: float = 90.0,
                 temperature: float = 25.0, humidity: float = 60.0):
        """
        初始化树模型能耗预测器
        
        参数:
            model_path: 训练好的LightGBM模型文件路径
            wind_speed: 默认风速 [m/s]
            wind_angle: 默认风向夹角 [度]
            temperature: 默认温度 [°C]
            humidity: 默认湿度 [%]
            
        注意: 载重(payload)和距离(distance)在调用预测方法时提供
        """
        self.model_path = model_path
        self.model = None
        
        # 环境参数
        self.default_wind_speed = wind_speed
        self.default_wind_angle = wind_angle
        self.default_temperature = temperature
        self.default_humidity = humidity
        
        self._load_model()
    
    def _load_model(self):
        """加载训练好的LightGBM模型"""
        try:
            if os.path.exists(self.model_path):
                self.model = lgb.Booster(model_file=self.model_path)
            else:
                print(f"警告: 模型文件不存在 {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.model = None
    
    def _predict_energy_consumption(self, distance: float, payload: float, 
                                   wind_speed: float = None, wind_angle: float = None,
                                   temperature: float = None, humidity: float = None) -> float:
        """
        使用树模型预测能耗
        
        参数:
            distance: 飞行距离 [m]
            payload: 载重 [kg]
            wind_speed: 风速 [m/s]
            wind_angle: 风向夹角 [度]
            temperature: 温度 [°C]
            humidity: 湿度 [%]
            
        返回:
            能量消耗 [kWh]
        """
        if self.model is None:
            raise ValueError("模型未加载，无法进行预测")
        
        # 使用默认值填充缺失的环境参数
        wind_speed = wind_speed if wind_speed is not None else self.default_wind_speed
        wind_angle = wind_angle if wind_angle is not None else self.default_wind_angle
        temperature = temperature if temperature is not None else self.default_temperature
        humidity = humidity if humidity is not None else self.default_humidity
        
        # 准备特征向量 [distance, payload, wind_speed, wind_angle, temperature, humidity]
        features = np.array([[distance, payload, wind_speed, wind_angle, temperature, humidity]])
        
        try:
            # 使用模型预测
            energy_kwh = self.model.predict(features)[0]
            return max(0.0, energy_kwh)  # 确保能耗非负
        except Exception as e:
            raise ValueError(f"树模型预测失败: {e}")
    
    def power(self, payload: float, speed: float = 10.0) -> float:
        """
        计算给定载重下的悬停功率 [W]
        
        注意: 树模型直接预测能耗，此方法通过估算得到功率
        
        参数:
            payload: 载重 [kg]
            speed: 飞行速度 [m/s]，默认10.0
            
        返回:
            功率 [W]
        """
        # 使用标准距离（1000m）和时间来估算功率
        standard_distance = 1000.0  # 1km
        travel_time_s = standard_distance / speed
        energy_kwh = self._predict_energy_consumption(standard_distance, payload)
        energy_j = energy_kwh * 3600000.0  # 转换为焦耳
        power_w = energy_j / travel_time_s
        return power_w
    
    def energy_consumption(self, payload: float, distance: float) -> float:
        """
        计算飞行能量消耗 [kWh]
        
        参数:
            payload: 载重 [kg]
            distance: 飞行距离 [m]
            
        返回:
            能量消耗 [kWh]
        """
        return self._predict_energy_consumption(distance, payload)
    
    def energy_consumption_for_arc(self, payload: float, travel_time_min: float, speed: float = 10.0) -> float:
        """
        计算弧上的能量消耗 [kWh]
        
        参数:
            payload: 载重 [kg]
            travel_time_min: 旅行时间 [分钟]
            speed: 飞行速度 [m/s]，默认10.0
            
        返回:
            能量消耗 [kWh]
        """
        # 根据时间和速度计算距离
        distance = (travel_time_min * 60.0) * speed  # 转换为米
        return self._predict_energy_consumption(distance, payload)
    
    def max_range_with_payload(self, payload: float, max_energy_kwh: float = 0.27) -> float:
        """
        计算给定载重下的最大飞行距离 [m]
        
        参数:
            payload: 载重 [kg]
            max_energy_kwh: 最大电池容量 [kWh]，默认0.27
            
        返回:
            最大飞行距离 [m]
        """
        # 使用二分搜索找到最大飞行距离
        min_distance = 0.0
        max_distance = 50000.0  # 50km作为上限
        tolerance = 10.0  # 10m精度
        
        while max_distance - min_distance > tolerance:
            mid_distance = (min_distance + max_distance) / 2.0
            predicted_energy = self._predict_energy_consumption(mid_distance, payload)
            
            if predicted_energy <= max_energy_kwh:
                min_distance = mid_distance
            else:
                max_distance = mid_distance
        
        return min_distance
    
    def set_environmental_conditions(self, wind_speed: float = None, wind_angle: float = None,
                                   temperature: float = None, humidity: float = None):
        """
        设置环境条件的默认值
        
        参数:
            wind_speed: 风速 [m/s]
            wind_angle: 风向夹角 [度]
            temperature: 温度 [°C]
            humidity: 湿度 [%]
        """
        if wind_speed is not None:
            self.default_wind_speed = wind_speed
        if wind_angle is not None:
            self.default_wind_angle = wind_angle
        if temperature is not None:
            self.default_temperature = temperature
        if humidity is not None:
            self.default_humidity = humidity


class LinearRegressionEnergyModel:
    """
    基于距离和载荷的线性回归能耗模型
    
    该模型仅使用距离和载荷两个特征进行线性回归预测，
    用于验证考虑气象因素的必要性。
    """
    
    def __init__(self, model_path: str = 'result/linear_regression_model.pkl'):
        """
        初始化线性回归能耗预测器
        
        参数:
            model_path: 训练好的线性回归模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self._load_model()
    
    def _load_model(self):
        """加载训练好的线性回归模型"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                print(f"[OK] 线性回归模型加载成功: {self.model_path}")
            else:
                print(f"[WARNING] 模型文件不存在: {self.model_path}")
                print("[INFO] 将使用默认线性回归模型")
                self._create_default_model()
        except Exception as e:
            print(f"[ERROR] 加载线性回归模型失败: {e}")
            print("[INFO] 将使用默认线性回归模型")
            self._create_default_model()
    
    def _create_default_model(self):
        """创建默认的线性回归模型（基于经验参数）"""
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
        # 基于物理模型的经验参数设置默认权重
        # E ≈ α * distance + β * payload + γ
        # 其中α、β、γ是根据物理模型估算的经验参数
        self.model.coef_ = np.array([0.00005, 0.01])  # [distance_coef, payload_coef]
        self.model.intercept_ = 0.05  # 基础能耗
        
        # 设置标准化器的参数（基于典型数据范围）
        self.scaler.mean_ = np.array([2500.0, 4.0])  # [distance_mean, payload_mean]
        self.scaler.scale_ = np.array([2000.0, 3.0])  # [distance_std, payload_std]
        
        print("[INFO] 使用默认线性回归参数")
    
    def _predict_energy_consumption(self, distance: float, payload: float) -> float:
        """
        使用线性回归模型预测能耗
        
        参数:
            distance: 飞行距离 [m]
            payload: 载重 [kg]
            
        返回:
            能量消耗 [kWh]
        """
        if self.model is None:
            raise ValueError("线性回归模型未加载，无法进行预测")
        
        # 准备特征向量 [distance, payload]
        features = np.array([[distance, payload]])
        
        try:
            # 标准化特征
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # 使用模型预测
            energy_kwh = self.model.predict(features_scaled)[0]
            return max(0.0, energy_kwh)  # 确保能耗非负
        except Exception as e:
            raise ValueError(f"线性回归模型预测失败: {e}")
    
    def power(self, payload: float, speed: float = 10.0) -> float:
        """
        计算给定载重下的悬停功率 [W]
        
        参数:
            payload: 载重 [kg]
            speed: 飞行速度 [m/s]，默认10.0
            
        返回:
            功率 [W]
        """
        # 使用标准距离（1000m）和时间来估算功率
        standard_distance = 1000.0  # 1km
        travel_time_s = standard_distance / speed
        energy_kwh = self._predict_energy_consumption(standard_distance, payload)
        energy_j = energy_kwh * 3600000.0  # 转换为焦耳
        power_w = energy_j / travel_time_s
        return power_w
    
    def energy_consumption(self, payload: float, distance: float) -> float:
        """
        计算飞行能量消耗 [kWh]
        
        参数:
            payload: 载重 [kg]
            distance: 飞行距离 [m]
            
        返回:
            能量消耗 [kWh]
        """
        return self._predict_energy_consumption(distance, payload)
    
    def energy_consumption_for_arc(self, payload: float, travel_time_min: float, speed: float = 10.0) -> float:
        """
        计算弧上的能量消耗 [kWh]
        
        参数:
            payload: 载重 [kg]
            travel_time_min: 旅行时间 [分钟]
            speed: 飞行速度 [m/s]，默认10.0
            
        返回:
            能量消耗 [kWh]
        """
        # 根据时间和速度计算距离
        distance = (travel_time_min * 60.0) * speed  # 转换为米
        return self._predict_energy_consumption(distance, payload)
    
    def max_range_with_payload(self, payload: float, max_energy_kwh: float = 0.27) -> float:
        """
        计算给定载重下的最大飞行距离 [m]
        
        参数:
            payload: 载重 [kg]
            max_energy_kwh: 最大电池容量 [kWh]，默认0.27
            
        返回:
            最大飞行距离 [m]
        """
        # 使用线性关系直接计算
        if self.model is None:
            return 10000.0  # 默认值
        
        try:
            # 对于线性模型: E = α * distance + β * payload + γ
            # 解出: distance = (E - β * payload - γ) / α
            coef_distance = self.model.coef_[0] if len(self.model.coef_) > 0 else 0.00005
            coef_payload = self.model.coef_[1] if len(self.model.coef_) > 1 else 0.01
            intercept = self.model.intercept_
            
            # 考虑标准化的影响
            if self.scaler is not None:
                # 反向计算原始特征空间的系数
                coef_distance = coef_distance / self.scaler.scale_[0]
                coef_payload = coef_payload / self.scaler.scale_[1]
                intercept = intercept + np.dot(self.model.coef_, self.scaler.mean_ / self.scaler.scale_)
            
            if coef_distance > 0:
                max_distance = (max_energy_kwh - coef_payload * payload - intercept) / coef_distance
                return max(0.0, max_distance)
            else:
                return 50000.0  # 如果系数异常，返回一个大值
        except Exception as e:
            print(f"[WARNING] 计算最大航程失败: {e}")
            return 10000.0


# PyTorch深度学习模型网络结构
if PYTORCH_AVAILABLE:
    class EnergyNet(nn.Module):
        """
        基于PyTorch的能耗预测神经网络
        """
        def __init__(self, input_size=6, hidden_sizes=[32, 16], dropout_rate=0.1):
            super(EnergyNet, self).__init__()
            
            layers = []
            prev_size = input_size
            
            # 隐藏层 - 简化为2层
            for i, hidden_size in enumerate(hidden_sizes):
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                # 只在第一层使用dropout，减少正则化强度
                if i == 0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            # 输出层
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.ReLU())  # 确保输出非负
            
            self.network = nn.Sequential(*layers)
            
            # 权重初始化
            self._initialize_weights()
        
        def _initialize_weights(self):
            """初始化网络权重"""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            return self.network(x)


class DeepLearningEnergyModel:
    """
    基于PyTorch深度学习的能耗模型
    
    使用全连接神经网络预测无人机能耗，
    输入特征包括距离、载荷、风速、风向夹角、温度、湿度。
    """
    
    def __init__(self, model_path: str = 'result/pytorch_energy_model.pth',
                 wind_speed: float = 5.0, wind_angle: float = 90.0,
                 temperature: float = 25.0, humidity: float = 60.0):
        """
        初始化深度学习能耗预测器
        
        参数:
            model_path: 训练好的PyTorch模型文件路径
            wind_speed: 默认风速 [m/s]
            wind_angle: 默认风向夹角 [度]
            temperature: 默认温度 [°C]
            humidity: 默认湿度 [%]
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        
        # 环境参数
        self.default_wind_speed = wind_speed
        self.default_wind_angle = wind_angle
        self.default_temperature = temperature
        self.default_humidity = humidity
        
        self._load_model()
    
    def _load_model(self):
        """加载训练好的PyTorch模型"""
        try:
            if PYTORCH_AVAILABLE and os.path.exists(self.model_path):
                self.model = EnergyNet()
                self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                self.model.eval()
                print(f"[OK] PyTorch深度学习模型加载成功: {self.model_path}")
                
                # 尝试加载标准化器
                scaler_path = self.model_path.replace('.pth', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f"[OK] 标准化器加载成功: {scaler_path}")
            else:
                if not PYTORCH_AVAILABLE:
                    print("[WARNING] PyTorch未安装，将创建简化深度学习模型")
                else:
                    print(f"[WARNING] 模型文件不存在: {self.model_path}")
                print("[INFO] 将创建默认深度学习模型")
                self._create_default_model()
        except Exception as e:
            print(f"[ERROR] 加载PyTorch深度学习模型失败: {e}")
            print("[INFO] 将创建默认深度学习模型")
            self._create_default_model()
    
    def _create_default_model(self):
        """创建默认的深度学习模型"""
        try:
            if PYTORCH_AVAILABLE:
                self.model = EnergyNet()
                self.model.eval()
                print("[INFO] 创建默认PyTorch深度学习模型")
            else:
                # 使用简化的模拟模型
                self.model = None
                print("[INFO] 创建简化深度学习模型（无PyTorch）")
            
            # 创建默认标准化器
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array([2500.0, 4.0, 5.0, 90.0, 25.0, 60.0])
            self.scaler.scale_ = np.array([2000.0, 3.0, 3.0, 45.0, 10.0, 20.0])
            
        except Exception as e:
            print(f"[ERROR] 创建默认深度学习模型失败: {e}")
            self.model = None
    
    def _predict_energy_consumption(self, distance: float, payload: float,
                                   wind_speed: float = None, wind_angle: float = None,
                                   temperature: float = None, humidity: float = None) -> float:
        """
        使用深度学习模型预测能耗
        
        参数:
            distance: 飞行距离 [m]
            payload: 载重 [kg]
            wind_speed: 风速 [m/s]
            wind_angle: 风向夹角 [度]
            temperature: 温度 [°C]
            humidity: 湿度 [%]
            
        返回:
            能量消耗 [kWh]
        """
        # 使用默认值填充缺失的环境参数
        wind_speed = wind_speed if wind_speed is not None else self.default_wind_speed
        wind_angle = wind_angle if wind_angle is not None else self.default_wind_angle
        temperature = temperature if temperature is not None else self.default_temperature
        humidity = humidity if humidity is not None else self.default_humidity
        
        # 准备特征向量 [distance, payload, wind_speed, wind_angle, temperature, humidity]
        features = np.array([[distance, payload, wind_speed, wind_angle, temperature, humidity]])
        
        try:
            if PYTORCH_AVAILABLE and self.model is not None:
                # 标准化特征
                if self.scaler is not None:
                    features_scaled = self.scaler.transform(features)
                else:
                    features_scaled = features
                
                # 转换为PyTorch张量
                input_tensor = torch.FloatTensor(features_scaled)
                
                # 使用模型预测
                with torch.no_grad():
                    energy_kwh = self.model(input_tensor).item()
                
                return max(0.0, energy_kwh)  # 确保能耗非负
            else:
                # 简化的预测逻辑（无PyTorch时）
                energy = 0.00005 * distance + 0.01 * payload + 0.05
                return max(0.0, energy)
                
        except Exception as e:
            raise ValueError(f"深度学习模型预测失败: {e}")
    
    def power(self, payload: float, speed: float = 10.0) -> float:
        """
        计算给定载重下的悬停功率 [W]
        
        参数:
            payload: 载重 [kg]
            speed: 飞行速度 [m/s]，默认10.0
            
        返回:
            功率 [W]
        """
        # 使用标准距离（1000m）和时间来估算功率
        standard_distance = 1000.0  # 1km
        travel_time_s = standard_distance / speed
        energy_kwh = self._predict_energy_consumption(standard_distance, payload)
        energy_j = energy_kwh * 3600000.0  # 转换为焦耳
        power_w = energy_j / travel_time_s
        return power_w
    
    def energy_consumption(self, payload: float, distance: float) -> float:
        """
        计算飞行能量消耗 [kWh]
        
        参数:
            payload: 载重 [kg]
            distance: 飞行距离 [m]
            
        返回:
            能量消耗 [kWh]
        """
        return self._predict_energy_consumption(distance, payload)
    
    def energy_consumption_for_arc(self, payload: float, travel_time_min: float, speed: float = 10.0) -> float:
        """
        计算弧上的能量消耗 [kWh]
        
        参数:
            payload: 载重 [kg]
            travel_time_min: 旅行时间 [分钟]
            speed: 飞行速度 [m/s]，默认10.0
            
        返回:
            能量消耗 [kWh]
        """
        # 根据时间和速度计算距离
        distance = (travel_time_min * 60.0) * speed  # 转换为米
        return self._predict_energy_consumption(distance, payload)
    
    def max_range_with_payload(self, payload: float, max_energy_kwh: float = 0.27) -> float:
        """
        计算给定载重下的最大飞行距离 [m]
        
        参数:
            payload: 载重 [kg]
            max_energy_kwh: 最大电池容量 [kWh]，默认0.27
            
        返回:
            最大飞行距离 [m]
        """
        # 使用二分搜索找到最大飞行距离
        min_distance = 0.0
        max_distance = 50000.0  # 50km作为上限
        tolerance = 10.0  # 10m精度
        
        while max_distance - min_distance > tolerance:
            mid_distance = (min_distance + max_distance) / 2.0
            predicted_energy = self._predict_energy_consumption(mid_distance, payload)
            
            if predicted_energy <= max_energy_kwh:
                min_distance = mid_distance
            else:
                max_distance = mid_distance
        
        return min_distance
    
    def set_environmental_conditions(self, wind_speed: float = None, wind_angle: float = None,
                                   temperature: float = None, humidity: float = None):
        """
        设置环境条件的默认值
        
        参数:
            wind_speed: 风速 [m/s]
            wind_angle: 风向夹角 [度]
            temperature: 温度 [°C]
            humidity: 湿度 [%]
        """
        if wind_speed is not None:
            self.default_wind_speed = wind_speed
        if wind_angle is not None:
            self.default_wind_angle = wind_angle
        if temperature is not None:
            self.default_temperature = temperature
        if humidity is not None:
            self.default_humidity = humidity


def create_energy_model(model_type: str = "tree", instance: 'MTDRPInstance' = None):
    """
    创建能耗模型的工厂函数
    
    参数:
        model_type: 模型类型 ("physical", "linear", "tree", "deep")
        instance: MTDRP问题实例（用于物理模型）
        
    返回:
        能耗模型实例
    """
    if model_type.lower() in ["physical", "nonlinear"]:  # 兼容旧的"nonlinear"标识
        if instance is None:
            raise ValueError("Physical Model需要MTDRPInstance参数")
        print("[OK] 使用Physical Model (基于物理公式的非线性模型)")
        return NonlinearEnergyModel(instance.drone_params)
    
    elif model_type.lower() == "linear":
        try:
            model = LinearRegressionEnergyModel()
            print("[OK] 使用LinearRegressionEnergyModel (基于距离和载荷的线性回归)")
            return model
        except Exception as e:
            print(f"[ERROR] LinearRegressionEnergyModel创建失败: {e}")
            raise
    
    elif model_type.lower() == "tree":
        try:
            model = TreeBasedEnergyModel()
            print("[OK] 使用TreeBasedEnergyModel (基于LightGBM的树模型)")
            return model
        except Exception as e:
            print(f"[ERROR] TreeBasedEnergyModel创建失败: {e}")
            raise
    
    elif model_type.lower() == "deep":
        try:
            model = DeepLearningEnergyModel()
            print("[OK] 使用DeepLearningEnergyModel (基于PyTorch的深度学习模型)")
            return model
        except Exception as e:
            print(f"[ERROR] DeepLearningEnergyModel创建失败: {e}")
            raise
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: 'physical', 'linear', 'tree', 'deep'")


