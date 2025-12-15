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


@dataclass
class DroneParameters:
    """无人机物理参数"""
    W: float = 1.5          # 机身重量 [kg]
    m: float = 1.5          # 电池重量 [kg]
    Q: float = 8.0          # 最大载重 [kg]
    g: float = 9.81         # 重力加速度 [N/kg]
    rho: float = 1.204      # 空气密度 [kg/m^3]
    xi: float = 0.0064      # 旋翼圆盘面积 [m^2]
    h: int = 6              # 旋翼数量
    sigma: float = 0.27     # 电池能量容量 [kWh] (Set A)
    speed: float = 10.0     # 飞行速度 [m/s] (假设值，可根据实际调整)
    
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
    
    参数:
        filepath: 数据文件路径
        
    返回:
        MTDRPInstance 对象
    """
    drone_params = DroneParameters()
    customers = []
    depot = Depot()
    num_drones = 12
    instance_name = os.path.basename(filepath)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 识别数据段
        if line.startswith('Drone_data'):
            section = 'drone'
            continue
        elif line.startswith('Battery_data'):
            section = 'battery'
            continue
        elif line.startswith('Customers_data'):
            section = 'customer'
            continue
        elif line.startswith('Num_drones'):
            parts = line.split()
            num_drones = int(parts[1])
            continue
        
        # 解析无人机参数
        if section == 'drone':
            parts = line.split()
            if len(parts) >= 2:
                param_name = parts[0]
                try:
                    param_value = float(parts[1])
                except ValueError:
                    continue
                if param_name == 'q_d':
                    # q_d 是总载重能力，Q 是最大有效载荷
                    drone_params.Q = 8.0  # 最大载重 8 kg
                elif param_name == 'W':
                    drone_params.W = param_value
                elif param_name == 'm':
                    drone_params.m = param_value
                elif param_name == 'g':
                    drone_params.g = param_value
                elif param_name == 'rho_d':
                    drone_params.rho = param_value
                elif param_name == 'xi_d':
                    drone_params.xi = param_value
                elif param_name == 'h_d':
                    drone_params.h = int(param_value)
        
        # 解析电池参数
        elif section == 'battery':
            parts = line.split()
            if len(parts) >= 2:
                param_name = parts[0]
                if param_name == 'max_energy_density':
                    drone_params.sigma = float(parts[1])
        
        # 解析客户数据
        elif section == 'customer':
            if line.startswith('id'):  # 跳过表头
                continue
            parts = line.split()
            if len(parts) >= 7:
                cust_id = int(parts[0])
                if cust_id == 0:  # depot
                    depot = Depot(
                        id=0,
                        x=float(parts[4]),
                        y=float(parts[5])
                    )
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
        num_drones=num_drones
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
    
    def energy_consumption(self, payload: float, distance: float) -> float:
        """
        计算飞行能量消耗 [kWh]
        
        E = P(q) * t = P(q) * (d / v)
        
        参数:
            payload: 载重 [kg]
            distance: 飞行距离 [m]
            
        返回:
            能量消耗 [kWh]
        """
        power_w = self.power(payload)
        travel_time_s = distance / self.params.speed  # 秒
        energy_j = power_w * travel_time_s  # 焦耳
        energy_kwh = energy_j / 3600000.0  # 转换为 kWh
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


class MTDRPSolution:
    """
    MTDRP解的表示
    
    一个解包含多架无人机的多个行程
    每个行程是一个客户访问序列
    """
    
    def __init__(self, instance: MTDRPInstance):
        self.instance = instance
        # routes[drone_id] = [trip1, trip2, ...] 
        # 每个trip是客户ID列表 (不包含depot)
        self.routes: Dict[int, List[List[int]]] = {k: [] for k in range(instance.num_drones)}
        
    def add_trip(self, drone_id: int, customers: List[int]):
        """为无人机添加一个行程"""
        if customers:
            self.routes[drone_id].append(customers)
    
    def get_all_trips(self) -> List[Tuple[int, int, List[int]]]:
        """获取所有行程 [(drone_id, trip_idx, customers), ...]"""
        trips = []
        for drone_id, drone_trips in self.routes.items():
            for trip_idx, customers in enumerate(drone_trips):
                trips.append((drone_id, trip_idx, customers))
        return trips
    
    def get_visited_customers(self) -> set:
        """获取所有被访问的客户"""
        visited = set()
        for drone_trips in self.routes.values():
            for trip in drone_trips:
                visited.update(trip)
        return visited
    
    def is_complete(self) -> bool:
        """检查是否所有客户都被访问"""
        visited = self.get_visited_customers()
        required = set(c.id for c in self.instance.customers)
        return visited == required


class MTDRPEvaluator:
    """
    MTDRP解的评估器
    
    计算目标函数值和约束违反情况
    """
    
    def __init__(self, instance: MTDRPInstance):
        self.instance = instance
        self.energy_model = NonlinearEnergyModel(instance.drone_params)
        
        # 创建客户ID到索引的映射
        self.customer_map = {c.id: c for c in instance.customers}
        
    def evaluate(self, solution: MTDRPSolution) -> Tuple[float, float, Dict]:
        """
        评估解的质量
        
        返回:
            (total_cost, total_penalty, details)
            - total_cost: 总成本 (目标函数值)
            - total_penalty: 约束违反惩罚
            - details: 详细信息字典
        """
        total_energy = 0.0
        total_distance = 0.0
        total_delay = 0.0
        penalties = {
            'unvisited': 0.0,
            'capacity': 0.0,
            'energy': 0.0,
            'time_window': 0.0,
            'duplicate': 0.0
        }
        
        visited_customers = set()
        
        # 评估每架无人机的每个行程
        for drone_id, drone_trips in solution.routes.items():
            for trip in drone_trips:
                trip_result = self._evaluate_trip(trip, visited_customers)
                total_energy += trip_result['energy']
                total_distance += trip_result['distance']
                total_delay += trip_result['delay']
                
                for key in penalties:
                    penalties[key] += trip_result['penalties'].get(key, 0.0)
                
                visited_customers.update(trip)
        
        # 检查未访问的客户
        required = set(c.id for c in self.instance.customers)
        unvisited = required - visited_customers
        penalties['unvisited'] = len(unvisited) * 10000.0  # 大惩罚
        
        total_penalty = sum(penalties.values())
        
        # 目标函数: min Σ(c_ij * x_ij + δ * e_ij)
        # 这里 c_ij 可以是距离成本，δ 是能量成本系数
        delta = 1.0  # 能量成本系数
        total_cost = total_distance + delta * total_energy * 1000  # 能量转换为合适的单位
        
        details = {
            'total_energy': total_energy,
            'total_distance': total_distance,
            'total_delay': total_delay,
            'penalties': penalties,
            'num_trips': sum(len(trips) for trips in solution.routes.values()),
            'visited_customers': len(visited_customers),
            'unvisited_customers': len(unvisited)
        }
        
        return total_cost, total_penalty, details
    
    def _evaluate_trip(self, trip: List[int], already_visited: set) -> Dict:
        """评估单个行程"""
        result = {
            'energy': 0.0,
            'distance': 0.0,
            'delay': 0.0,
            'penalties': {}
        }
        
        if not trip:
            return result
        
        # 检查重复访问
        for cust_id in trip:
            if cust_id in already_visited:
                result['penalties']['duplicate'] = result['penalties'].get('duplicate', 0) + 5000.0
        
        # 计算行程的载重、能量、时间
        # 从depot出发，访问所有客户，返回depot
        
        # 计算初始载重 (所有客户需求之和)
        total_demand = sum(self.customer_map[cid].demand for cid in trip if cid in self.customer_map)
        
        # 检查载重约束
        if total_demand > self.instance.drone_params.Q:
            result['penalties']['capacity'] = (total_demand - self.instance.drone_params.Q) * 1000.0
        
        # 模拟行程
        current_node = 0  # depot
        current_time = 0.0
        current_payload = total_demand
        trip_energy = 0.0
        trip_distance = 0.0
        
        for cust_id in trip:
            if cust_id not in self.customer_map:
                continue
                
            customer = self.customer_map[cust_id]
            cust_idx = cust_id  # 客户在矩阵中的索引
            
            # 计算到客户的距离和时间
            dist = self.instance.distance_matrix[current_node, cust_idx]
            travel_time = self.instance.travel_time_matrix[current_node, cust_idx]
            
            # 计算能量消耗
            energy = self.energy_model.energy_consumption(current_payload, dist)
            trip_energy += energy
            trip_distance += dist
            
            # 更新时间
            arrival_time = current_time + travel_time
            
            # 检查时间窗约束
            if arrival_time < customer.earliest_time:
                # 等待到最早服务时间
                current_time = customer.earliest_time + customer.service_time
            elif arrival_time > customer.latest_time:
                # 违反时间窗
                delay = arrival_time - customer.latest_time
                result['delay'] += delay
                result['penalties']['time_window'] = result['penalties'].get('time_window', 0) + delay * 100.0
                current_time = arrival_time + customer.service_time
            else:
                current_time = arrival_time + customer.service_time
            
            # 卸货后更新载重
            current_payload -= customer.demand
            current_node = cust_idx
        
        # 返回depot
        dist_to_depot = self.instance.distance_matrix[current_node, 0]
        energy_to_depot = self.energy_model.energy_consumption(current_payload, dist_to_depot)
        trip_energy += energy_to_depot
        trip_distance += dist_to_depot
        
        # 检查电池能量约束
        if trip_energy > self.instance.drone_params.sigma:
            result['penalties']['energy'] = (trip_energy - self.instance.drone_params.sigma) * 10000.0
        
        result['energy'] = trip_energy
        result['distance'] = trip_distance
        
        return result


def print_instance_info(instance: MTDRPInstance):
    """打印问题实例信息"""
    print(f"\n{'='*60}")
    print(f"MTDRP 问题实例: {instance.name}")
    print(f"{'='*60}")
    print(f"配送中心位置: ({instance.depot.x}, {instance.depot.y})")
    print(f"客户数量: {instance.num_customers}")
    print(f"无人机数量: {instance.num_drones}")
    print(f"\n无人机参数:")
    print(f"  机身重量 W: {instance.drone_params.W} kg")
    print(f"  电池重量 m: {instance.drone_params.m} kg")
    print(f"  最大载重 Q: {instance.drone_params.Q} kg")
    print(f"  电池容量 σ: {instance.drone_params.sigma} kWh")
    print(f"  能量常数 k: {instance.drone_params.k:.4f}")
    print(f"\n客户需求统计:")
    demands = [c.demand for c in instance.customers]
    print(f"  最小需求: {min(demands):.2f} kg")
    print(f"  最大需求: {max(demands):.2f} kg")
    print(f"  平均需求: {np.mean(demands):.2f} kg")
    print(f"  总需求: {sum(demands):.2f} kg")
    
    # 计算能量模型示例
    energy_model = NonlinearEnergyModel(instance.drone_params)
    print(f"\n能量消耗示例 (飞行1km):")
    for payload in [0.0, 0.5, 1.0, 1.5]:
        energy = energy_model.energy_consumption(payload, 1000.0)
        print(f"  载重 {payload} kg: {energy*1000:.4f} Wh")


# 测试代码
if __name__ == "__main__":
    # 加载测试实例
    test_file = "dataset/instances/200/bccl1_ud_m200.dat"
    
    if os.path.exists(test_file):
        instance = load_instance(test_file)
        print_instance_info(instance)
        
        # 创建一个简单的测试解
        solution = MTDRPSolution(instance)
        
        # 简单贪心: 按时间窗排序，依次分配给无人机
        sorted_customers = sorted(instance.customers, key=lambda c: c.earliest_time)
        
        current_drone = 0
        current_trip = []
        current_load = 0.0
        
        for customer in sorted_customers[:20]:  # 只测试前20个客户
            if current_load + customer.demand <= instance.drone_params.Q:
                current_trip.append(customer.id)
                current_load += customer.demand
            else:
                if current_trip:
                    solution.add_trip(current_drone, current_trip)
                current_drone = (current_drone + 1) % instance.num_drones
                current_trip = [customer.id]
                current_load = customer.demand
        
        if current_trip:
            solution.add_trip(current_drone, current_trip)
        
        # 评估解
        evaluator = MTDRPEvaluator(instance)
        cost, penalty, details = evaluator.evaluate(solution)
        
        print(f"\n{'='*60}")
        print("测试解评估结果:")
        print(f"{'='*60}")
        print(f"总成本: {cost:.2f}")
        print(f"总惩罚: {penalty:.2f}")
        print(f"总能耗: {details['total_energy']*1000:.2f} Wh")
        print(f"总距离: {details['total_distance']:.2f} m")
        print(f"行程数: {details['num_trips']}")
        print(f"已访问客户: {details['visited_customers']}")
        print(f"未访问客户: {details['unvisited_customers']}")
    else:
        print(f"测试文件不存在: {test_file}")
