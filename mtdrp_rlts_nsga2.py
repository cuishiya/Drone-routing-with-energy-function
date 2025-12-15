#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTDRP问题的RLTS-NSGA-II求解器

基于论文: "Drone routing with energy function: Formulation and exact algorithm"
使用改进的RLTS-NSGA-II算法求解多行程无人机路径问题

算法特点:
1. Q-learning自适应参数调节
2. 基于关键路径的禁忌搜索局部优化
3. 多目标优化: 最小化总成本 + 最小化总能耗
"""

import pygmo as pg
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import math
from collections import deque
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from mtdrp_energy_model import (
    MTDRPInstance, MTDRPSolution, MTDRPEvaluator,
    NonlinearEnergyModel, load_instance, print_instance_info,
    Customer, DroneParameters
)


class QLearningMTDRPController:
    """
    Q-learning参数控制器 (适配MTDRP问题)
    """
    
    def __init__(self, max_generations: int):
        self.max_generations = max_generations
        self.num_states = 20
        
        # 动作空间
        self.cr_actions = [0.60, 0.70, 0.80, 0.85, 0.90]
        self.m_actions = [0.01, 0.05, 0.10, 0.15, 0.20]
        
        # Q表
        self.q_table_cr = np.zeros((self.num_states, len(self.cr_actions)))
        self.q_table_m = np.zeros((self.num_states, len(self.m_actions)))
        
        # Q-learning参数
        self.alpha = 0.8
        self.gamma = 0.6
        self.epsilon_start = 0.7
        self.epsilon_end = 0.05
        
        # 状态记录
        self.current_state = 0
        self.current_cr_action = 0
        self.current_m_action = 0
        self.previous_f_value = None
        self.first_gen_avg_fitness = None
        self.first_gen_best_fitness = None
    
    def calculate_population_evaluation(self, population_fitness) -> float:
        """计算种群综合评价值F"""
        try:
            if population_fitness is None or len(population_fitness) == 0:
                return 1.0
            
            fitness_array = np.array(population_fitness)
            if fitness_array.ndim == 1:
                fitness_array = fitness_array.reshape(-1, 1)
            
            # 综合分 = 目标1 + 目标2
            comprehensive_scores = np.sum(fitness_array, axis=1)
            
            current_avg = float(np.mean(comprehensive_scores))
            current_best = float(np.min(comprehensive_scores))
            
            if self.first_gen_avg_fitness is None:
                self.first_gen_avg_fitness = current_avg
                self.first_gen_best_fitness = current_best
                return 1.0
            
            avg_ratio = current_avg / self.first_gen_avg_fitness if self.first_gen_avg_fitness > 0 else 1.0
            best_ratio = current_best / self.first_gen_best_fitness if self.first_gen_best_fitness > 0 else 1.0
            
            F_raw = 0.5 * avg_ratio + 0.5 * best_ratio
            F_normalized = 1.0 / (1.0 + np.exp(-(F_raw - 1.0) * 4.0))
            
            return max(0.0, min(1.0, float(F_normalized)))
            
        except Exception as e:
            print(f"计算种群评价值时出错: {e}")
            return 1.0
    
    def get_state_from_f_value(self, f_value: float) -> int:
        """将F值离散化为状态"""
        f_value = max(0.0, min(1.0, float(f_value)))
        state = int(f_value * 20)
        return max(0, min(19, state))
    
    def get_epsilon(self, current_generation: int) -> float:
        """计算当前ε值"""
        epsilon = self.epsilon_start - current_generation * (
            (self.epsilon_start - self.epsilon_end) / self.max_generations)
        return max(epsilon, self.epsilon_end)
    
    def select_actions(self, state: int, current_generation: int) -> Tuple[int, int]:
        """使用ε-greedy策略选择动作"""
        state = max(0, min(self.num_states - 1, int(state)))
        epsilon = self.get_epsilon(current_generation)
        
        if random.random() < epsilon:
            cr_action = random.randint(0, len(self.cr_actions) - 1)
        else:
            cr_action = int(np.argmax(self.q_table_cr[state]))
        
        if random.random() < epsilon:
            m_action = random.randint(0, len(self.m_actions) - 1)
        else:
            m_action = int(np.argmax(self.q_table_m[state]))
        
        return cr_action, m_action
    
    def get_parameters(self, population_fitness, current_generation: int) -> Tuple[float, float]:
        """获取当前代的交叉率和变异率"""
        current_f_value = self.calculate_population_evaluation(population_fitness)
        self.current_state = self.get_state_from_f_value(current_f_value)
        
        self.current_cr_action, self.current_m_action = self.select_actions(
            self.current_state, current_generation)
        
        cr = self.cr_actions[self.current_cr_action]
        m = self.m_actions[self.current_m_action]
        
        if self.previous_f_value is not None:
            self._update_q_tables(current_f_value)
        
        self.previous_f_value = current_f_value
        return cr, m
    
    def _update_q_tables(self, new_f_value: float):
        """更新Q表"""
        reward = 1.0 if new_f_value < self.previous_f_value else -1.0
        new_state = self.get_state_from_f_value(new_f_value)
        
        # 更新交叉率Q表
        old_q_cr = self.q_table_cr[self.current_state, self.current_cr_action]
        max_future_q_cr = np.max(self.q_table_cr[new_state])
        self.q_table_cr[self.current_state, self.current_cr_action] = \
            old_q_cr + self.alpha * (reward + self.gamma * max_future_q_cr - old_q_cr)
        
        # 更新变异率Q表
        old_q_m = self.q_table_m[self.current_state, self.current_m_action]
        max_future_q_m = np.max(self.q_table_m[new_state])
        self.q_table_m[self.current_state, self.current_m_action] = \
            old_q_m + self.alpha * (reward + self.gamma * max_future_q_m - old_q_m)


class TabuSearchMTDRP:
    """
    MTDRP问题的禁忌搜索管理器
    """
    
    def __init__(self, tabu_tenure: int = 7, max_moves: int = 5):
        self.tabu_tenure = tabu_tenure
        self.max_moves = max_moves
        self.tabu_list = deque(maxlen=tabu_tenure)
    
    def local_search(self, individual: np.ndarray, problem: 'MTDRPProblem', 
                     max_iterations: int = 3) -> np.ndarray:
        """
        基于禁忌搜索的局部优化
        """
        best_individual = individual.copy()
        best_fitness = problem.fitness(individual)
        current_individual = individual.copy()
        
        for iteration in range(max_iterations):
            # 生成邻域移动
            moves = self._generate_moves(current_individual, problem)
            
            if not moves:
                break
            
            # 评估移动
            best_move = None
            best_move_fitness = None
            best_move_individual = None
            
            for move in moves[:self.max_moves]:
                if self._is_tabu(move):
                    continue
                
                new_individual = self._apply_move(move, current_individual, problem)
                new_fitness = problem.fitness(new_individual)
                
                if best_move_fitness is None or self._is_better(new_fitness, best_move_fitness):
                    best_move = move
                    best_move_fitness = new_fitness
                    best_move_individual = new_individual
            
            if best_move is not None:
                current_individual = best_move_individual
                self.tabu_list.append(self._get_tabu_key(best_move))
                
                if self._is_better(best_move_fitness, best_fitness):
                    best_individual = best_move_individual.copy()
                    best_fitness = best_move_fitness
            else:
                break
        
        return best_individual
    
    def _generate_moves(self, individual: np.ndarray, problem: 'MTDRPProblem') -> List[Dict]:
        """生成邻域移动"""
        moves = []
        n_customers = problem.instance.num_customers
        
        # 移动类型1: 交换两个客户的位置
        for i in range(min(20, n_customers)):
            for j in range(i + 1, min(20, n_customers)):
                moves.append({
                    'type': 'swap',
                    'i': i,
                    'j': j
                })
        
        # 移动类型2: 重新分配客户到不同无人机
        for i in range(min(10, n_customers)):
            for k in range(problem.instance.num_drones):
                moves.append({
                    'type': 'reassign',
                    'customer_idx': i,
                    'new_drone': k
                })
        
        # 移动类型3: 2-opt改进
        for i in range(min(15, n_customers - 2)):
            for j in range(i + 2, min(i + 5, n_customers)):
                moves.append({
                    'type': '2opt',
                    'i': i,
                    'j': j
                })
        
        random.shuffle(moves)
        return moves
    
    def _apply_move(self, move: Dict, individual: np.ndarray, 
                    problem: 'MTDRPProblem') -> np.ndarray:
        """应用移动操作"""
        new_individual = individual.copy()
        n_customers = problem.instance.num_customers
        
        if move['type'] == 'swap':
            i, j = move['i'], move['j']
            if i < n_customers and j < n_customers:
                # 交换访问顺序
                new_individual[i], new_individual[j] = new_individual[j], new_individual[i]
                
        elif move['type'] == 'reassign':
            idx = move['customer_idx']
            new_drone = move['new_drone']
            if idx < n_customers:
                # 修改无人机分配
                drone_idx = n_customers + idx
                if drone_idx < len(new_individual):
                    new_individual[drone_idx] = new_drone
                    
        elif move['type'] == '2opt':
            i, j = move['i'], move['j']
            if j < n_customers:
                # 反转子序列
                new_individual[i:j+1] = new_individual[i:j+1][::-1]
        
        return new_individual
    
    def _is_tabu(self, move: Dict) -> bool:
        """检查移动是否在禁忌表中"""
        key = self._get_tabu_key(move)
        return key in self.tabu_list
    
    def _get_tabu_key(self, move: Dict) -> str:
        """生成移动的禁忌键"""
        if move['type'] == 'swap':
            return f"swap_{move['i']}_{move['j']}"
        elif move['type'] == 'reassign':
            return f"reassign_{move['customer_idx']}_{move['new_drone']}"
        elif move['type'] == '2opt':
            return f"2opt_{move['i']}_{move['j']}"
        return str(move)
    
    def _is_better(self, fitness1, fitness2) -> bool:
        """比较两个适应度值"""
        if fitness2 is None:
            return True
        return sum(fitness1) < sum(fitness2)


class MTDRPProblem:
    """
    MTDRP问题的PyGMO封装类
    
    编码方案:
    [customer_order[0..n-1], drone_assignment[0..n-1], trip_assignment[0..n-1]]
    
    - customer_order: 客户访问优先级 (用于构建路径)
    - drone_assignment: 每个客户分配给哪架无人机
    - trip_assignment: 每个客户属于该无人机的第几个行程
    """
    
    def __init__(self, instance: MTDRPInstance):
        self.instance = instance
        self.evaluator = MTDRPEvaluator(instance)
        self.energy_model = NonlinearEnergyModel(instance.drone_params)
        
        self.n_customers = instance.num_customers
        self.n_drones = instance.num_drones
        self.max_trips_per_drone = 5  # 每架无人机最多行程数
        
        # 编码维度
        self.dimensions = self.n_customers * 3  # order + drone + trip
        
        # 边界设置
        self._setup_bounds()
        
        print(f"MTDRP问题初始化:")
        print(f"  客户数: {self.n_customers}")
        print(f"  无人机数: {self.n_drones}")
        print(f"  决策变量维度: {self.dimensions}")
    
    def _setup_bounds(self):
        """设置变量边界"""
        self.lower_bounds = []
        self.upper_bounds = []
        
        # customer_order: [0, n_customers-1] 的排列优先级
        for _ in range(self.n_customers):
            self.lower_bounds.append(0)
            self.upper_bounds.append(self.n_customers - 1)
        
        # drone_assignment: [0, n_drones-1]
        for _ in range(self.n_customers):
            self.lower_bounds.append(0)
            self.upper_bounds.append(self.n_drones - 1)
        
        # trip_assignment: [0, max_trips-1]
        for _ in range(self.n_customers):
            self.lower_bounds.append(0)
            self.upper_bounds.append(self.max_trips_per_drone - 1)
    
    def get_bounds(self):
        return (self.lower_bounds, self.upper_bounds)
    
    def get_nobj(self):
        """目标函数数量: 总成本 + 总能耗"""
        return 2
    
    def get_nec(self):
        return 0
    
    def get_nic(self):
        return 0
    
    def get_nix(self):
        return self.dimensions
    
    def _decode_solution(self, x: np.ndarray) -> MTDRPSolution:
        """
        解码遗传个体为MTDRP解
        """
        solution = MTDRPSolution(self.instance)
        
        # 提取编码段
        order_genes = x[:self.n_customers]
        drone_genes = x[self.n_customers:2*self.n_customers]
        trip_genes = x[2*self.n_customers:3*self.n_customers]
        
        # 按优先级排序客户
        customer_ids = [c.id for c in self.instance.customers]
        sorted_indices = np.argsort(order_genes)
        
        # 构建每架无人机每个行程的客户列表
        # routes[drone][trip] = [customer_ids]
        routes = {k: {t: [] for t in range(self.max_trips_per_drone)} 
                  for k in range(self.n_drones)}
        
        for idx in sorted_indices:
            cust_id = customer_ids[idx]
            drone = int(drone_genes[idx]) % self.n_drones
            trip = int(trip_genes[idx]) % self.max_trips_per_drone
            routes[drone][trip].append(cust_id)
        
        # 验证并修复每个行程的可行性
        for drone in range(self.n_drones):
            for trip in range(self.max_trips_per_drone):
                customers = routes[drone][trip]
                if customers:
                    # 检查载重约束，必要时拆分行程
                    feasible_trips = self._split_by_capacity(customers)
                    for ft in feasible_trips:
                        solution.add_trip(drone, ft)
        
        return solution
    
    def _split_by_capacity(self, customers: List[int]) -> List[List[int]]:
        """按载重约束拆分客户列表"""
        trips = []
        current_trip = []
        current_load = 0.0
        max_load = self.instance.drone_params.Q
        
        for cust_id in customers:
            customer = next((c for c in self.instance.customers if c.id == cust_id), None)
            if customer is None:
                continue
            
            if current_load + customer.demand <= max_load:
                current_trip.append(cust_id)
                current_load += customer.demand
            else:
                if current_trip:
                    trips.append(current_trip)
                current_trip = [cust_id]
                current_load = customer.demand
        
        if current_trip:
            trips.append(current_trip)
        
        return trips
    
    def fitness(self, x) -> List[float]:
        """
        计算适应度函数 (多目标)
        
        目标1: 总成本 (距离 + 能量成本)
        目标2: 总能耗
        """
        try:
            solution = self._decode_solution(np.array(x))
            cost, penalty, details = self.evaluator.evaluate(solution)
            
            # 目标1: 总成本 + 惩罚
            obj1 = cost + penalty
            
            # 目标2: 总能耗 + 惩罚
            obj2 = details['total_energy'] * 1000 + penalty  # 转换为Wh
            
            return [obj1, obj2]
            
        except Exception as e:
            print(f"适应度计算错误: {e}")
            return [1e10, 1e10]


def solve_mtdrp_rlts_nsga2(instance: MTDRPInstance,
                           population_size: int = 100,
                           generations: int = 200,
                           tabu_frequency: int = 10,
                           tabu_intensity: float = 0.3,
                           enable_qlearning: bool = True,
                           verbose: bool = True) -> Dict:
    """
    使用RLTS-NSGA-II算法求解MTDRP问题
    
    参数:
        instance: MTDRP问题实例
        population_size: 种群大小
        generations: 进化代数
        tabu_frequency: 禁忌搜索频率
        tabu_intensity: 禁忌搜索强度
        enable_qlearning: 是否启用Q-learning
        verbose: 是否显示详细信息
        
    返回:
        结果字典
    """
    if verbose:
        print(f"\n{'='*60}")
        print("RLTS-NSGA-II 求解 MTDRP 问题")
        print(f"{'='*60}")
    
    # 确保种群大小符合要求
    if population_size < 8:
        population_size = 8
    if population_size % 4 != 0:
        population_size = ((population_size // 4) + 1) * 4
    
    # 创建问题实例
    problem = MTDRPProblem(instance)
    pg_problem = pg.problem(problem)
    
    # 初始化Q-learning控制器
    qlearning = QLearningMTDRPController(generations) if enable_qlearning else None
    
    # 初始化禁忌搜索
    tabu_search = TabuSearchMTDRP(tabu_tenure=7, max_moves=5)
    
    # 初始化NSGA-II算法
    initial_cr = 0.75
    initial_m = 15.0 / problem.dimensions
    
    algo_obj = pg.nsga2(
        gen=1,
        cr=initial_cr,
        eta_c=8,
        m=initial_m,
        eta_m=5
    )
    algo = pg.algorithm(algo_obj)
    
    # 创建初始种群
    pop = pg.population(pg_problem, population_size)
    
    if verbose:
        print(f"种群大小: {population_size}")
        print(f"进化代数: {generations}")
        print(f"Q-learning: {'启用' if enable_qlearning else '禁用'}")
        print(f"禁忌搜索频率: 每{tabu_frequency}代")
        print(f"{'='*60}")
    
    # 进化数据记录
    evolution_data = {
        'generations': [],
        'min_cost': [],
        'avg_cost': [],
        'min_energy': [],
        'avg_energy': [],
        'pareto_count': []
    }
    
    # 进化循环
    for gen in range(generations):
        # Q-learning参数调节
        if qlearning is not None:
            current_fitness = pop.get_f()
            cr, m = qlearning.get_parameters(current_fitness, gen)
            
            algo_obj = pg.nsga2(gen=1, cr=cr, eta_c=8, m=m, eta_m=5)
            algo = pg.algorithm(algo_obj)
        
        # 进化一代
        pop = algo.evolve(pop)
        
        # 禁忌搜索改进
        if (gen + 1) % tabu_frequency == 0 and gen < generations - 1:
            individuals = pop.get_x()
            fitness_values = pop.get_f()
            
            # 选择前X%个体进行改进
            num_improve = max(1, int(len(individuals) * tabu_intensity))
            scores = [f[0] + f[1] for f in fitness_values]
            sorted_indices = np.argsort(scores)
            
            improved_count = 0
            for i in range(num_improve):
                idx = sorted_indices[i]
                original = individuals[idx]
                improved = tabu_search.local_search(original, problem, max_iterations=3)
                
                original_fitness = problem.fitness(original)
                improved_fitness = problem.fitness(improved)
                
                if sum(improved_fitness) < sum(original_fitness):
                    pop.set_x(idx, improved)
                    improved_count += 1
            
            if verbose and improved_count > 0:
                print(f"  第{gen+1}代: 禁忌搜索改进了 {improved_count} 个个体")
        
        # 记录进化数据
        fitness_values = pop.get_f()
        costs = fitness_values[:, 0]
        energies = fitness_values[:, 1]
        
        # 计算帕累托前沿
        pareto_indices = pg.non_dominated_front_2d(fitness_values)
        
        evolution_data['generations'].append(gen + 1)
        evolution_data['min_cost'].append(np.min(costs))
        evolution_data['avg_cost'].append(np.mean(costs))
        evolution_data['min_energy'].append(np.min(energies))
        evolution_data['avg_energy'].append(np.mean(energies))
        evolution_data['pareto_count'].append(len(pareto_indices))
        
        # 打印进度
        if verbose and (gen + 1) % 10 == 0:
            print(f"第{gen+1:3d}代 | "
                  f"帕累托解: {len(pareto_indices):2d} | "
                  f"最小成本: {np.min(costs):.1f} | "
                  f"最小能耗: {np.min(energies):.1f} Wh")
    
    # 提取最终帕累托前沿
    final_fitness = pop.get_f()
    final_individuals = pop.get_x()
    pareto_indices = pg.non_dominated_front_2d(final_fitness)
    
    pareto_front = []
    for idx in pareto_indices:
        solution = problem._decode_solution(final_individuals[idx])
        cost, penalty, details = problem.evaluator.evaluate(solution)
        
        pareto_front.append({
            'individual': final_individuals[idx],
            'fitness': final_fitness[idx],
            'cost': cost,
            'energy': details['total_energy'],
            'distance': details['total_distance'],
            'num_trips': details['num_trips'],
            'solution': solution
        })
    
    if verbose:
        print(f"\n{'='*60}")
        print("RLTS-NSGA-II 求解完成!")
        print(f"{'='*60}")
        print(f"帕累托前沿解数量: {len(pareto_front)}")
        
        if pareto_front:
            costs = [s['cost'] for s in pareto_front]
            energies = [s['energy'] * 1000 for s in pareto_front]  # 转Wh
            print(f"成本范围: {min(costs):.1f} - {max(costs):.1f}")
            print(f"能耗范围: {min(energies):.1f} - {max(energies):.1f} Wh")
    
    return {
        'pareto_front': pareto_front,
        'problem': problem,
        'evolution_data': evolution_data,
        'population': pop
    }


def visualize_mtdrp_results(result: Dict, instance: MTDRPInstance, 
                            save_path: str = "picture_result/mtdrp"):
    """
    可视化MTDRP求解结果
    """
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "picture_result", exist_ok=True)
    
    import matplotlib
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    evolution_data = result['evolution_data']
    pareto_front = result['pareto_front']
    
    # 图1: 进化曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 成本进化
    ax1 = axes[0, 0]
    ax1.plot(evolution_data['generations'], evolution_data['min_cost'], 'b-', label='最小成本')
    ax1.plot(evolution_data['generations'], evolution_data['avg_cost'], 'b--', alpha=0.5, label='平均成本')
    ax1.set_xlabel('代数')
    ax1.set_ylabel('成本')
    ax1.set_title('成本进化曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 能耗进化
    ax2 = axes[0, 1]
    ax2.plot(evolution_data['generations'], evolution_data['min_energy'], 'r-', label='最小能耗')
    ax2.plot(evolution_data['generations'], evolution_data['avg_energy'], 'r--', alpha=0.5, label='平均能耗')
    ax2.set_xlabel('代数')
    ax2.set_ylabel('能耗 (Wh)')
    ax2.set_title('能耗进化曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 帕累托解数量
    ax3 = axes[1, 0]
    ax3.plot(evolution_data['generations'], evolution_data['pareto_count'], 'g-')
    ax3.set_xlabel('代数')
    ax3.set_ylabel('帕累托解数量')
    ax3.set_title('帕累托前沿规模进化')
    ax3.grid(True, alpha=0.3)
    
    # 帕累托前沿
    ax4 = axes[1, 1]
    if pareto_front:
        costs = [s['cost'] for s in pareto_front]
        energies = [s['energy'] * 1000 for s in pareto_front]
        ax4.scatter(costs, energies, c='purple', s=50, alpha=0.7)
        ax4.set_xlabel('总成本')
        ax4.set_ylabel('总能耗 (Wh)')
        ax4.set_title('帕累托前沿')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_evolution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图2: 最优解的路径可视化
    if pareto_front:
        best_solution = min(pareto_front, key=lambda x: x['cost'])
        visualize_routes(best_solution['solution'], instance, f"{save_path}_routes.png")


def visualize_routes(solution: MTDRPSolution, instance: MTDRPInstance, save_path: str):
    """
    可视化无人机路径
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 10))
    
    # 绘制depot
    plt.scatter([instance.depot.x], [instance.depot.y], 
                c='red', s=200, marker='s', zorder=5, label='配送中心')
    
    # 绘制客户点
    customer_x = [c.x for c in instance.customers]
    customer_y = [c.y for c in instance.customers]
    plt.scatter(customer_x, customer_y, c='blue', s=30, alpha=0.5, label='客户')
    
    # 绘制路径
    colors = plt.cm.tab10(np.linspace(0, 1, instance.num_drones))
    
    for drone_id, trips in solution.routes.items():
        color = colors[drone_id % len(colors)]
        for trip_idx, trip in enumerate(trips):
            if not trip:
                continue
            
            # 构建路径点
            path_x = [instance.depot.x]
            path_y = [instance.depot.y]
            
            for cust_id in trip:
                customer = next((c for c in instance.customers if c.id == cust_id), None)
                if customer:
                    path_x.append(customer.x)
                    path_y.append(customer.y)
            
            path_x.append(instance.depot.x)
            path_y.append(instance.depot.y)
            
            # 绘制路径
            plt.plot(path_x, path_y, c=color, linewidth=1.5, alpha=0.7,
                    label=f'无人机{drone_id}-行程{trip_idx}' if trip_idx == 0 else None)
    
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('MTDRP 无人机路径规划结果')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 主函数
if __name__ == "__main__":
    import os
    
    # 加载测试实例
    test_file = "dataset/instances/test_small_20.dat"
    
    if os.path.exists(test_file):
        print("加载MTDRP问题实例...")
        instance = load_instance(test_file)
        print_instance_info(instance)
        
        # 使用RLTS-NSGA-II求解
        result = solve_mtdrp_rlts_nsga2(
            instance,
            population_size=100,
            generations=100,  # 测试用较少代数
            tabu_frequency=10,
            tabu_intensity=0.3,
            enable_qlearning=True,
            verbose=True
        )
        
        # 可视化结果
        visualize_mtdrp_results(result, instance, "picture_result/mtdrp_rlts_nsga2")
        
        # 打印最优解详情
        if result['pareto_front']:
            print("\n最优解详情:")
            best = min(result['pareto_front'], key=lambda x: x['cost'])
            print(f"  总成本: {best['cost']:.2f}")
            print(f"  总能耗: {best['energy']*1000:.2f} Wh")
            print(f"  总距离: {best['distance']:.2f} m")
            print(f"  行程数: {best['num_trips']}")
    else:
        print(f"测试文件不存在: {test_file}")
        print("请确保dataset文件夹中有测试数据")
