#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度改进遗传算法模块 - RLTS-NSGA-II (Reinforcement Learning and Tabu Search enhanced NSGA-II)
基于NSGA-II进行改进，支持eVTOL调度多目标优化问题
"""

import pygmo as pg
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_definitions import get_evtols, get_locations, get_tasks
from gurobi.evtol_scheduling_gurobi import generate_task_chains
import itertools
from collections import defaultdict, deque
import random
import copy


class QLearningParameterController:
    """
    Q-learning实现的交叉率与变异率自适应调节器
    
    基于强化学习动态调整NSGA-II的交叉率和变异率参数：
    1. 状态空间：基于种群综合评价值的11个离散状态
    2. 动作空间：交叉率5个动作，变异率5个动作
    3. 奖惩机制：种群性能提升+1，否则-1
    4. 动作选择：ε-greedy策略，ε值动态衰减
    """
    
    def __init__(self, max_generations):
        """
        初始化Q-learning参数控制器
        
        参数:
            max_generations: 最大进化代数，用于ε衰减计算
        """
        self.max_generations = max_generations
        
        # 状态空间：20个状态 (0-19)
        self.num_states = 20
        
        # 动作空间定义
        self.cr_actions = [0.60, 0.70, 0.80, 0.85, 0.90]  # 交叉率动作
        self.m_actions = [0.01, 0.05, 0.10, 0.15, 0.20]   # 变异率动作
        
        # Q表初始化 (状态 × 动作)
        self.q_table_cr = np.zeros((self.num_states, len(self.cr_actions)))  # 交叉率Q表
        self.q_table_m = np.zeros((self.num_states, len(self.m_actions)))    # 变异率Q表
        
        # Q-learning参数
        self.alpha = 0.8  # 学习率
        self.gamma = 0.6  # 折扣因子
        
        # ε-greedy参数
        self.epsilon_start = 0.7
        self.epsilon_end = 0.05
        
        # 状态记录
        self.current_state = 0
        self.current_cr_action = 0
        self.current_m_action = 0
        self.previous_f_value = None
        
        # 第一代基准数据
        self.first_gen_avg_fitness = None
        self.first_gen_best_fitness = None
        
        print(f"Q-learning参数控制器初始化完成:")
        print(f"  状态空间: {self.num_states}个状态（F值范围[0,1]）")
        print(f"  状态设计: 每个状态宽度0.05，精细化状态划分")
        print(f"  交叉率动作: {self.cr_actions}")
        print(f"  变异率动作: {self.m_actions}")
        print(f"  学习率α: {self.alpha}, 折扣因子γ: {self.gamma}")
        print(f"  ε衰减: {self.epsilon_start} → {self.epsilon_end}")
        print(f"  评价公式: F = sigmoid(0.5*(当前平均/第一代平均) + 0.5*(当前最佳/第一代最佳))")
        print(f"  F=0.5表示与第一代相同，<0.5表示改善，>0.5表示恶化")
    
    def calculate_population_evaluation(self, population_fitness):
        """
        计算种群综合评价值F（基于第一代归一化）
        
        新设计:
        1. 计算当前种群平均适应度和最佳适应度
        2. 分别除以第一代对应值进行归一化
        3. 加权求和: F = 0.5 * (当前平均/第一代平均) + 0.5 * (当前最佳/第一代最佳)
        
        参数:
            population_fitness: 种群适应度值列表 [[能耗1, 延误1], [能耗2, 延误2], ...]
            
        返回:
            归一化评价值F
        """
        try:
            # 检查输入是否为空
            if population_fitness is None:
                return 1.0
            
            # 转换为numpy数组以便处理
            if isinstance(population_fitness, list):
                if len(population_fitness) == 0:
                    return 1.0
                fitness_array = np.array(population_fitness)
            else:
                fitness_array = population_fitness
            
            # 检查数组形状
            if fitness_array.ndim == 0 or fitness_array.size == 0:
                return 1.0
            
            # 确保是二维数组
            if fitness_array.ndim == 1:
                if len(fitness_array) < 2:
                    return 1.0
                fitness_array = fitness_array.reshape(1, -1)
            
            # 检查是否有足够的列（至少需要能耗和延误两列）
            if fitness_array.shape[1] < 2:
                return 1.0
            
            # 计算每个个体的综合分
            # 综合分 = 0.5 * 能耗 + 0.5 * 延误
            comprehensive_scores = 0.5 * fitness_array[:, 0] + 0.5 * fitness_array[:, 1]
            
            # 检查是否有有效分数
            if len(comprehensive_scores) == 0:
                return 1.0
            
            # 计算当前种群平均适应度和最佳适应度
            current_avg_fitness = float(np.mean(comprehensive_scores))
            current_best_fitness = float(np.min(comprehensive_scores))  # 最小值为最佳（目标是最小化）
            
            # 如果是第一代，记录基准数据
            if self.first_gen_avg_fitness is None:
                self.first_gen_avg_fitness = current_avg_fitness
                self.first_gen_best_fitness = current_best_fitness
                # 第一代返回固定值，避免除零
                return 1.0
            
            # 计算相对于第一代的归一化指标
            avg_ratio = current_avg_fitness / self.first_gen_avg_fitness if self.first_gen_avg_fitness > 0 else 1.0
            best_ratio = current_best_fitness / self.first_gen_best_fitness if self.first_gen_best_fitness > 0 else 1.0
            
            # 加权求和得到评价值F = 0.5 * 平均比率 + 0.5 * 最佳比率
            F_raw = 0.5 * avg_ratio + 0.5 * best_ratio
            
            # 将F值映射到[0,1]区间
            # 使用sigmoid函数变换：F = 1 / (1 + exp(-(F_raw - 1) * 4))
            # 当F_raw=1时，F=0.5；F_raw<1时，F<0.5；F_raw>1时，F>0.5
            F_normalized = 1.0 / (1.0 + np.exp(-(F_raw - 1.0) * 4.0))
            
            # 确保返回值严格在[0,1]区间内
            F_normalized = max(0.0, min(1.0, F_normalized))
            
            return float(F_normalized)
            
        except Exception as e:
            print(f"计算种群评价值时出错: {e}")
            print(f"population_fitness类型: {type(population_fitness)}")
            if hasattr(population_fitness, 'shape'):
                print(f"population_fitness形状: {population_fitness.shape}")
            return 1.0
    
    def get_state_from_f_value(self, f_value):
        """
        将评价值F离散化为状态（20个状态，F值范围[0,1]）
        
        状态划分（将[0,1]区间等分为20个状态）:
        - 状态0: 0.00 ≤ F < 0.05 (极佳表现)
        - 状态1: 0.05 ≤ F < 0.10 (优秀表现)
        - 状态2: 0.10 ≤ F < 0.15 (很好表现)
        - ...
        - 状态9: 0.45 ≤ F < 0.50 (较好表现)
        - 状态10: 0.50 ≤ F < 0.55 (中等表现，接近第一代水平)
        - 状态11: 0.55 ≤ F < 0.60 (略差表现)
        - ...
        - 状态18: 0.90 ≤ F < 0.95 (较差表现)
        - 状态19: 0.95 ≤ F ≤ 1.00 (很差表现)
        
        参数:
            f_value: 评价值F（范围[0,1]）
            
        返回:
            状态编号 (0-19)
        """
        try:
            # 确保f_value是标量
            f_value = float(f_value)
            
            # 确保f_value在[0,1]范围内
            f_value = max(0.0, min(1.0, f_value))
            
            # 将[0,1]区间等分为20个状态
            # 每个状态的宽度为0.05
            state = int(f_value * 20)
            
            # 确保状态在[0,19]范围内
            state = max(0, min(19, state))
            
            return int(state)
                
        except Exception as e:
            print(f"状态转换时出错: {e}")
            print(f"f_value: {f_value}, type: {type(f_value)}")
            return 10  # 返回中间状态作为默认值
    
    def get_epsilon(self, current_generation):
        """
        计算当前代的ε值（用于ε-greedy策略）
        
        ε衰减公式: ε = 0.7 - 当前代数 * (0.65 / 最大代数)
        
        参数:
            current_generation: 当前代数
            
        返回:
            ε值
        """
        try:
            epsilon = self.epsilon_start - current_generation * (
                (self.epsilon_start - self.epsilon_end) / self.max_generations)
            return max(epsilon, self.epsilon_end)
            
        except Exception as e:
            print(f"计算ε值时出错: {e}")
            return self.epsilon_end
    
    def select_actions(self, state, current_generation):
        """
        使用ε-greedy策略选择动作
        
        参数:
            state: 当前状态
            current_generation: 当前代数
            
        返回:
            (交叉率动作索引, 变异率动作索引)
        """
        try:
            # 确保state是整数且在有效范围内
            state = int(state)
            state = max(0, min(self.num_states - 1, state))
            
            epsilon = self.get_epsilon(current_generation)
            
            # 选择交叉率动作
            if random.random() < epsilon:
                # 随机选择
                cr_action = random.randint(0, len(self.cr_actions) - 1)
            else:
                # 选择Q值最高的动作
                cr_action = int(np.argmax(self.q_table_cr[state]))
            
            # 选择变异率动作
            if random.random() < epsilon:
                # 随机选择
                m_action = random.randint(0, len(self.m_actions) - 1)
            else:
                # 选择Q值最高的动作
                m_action = int(np.argmax(self.q_table_m[state]))
            
            # 确保动作索引在有效范围内
            cr_action = max(0, min(len(self.cr_actions) - 1, cr_action))
            m_action = max(0, min(len(self.m_actions) - 1, m_action))
            
            return cr_action, m_action
            
        except Exception as e:
            print(f"选择动作时出错: {e}")
            print(f"state: {state}, current_generation: {current_generation}")
            return 0, 0
    
    def get_parameters(self, population_fitness, current_generation):
        """
        获取当前代的交叉率和变异率参数
        
        参数:
            population_fitness: 当前种群适应度值
            current_generation: 当前代数
            
        返回:
            (交叉率, 变异率)
        """
        try:
            # 计算当前种群评价值
            current_f_value = self.calculate_population_evaluation(population_fitness)
            
            # 获取当前状态
            self.current_state = self.get_state_from_f_value(current_f_value)
            
            # 选择动作
            self.current_cr_action, self.current_m_action = self.select_actions(
                self.current_state, current_generation)
            
            # 获取对应的参数值
            cr = self.cr_actions[self.current_cr_action]
            m = self.m_actions[self.current_m_action]
            
            # 更新Q表（如果不是第一代）
            if self.previous_f_value is not None:
                self.update_q_tables(current_f_value)
            
            # 记录当前F值用于下一代更新
            self.previous_f_value = current_f_value
            
            return cr, m
            
        except Exception as e:
            print(f"获取参数时出错: {e}")
            return 0.8, 0.1  # 返回默认值
    
    def update_q_tables(self, new_f_value):
        """
        更新Q表
        
        奖励机制:
        - 如果F_new < F_old（种群性能提升），奖励+1
        - 否则惩罚-1
        
        Q-learning更新公式:
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        参数:
            new_f_value: 新的评价值F
        """
        try:
            # 确保值是标量
            new_f_value = float(new_f_value)
            previous_f_value = float(self.previous_f_value) if self.previous_f_value is not None else 1.0
            
            # 计算奖励
            if new_f_value < previous_f_value:
                reward = 1.0  # 性能提升，给予奖励
            else:
                reward = -1.0  # 性能下降或无变化，给予惩罚
            
            # 获取新状态
            new_state = self.get_state_from_f_value(new_f_value)
            
            # 确保状态和动作索引在有效范围内
            current_state = max(0, min(self.num_states - 1, self.current_state))
            new_state = max(0, min(self.num_states - 1, new_state))
            current_cr_action = max(0, min(len(self.cr_actions) - 1, self.current_cr_action))
            current_m_action = max(0, min(len(self.m_actions) - 1, self.current_m_action))
            
            # 更新交叉率Q表
            old_q_cr = float(self.q_table_cr[current_state, current_cr_action])
            max_future_q_cr = float(np.max(self.q_table_cr[new_state]))
            new_q_cr = old_q_cr + self.alpha * (reward + self.gamma * max_future_q_cr - old_q_cr)
            self.q_table_cr[current_state, current_cr_action] = new_q_cr
            
            # 更新变异率Q表
            old_q_m = float(self.q_table_m[current_state, current_m_action])
            max_future_q_m = float(np.max(self.q_table_m[new_state]))
            new_q_m = old_q_m + self.alpha * (reward + self.gamma * max_future_q_m - old_q_m)
            self.q_table_m[current_state, current_m_action] = new_q_m
            
        except Exception as e:
            print(f"更新Q表时出错: {e}")
            print(f"new_f_value: {new_f_value}, previous_f_value: {self.previous_f_value}")
            print(f"current_state: {self.current_state}, current_cr_action: {self.current_cr_action}")
    
    def get_learning_stats(self):
        """
        获取学习统计信息
        
        返回:
            学习统计字典
        """
        try:
            return {
                'current_state': self.current_state,
                'current_cr': self.cr_actions[self.current_cr_action],
                'current_m': self.m_actions[self.current_m_action],
                'q_table_cr_avg': np.mean(self.q_table_cr),
                'q_table_m_avg': np.mean(self.q_table_m),
                'previous_f_value': self.previous_f_value
            }
            
        except Exception as e:
            print(f"获取学习统计时出错: {e}")
            return {}


class TabuSearchManager:
    """
    禁忌搜索管理器
    
    管理基于关键任务串的局部禁忌搜索策略：
    1. 识别关键任务串：基于延误+能耗+同一时刻高度层冲突惩罚的加权和最高的任务串
    2. 邻域移动：对关键任务串进行多种类型的移动操作：
       - eVTOL重新分配：改变任务串的eVTOL分配
       - 任务串时间调整：调整任务串的开始时间
       - 任务高度层调整：调整任务串内任务的高度层航线
       - 任务时间微调：调整任务串内各任务的开始时间
    3. 禁忌表：记录最近的移动操作，短期内禁止反向操作，避免循环搜索
    """
    
    def __init__(self, tabu_tenure=7, max_moves_per_iteration=3):
        """
        初始化禁忌搜索管理器
        
        参数:
            tabu_tenure: 禁忌期长度（禁忌表中保存的移动操作数量）
            max_moves_per_iteration: 每次迭代最大移动操作数
        """
        self.tabu_tenure = tabu_tenure
        self.max_moves_per_iteration = max_moves_per_iteration
        self.tabu_list = deque(maxlen=tabu_tenure)  # 禁忌表，使用双端队列自动维护大小
        
    def identify_critical_paths(self, solution, problem):
        """
        识别关键任务串：基于延误+能耗与同一时刻高度层冲突惩罚的加权和最高的任务串
        
        评估指标：
        1. 任务串的延误惩罚
        2. 任务串的能耗
        3. 同一时刻高度层冲突惩罚
        4. 加权和 = w1*延误 + w2*能耗 + w3*冲突惩罚
        
        返回: (关键任务串ID, 该任务串的评估分数)
        """
        if not solution:
            return None, 0.0
            
        try:
            # 权重设置
            w_delay = 0.4      # 延误权重
            w_energy = 0.3     # 能耗权重  
            w_conflict = 0.3   # 冲突惩罚权重
            
            chain_scores = {}  # 任务串ID -> 综合评估分数
            
            # 计算每个任务串的综合评估分数
            for c in range(problem.num_chains):
                # 1. 计算延误惩罚
                delay_penalty = 0
                chain_start_time = solution['chain_start'][c]
                chain_tasks = problem.task_chains[c]
                
                for task_idx, task_id in enumerate(chain_tasks):
                    # 从任务列表中找到对应的任务
                    task = None
                    for t in problem.tasks:
                        if t['id'] == task_id:
                            task = t
                            break
                    
                    if task:
                        # 计算任务实际开始时间
                        if task_idx == 0:
                            actual_start = chain_start_time
                        else:
                            # 简化处理：使用任务串开始时间 + 任务在串中的位置估算
                            actual_start = chain_start_time + task_idx * 30  # 假设每个任务间隔30分钟
                        
                        # 延误 = max(0, 实际开始时间 - 最晚开始时间)
                        if actual_start > task['latest_start']:
                            delay_penalty += (actual_start - task['latest_start']) * 10  # 延误惩罚系数
                
                # 2. 计算能耗
                energy_consumption = 0
                for task_id in chain_tasks:
                    # 获取任务的能耗（根据高度层选择）
                    altitude_idx = solution.get(f'task_altitude_{task_id}', 1)  # 默认中等高度层
                    # 从任务列表中找到对应的任务
                    task = None
                    for t in problem.tasks:
                        if t['id'] == task_id:
                            task = t
                            break
                    if task:
                        energy_consumption += task['soc_consumption'][altitude_idx]
                
                # 3. 计算同一时刻高度层冲突惩罚
                conflict_penalty = 0
                chain_start = solution['chain_start'][c]
                chain_end = solution['chain_end'][c]
                
                # 检查与其他任务串在相同时间窗口内的高度层冲突
                for other_c in range(problem.num_chains):
                    if other_c != c:
                        other_start = solution['chain_start'][other_c]
                        other_end = solution['chain_end'][other_c]
                        
                        # 检查时间重叠
                        if not (chain_end <= other_start or other_end <= chain_start):
                            # 有时间重叠，检查高度层冲突
                            chain_tasks = problem.task_chains[c]
                            other_tasks = problem.task_chains[other_c]
                            
                            for task_id in chain_tasks:
                                task_altitude = solution.get(f'task_altitude_{task_id}', 1)
                                for other_task_id in other_tasks:
                                    other_altitude = solution.get(f'task_altitude_{other_task_id}', 1)
                                    
                                    # 如果在相同高度层且路径可能冲突
                                    if task_altitude == other_altitude:
                                        # 从任务列表中找到对应的任务
                                        task = None
                                        other_task = None
                                        for t in problem.tasks:
                                            if t['id'] == task_id:
                                                task = t
                                            if t['id'] == other_task_id:
                                                other_task = t
                                        
                                        if task and other_task:
                                            # 简化的路径冲突检测：相同起点或终点
                                            if (task['from'] == other_task['from'] or 
                                                task['to'] == other_task['to'] or
                                                task['from'] == other_task['to'] or
                                                task['to'] == other_task['from']):
                                                conflict_penalty += 50  # 冲突惩罚
                
                # 4. 计算综合评估分数
                total_score = (w_delay * delay_penalty + 
                              w_energy * energy_consumption + 
                              w_conflict * conflict_penalty)
                
                chain_scores[c] = total_score
            
            # 找到评估分数最高的任务串（最关键的任务串）
            if chain_scores:
                critical_chain = max(chain_scores.keys(), key=lambda x: chain_scores[x])
                return critical_chain, chain_scores[critical_chain]
            else:
                return None, 0.0
                
        except Exception as e:
            print(f"识别关键任务串时出错: {e}")
            return None, 0.0
    
    def generate_neighborhood_moves(self, critical_chain, critical_score, solution, problem):
        """
        生成邻域移动操作
        
        移动类型：
        1. eVTOL重新分配：将关键任务串分配给其他eVTOL
        2. 任务串开始时间调整：调整关键任务串的开始时间
        3. 任务高度层调整：调整任务串内任务的高度层
        4. 任务开始时间微调：调整任务串内各任务的开始时间
        
        参数:
            critical_chain: 关键任务串ID
            critical_score: 关键任务串的评估分数
            solution: 当前解
            problem: 问题实例
            
        返回: [move_dict, ...]
        """
        moves = []
        
        try:
            if critical_chain is None:
                return moves
            
            # 获取当前任务串的eVTOL分配
            current_evtol = None
            for k in range(problem.num_evtols):
                if solution['b_chain_evtol'][critical_chain, k] == 1:
                    current_evtol = k
                    break
            
            if current_evtol is None:
                return moves
            
            # 移动类型1: eVTOL重新分配
            for target_evtol in range(problem.num_evtols):
                if target_evtol != current_evtol:
                    move = {
                        'type': 'reassign_evtol',
                        'chain_id': critical_chain,
                        'from_evtol': current_evtol,
                        'to_evtol': target_evtol,
                        'original_time': solution['chain_start'][critical_chain]
                    }
                    moves.append(move)
            
            # 移动类型2: 任务串开始时间调整
            current_start = solution['chain_start'][critical_chain]
            time_adjustments = [-60, -30, -15, 15, 30, 60]  # 时间调整选项（分钟）
            
            for adjustment in time_adjustments:
                new_start = current_start + adjustment
                if 0 <= new_start <= problem.time_horizon - 60:  # 确保在合理范围内
                    move = {
                        'type': 'adjust_chain_time',
                        'chain_id': critical_chain,
                        'original_start': current_start,
                        'new_start': new_start,
                        'adjustment': adjustment
                    }
                    moves.append(move)
            
            # 移动类型3: 任务高度层调整（降低概率）
            chain_tasks = problem.task_chains[critical_chain]
            
            # 只对部分任务生成高度层调整移动（概率约30%）
            selected_tasks_altitude = random.sample(chain_tasks, max(1, len(chain_tasks) // 3))
            for task_id in selected_tasks_altitude:
                current_altitude = solution.get(f'task_altitude_{task_id}', 1)
                
                # 只尝试一个随机的其他高度层
                other_altitudes = [alt for alt in [0, 1, 2] if alt != current_altitude]
                if other_altitudes:
                    new_altitude = random.choice(other_altitudes)
                    move = {
                        'type': 'adjust_altitude',
                        'chain_id': critical_chain,
                        'task_id': task_id,
                        'original_altitude': current_altitude,
                        'new_altitude': new_altitude
                    }
                    moves.append(move)
            
            # 移动类型4: 任务开始时间微调（降低概率）
            # 只对部分任务生成时间微调移动（概率约30%）
            selected_tasks_time = random.sample(chain_tasks, max(1, len(chain_tasks) // 3))
            for task_id in selected_tasks_time:
                current_task_start = solution.get(f'task_start_{task_id}', current_start)
                
                # 只选择一个随机的时间调整
                task_adjustments = [-20, -10, 10, 20]
                adjustment = random.choice(task_adjustments)
                new_task_start = current_task_start + adjustment
                if new_task_start >= 0:  # 确保不为负
                    move = {
                        'type': 'adjust_task_time',
                        'chain_id': critical_chain,
                        'task_id': task_id,
                        'original_start': current_task_start,
                        'new_start': new_task_start,
                        'adjustment': adjustment
                    }
                    moves.append(move)
            
            return moves
            
        except Exception as e:
            print(f"生成邻域移动时出错: {e}")
            return []
    
    def is_move_tabu(self, move):
        """
        检查移动操作是否在禁忌表中
        
        参数:
            move: 移动操作字典
            
        返回: True if 禁忌, False otherwise
        """
        try:
            # 生成移动操作的反向操作标识
            if move['type'] == 'reassign_evtol':
                # eVTOL重新分配的反向操作：从目标eVTOL分配回原eVTOL
                reverse_key = f"reassign_evtol_{move['chain_id']}_{move['to_evtol']}_{move['from_evtol']}"
            elif move['type'] == 'adjust_chain_time':
                # 时间调整的反向操作：调整回原时间
                reverse_key = f"adjust_chain_time_{move['chain_id']}_{move['new_start']}_{move['original_start']}"
            elif move['type'] == 'adjust_altitude':
                # 高度层调整的反向操作：调整回原高度层
                reverse_key = f"adjust_altitude_{move['task_id']}_{move['new_altitude']}_{move['original_altitude']}"
            elif move['type'] == 'adjust_task_time':
                # 任务时间调整的反向操作：调整回原时间
                reverse_key = f"adjust_task_time_{move['task_id']}_{move['new_start']}_{move['original_start']}"
            else:
                return False
            
            return reverse_key in self.tabu_list
            
        except Exception as e:
            print(f"检查禁忌状态时出错: {e}")
            return False
    
    def add_to_tabu_list(self, move):
        """
        将移动操作添加到禁忌表
        
        参数:
            move: 移动操作字典
        """
        try:
            if move['type'] == 'reassign_evtol':
                tabu_key = f"reassign_evtol_{move['chain_id']}_{move['from_evtol']}_{move['to_evtol']}"
            elif move['type'] == 'adjust_chain_time':
                tabu_key = f"adjust_chain_time_{move['chain_id']}_{move['original_start']}_{move['new_start']}"
            elif move['type'] == 'adjust_altitude':
                tabu_key = f"adjust_altitude_{move['task_id']}_{move['original_altitude']}_{move['new_altitude']}"
            elif move['type'] == 'adjust_task_time':
                tabu_key = f"adjust_task_time_{move['task_id']}_{move['original_start']}_{move['new_start']}"
            else:
                return
            
            self.tabu_list.append(tabu_key)
            
        except Exception as e:
            print(f"添加到禁忌表时出错: {e}")
    
    def apply_move(self, move, individual, problem):
        """
        应用移动操作到个体编码
        
        参数:
            move: 移动操作字典
            individual: 个体编码（numpy数组）
            problem: 问题实例
            
        返回: 修改后的个体编码
        """
        try:
            new_individual = individual.copy()
            
            # 调试信息
            if len(new_individual) != problem.dimensions:
                print(f"警告: 个体长度({len(new_individual)}) != 问题维度({problem.dimensions})")
            
            if move['type'] == 'reassign_evtol':
                # eVTOL重新分配：将任务串分配给不同的eVTOL
                chain_id = move['chain_id']
                new_evtol = move['to_evtol']
                
                # 确保chain_id在有效范围内
                if 0 <= chain_id < problem.num_chains:
                    # 修改y变量中的eVTOL分配
                    evtol_idx = problem.y_start + chain_id * 2
                    if evtol_idx < len(new_individual):
                        new_individual[evtol_idx] = min(new_evtol, problem.num_evtols - 1)
                    
                    # 随机微调开始时间，避免冲突
                    time_idx = problem.y_start + chain_id * 2 + 1
                    if time_idx < len(new_individual):
                        current_time = new_individual[time_idx]
                        time_adjustment = random.randint(-30, 30)  # ±30分钟的随机调整
                        new_time = max(0, min(problem.time_horizon - 1, current_time + time_adjustment))
                        new_individual[time_idx] = new_time
                
            elif move['type'] == 'adjust_chain_time':
                # 任务串开始时间调整
                chain_id = move['chain_id']
                new_start = move['new_start']
                
                # 确保chain_id在有效范围内
                if 0 <= chain_id < problem.num_chains:
                    # 修改任务串开始时间
                    time_idx = problem.y_start + chain_id * 2 + 1
                    if time_idx < len(new_individual):
                        new_start_clamped = max(0, min(problem.time_horizon - 1, new_start))
                        new_individual[time_idx] = new_start_clamped
                
            elif move['type'] == 'adjust_altitude':
                # 任务高度层调整（实际上是航线调整）
                task_id = move['task_id']
                new_altitude = move['new_altitude']
                
                # 修改z变量中的航线选择（航线对应高度层）
                # 需要找到task_id在任务列表中的索引
                task_idx = None
                for i, task in enumerate(problem.tasks):
                    if task['id'] == task_id:
                        task_idx = i
                        break
                
                if task_idx is not None:
                    route_idx = problem.z_start + task_idx
                    # 确保索引在有效范围内
                    if route_idx < len(new_individual):
                        new_individual[route_idx] = new_altitude  # 直接设置航线ID（对应高度层）
                        
            elif move['type'] == 'adjust_task_time':
                # 任务开始时间微调
                task_id = move['task_id']
                new_start = move['new_start']
                
                # 修改task_start变量中的任务开始时间
                # 需要找到task_id在任务列表中的索引
                task_idx = None
                for i, task in enumerate(problem.tasks):
                    if task['id'] == task_id:
                        task_idx = i
                        break
                
                if task_idx is not None:
                    task_time_idx = problem.task_start_var_start + task_idx
                    # 确保索引在有效范围内
                    if task_time_idx < len(new_individual):
                        new_start_clamped = max(0, min(problem.time_horizon - 1, new_start))
                        new_individual[task_time_idx] = new_start_clamped
            
            return new_individual
            
        except Exception as e:
            print(f"应用移动操作时出错: {e}")
            return individual
    
    def local_search_improvement(self, individual, problem, max_iterations=5):
        """
        基于关键任务串的局部禁忌搜索改进
        
        改进策略：
        1. 识别延误+能耗+冲突惩罚最高的关键任务串
        2. 对关键任务串进行多种类型的邻域移动
        3. 使用禁忌表避免循环搜索
        
        参数:
            individual: 当前个体编码
            problem: 问题实例
            max_iterations: 最大迭代次数
            
        返回: 改进后的个体编码
        """
        try:
            current_individual = individual.copy()
            best_individual = individual.copy()
            best_fitness = problem.fitness(individual)
            
            for iteration in range(max_iterations):
                # 解码当前解
                current_solution = problem._decode_solution(current_individual)
                if not current_solution:
                    break
                
                # 识别关键任务串
                critical_chain, critical_score = \
                    self.identify_critical_paths(current_solution, problem)
                
                if critical_chain is None or critical_score <= 0:
                    break  # 没有找到关键任务串或评估分数很低
                
                # 生成邻域移动
                possible_moves = self.generate_neighborhood_moves(
                    critical_chain, critical_score, current_solution, problem)
                
                if not possible_moves:
                    break  # 没有可行的移动
                
                # 过滤禁忌移动并评估剩余移动
                best_move = None
                best_move_fitness = None
                best_move_individual = None
                
                # 随机选择一部分移动进行评估（避免计算开销过大）
                evaluated_moves = random.sample(possible_moves, 
                                              min(len(possible_moves), self.max_moves_per_iteration))
                
                for move in evaluated_moves:
                    if self.is_move_tabu(move):
                        continue  # 跳过禁忌移动
                    
                    # 应用移动并评估
                    move_individual = self.apply_move(move, current_individual, problem)
                    move_fitness = problem.fitness(move_individual)
                    
                    # 选择最好的非禁忌移动
                    if (best_move_fitness is None or 
                        self._is_better_fitness(move_fitness, best_move_fitness)):
                        best_move = move
                        best_move_fitness = move_fitness
                        best_move_individual = move_individual
                
                # 如果找到了改进的移动
                if best_move is not None:
                    # 应用最佳移动
                    current_individual = best_move_individual
                    self.add_to_tabu_list(best_move)
                    
                    # 更新全局最佳解
                    if self._is_better_fitness(best_move_fitness, best_fitness):
                        best_individual = best_move_individual.copy()
                        best_fitness = best_move_fitness
                        
                    # 输出改进信息
                    print(f"  禁忌搜索第{iteration+1}次迭代: 关键任务串{critical_chain}, "
                          f"移动类型{best_move['type']}, 适应度改进")
                else:
                    # 没有找到改进的移动，结束搜索
                    break
            
            return best_individual
            
        except Exception as e:
            print(f"局部禁忌搜索时出错: {e}")
            return individual
    
    def _is_better_fitness(self, fitness1, fitness2):
        """
        比较两个适应度值，判断fitness1是否比fitness2更好
        对于多目标优化，这里使用简单的加权和比较
        """
        try:
            if fitness2 is None:
                return True
            
            # 简单的加权和比较（可以根据需要调整权重）
            score1 = fitness1[0] + fitness1[1]  # 能耗 + 延误
            score2 = fitness2[0] + fitness2[1]
            
            return score1 < score2
            
        except Exception as e:
            print(f"比较适应度时出错: {e}")
            return False


class eVTOLSchedulingProblem:
    """
    eVTOL调度问题的PyGMO封装类 - 用于RLTS-NSGA-II算法
    
    数学模型完全对应gurobi_multi实现，包含相同的决策变量和约束条件：
    
    决策变量：
    1. y[c,k,t] - eVTOL k在时刻t开始执行任务串c (二进制)
    2. z[i,h] - 任务i使用航线h (二进制)
    3. task_start[i] - 任务i的开始时间 (整数) - 直接编码
    4. task_end[i] - 任务i的结束时间 (整数) - 从task_start推导
    5. chain_start[c] - 任务串c的开始时间 (整数) - 从y推导
    6. chain_end[c] - 任务串c的结束时间 (整数) - 从任务推导
    7. b_chain_evtol[c,k] - 任务串c是否分配给eVTOL k (二进制) - 从y推导
    8. both_assigned[c1,c2,k] - 任务串c1和c2是否都分配给eVTOL k (二进制) - 推导
    9. chain_order[c1,c2,k] - 任务串c1是否在c2之前执行 (二进制) - 推导
    10. both_use_route_h[i,j,h] - 任务i和j是否都使用航线h (二进制) - 推导
    11. i_before_j[i,j,h] - 任务i是否在任务j之前完成 (二进制) - 推导
    
    编码方案 (简化版):
    [y_evtol[0], y_time[0], ..., y_evtol[num_chains-1], y_time[num_chains-1],
     z_route[0], ..., z_route[num_tasks-1],
     task_start[0], ..., task_start[num_tasks-1]]
    
    约束条件：
    2.1 任务串分配唯一性约束
    2.2 每个任务必须选择一条航线
    2.3 任务串开始时间约束
    2.4 任务串内任务的时间约束 - 简化为直接检查
    2.5 eVTOL同一时刻只能执行一个任务串
    2.6 高度层防撞约束
    2.7 任务串之间的时间间隔约束
    2.8 任务时间窗约束
    
    多目标函数 (对应gurobi_multi的epsilon约束方法)：
    目标1: minimize 总能耗
    目标2: minimize 总延误
    
    注意: 这是真正的多目标优化，无权重组合！
    编码优化: 任务开始时间直接编码，大幅简化解码计算
    """
    
    def __init__(self, tasks, evtols, task_chains, time_horizon=720):
        self.tasks = tasks
        self.evtols = evtols
        self.task_chains = task_chains
        self.time_horizon = time_horizon
        
        self.num_tasks = len(tasks)
        self.num_evtols = len(evtols)
        self.num_chains = len(task_chains)
        self.num_routes = 3
        
        # 遗传编码设计
        self._setup_encoding()
        
        # 约束参数
        self.chain_interval_time = 40  # 任务串之间的最小间隔时间（与任务串内间隔保持一致）
        
        print(f"问题规模: {self.num_tasks}个任务, {self.num_evtols}架eVTOL, {self.num_chains}个任务串")
        print(f"决策变量维度: {self.dimensions} (简化的纯整数编码)")
        print(f"编码方案: 任务串分配({self.num_chains*2}) + 航线选择({self.num_tasks}) + 任务开始时间({self.num_tasks})")
        print(f"多目标优化: 目标1=总能耗, 目标2=总延误 (无权重组合)")
        print(f"优化特点: 任务开始时间直接编码，无需复杂解码计算")
    
    def _setup_encoding(self):
        """
        设计遗传编码方案 - 直接编码任务开始时间
        
        编码结构：
        [y_variables | z_variables | task_start_variables]
        
        1. y_variables (任务串分配和开始时间):
           - 每个任务串c对应2个整数：
             * y_evtol[c]: 范围[0, num_evtols-1] -> 直接eVTOL ID
             * y_time[c]: 范围[0, time_horizon-1] -> 直接开始时间
           - 总计: num_chains * 2 个变量
        
        2. z_variables (航线选择):
           - 每个任务i对应1个整数：
             * z_route[i]: 范围[0, num_routes-1] -> 直接航线ID
           - 总计: num_tasks 个变量
        
        3. task_start_variables (任务开始时间):
           - 每个任务i对应1个整数：
             * task_start[i]: 范围[0, time_horizon-1] -> 直接开始时间
           - 总计: num_tasks 个变量
           
        总维度 = num_chains * 2 + num_tasks + num_tasks
                = num_chains * 2 + num_tasks * 2
        """
        # 编码段索引
        self.y_start = 0
        self.y_end = self.num_chains * 2
        self.z_start = self.y_end
        self.z_end = self.z_start + self.num_tasks
        self.task_start_var_start = self.z_end
        self.task_start_var_end = self.task_start_var_start + self.num_tasks
        
        self.dimensions = self.task_start_var_end
        
        # 设置各段的边界
        self._setup_bounds()
    
    def _setup_bounds(self):
        """设置各变量段的边界"""
        self.lower_bounds = []
        self.upper_bounds = []
        
        # 1. y_variables边界 (任务串分配)
        for c in range(self.num_chains):
            self.lower_bounds.append(0)                    # eVTOL ID下界
            self.upper_bounds.append(self.num_evtols - 1)  # eVTOL ID上界
            self.lower_bounds.append(0)                    # 开始时间下界
            self.upper_bounds.append(self.time_horizon - 1) # 开始时间上界
        
        # 2. z_variables边界 (航线选择)
        for i in range(self.num_tasks):
            self.lower_bounds.append(0)                    # 航线ID下界
            self.upper_bounds.append(self.num_routes - 1)  # 航线ID上界
        
        # 3. task_start_variables边界 (任务开始时间)
        for i in range(self.num_tasks):
            self.lower_bounds.append(0)                    # 任务开始时间下界
            self.upper_bounds.append(self.time_horizon - 1) # 任务开始时间上界
    
    def get_bounds(self):
        """返回决策变量的边界"""
        return (self.lower_bounds, self.upper_bounds)
    
    def get_nobj(self):
        """返回目标函数数量"""
        return 2  # 能耗 + 延误
    
    def get_nec(self):
        """返回等式约束数量"""
        return 0  # 使用惩罚函数处理所有约束
    
    def get_nic(self):
        """返回不等式约束数量"""
        return 0  # 使用惩罚函数处理所有约束
    
    def get_nix(self):
        """返回整数变量数量"""
        return self.dimensions  # 所有变量都是整数
    
    def _decode_solution(self, x):
        """
        解码遗传个体为调度方案
        
        返回解码后的调度变量字典，对应gurobi中的决策变量
        """
        try:
            # 边界修复：确保所有值都在有效范围内
            x_repaired = self._repair_solution(x)
            
            # 1. 解码y变量 (任务串分配) - 直接使用整数值
            y = {}
            for c in range(self.num_chains):
                evtol_id = int(x_repaired[self.y_start + c * 2])
                start_time = int(x_repaired[self.y_start + c * 2 + 1])
                
                # 初始化y矩阵
                for k in range(self.num_evtols):
                    for t in range(self.time_horizon):
                        y[c, k, t] = 0
                
                # 设置选中的分配
                y[c, evtol_id, start_time] = 1
            
            # 2. 解码z变量 (航线选择) - 直接使用整数值
            z = {}
            for i in range(self.num_tasks):
                route_id = int(x_repaired[self.z_start + i])
                
                # 初始化z矩阵
                for h in range(self.num_routes):
                    z[i, h] = 0
                
                # 设置选中的航线
                z[i, route_id] = 1
            
            # 3. 解码并修复任务开始时间 - 确保任务串时序约束
            task_start = {}
            for i in range(self.num_tasks):
                task_start[i] = int(x_repaired[self.task_start_var_start + i])
            
            # 修复任务串内的时序约束
            task_start = self._repair_task_chain_timing(task_start, y, z)
            
            # 4. 计算任务结束时间
            task_end = {}
            for i in range(self.num_tasks):
                duration = 0
                for h in range(self.num_routes):
                    if z[i, h] == 1:
                        duration = self.tasks[i]['duration'][h]
                        break
                task_end[i] = task_start[i] + duration
            
            # 5. 计算任务串开始时间
            chain_start = {}
            for c in range(self.num_chains):
                for k in range(self.num_evtols):
                    for t in range(self.time_horizon):
                        if y[c, k, t] == 1:
                            chain_start[c] = t
                            break
            
            # 6. 计算任务串结束时间
            chain_end = {}
            for c, chain in enumerate(self.task_chains):
                last_task_id = chain[-1]
                chain_end[c] = task_end[last_task_id]
            
            # 7. 计算辅助变量
            b_chain_evtol = {}
            for c in range(self.num_chains):
                for k in range(self.num_evtols):
                    b_chain_evtol[c, k] = sum(y[c, k, t] for t in range(self.time_horizon))
            
            both_assigned = {}
            for k in range(self.num_evtols):
                for c1 in range(self.num_chains):
                    for c2 in range(c1 + 1, self.num_chains):
                        both_assigned[c1, c2, k] = min(b_chain_evtol[c1, k], b_chain_evtol[c2, k])
            
            return {
                'y': y,
                'z': z,
                'task_start': task_start,
                'task_end': task_end,
                'chain_start': chain_start,
                'chain_end': chain_end,
                'b_chain_evtol': b_chain_evtol,
                'both_assigned': both_assigned
            }
            
        except Exception as e:
            print(f"解码错误: {e}")
            return None
    
    def _repair_task_chain_timing(self, task_start, y, z):
        """
        修复任务串内的时序约束
        
        确保:
        1. 第一个任务开始时间 = 任务串开始时间
        2. 后续任务开始时间 >= 前一个任务结束时间 + 40分钟 (最小约束，允许更大间隔)
        
        保持编码的灵活性：如果编码的间隔 > 40分钟，则保持原有间隔
        """
        repaired_task_start = task_start.copy()
        
        # 获取任务串开始时间
        chain_start_times = {}
        for c in range(self.num_chains):
            for k in range(self.num_evtols):
                for t in range(self.time_horizon):
                    if y[c, k, t] == 1:
                        chain_start_times[c] = t
                        break
        
        # 修复每个任务串内的时序
        for c, chain in enumerate(self.task_chains):
            if len(chain) == 0:
                continue
                
            chain_start_time = chain_start_times[c]
            
            # 修复第一个任务：必须等于任务串开始时间
            first_task = chain[0]
            repaired_task_start[first_task] = max(chain_start_time, self.tasks[first_task]['earliest_start'])
            
            # 修复后续任务：只修复违反最小间隔的情况，保持更大间隔
            for i in range(1, len(chain)):
                curr_task = chain[i]
                prev_task = chain[i-1]
                
                # 计算前一个任务的结束时间
                prev_duration = 0
                for h in range(self.num_routes):
                    if z[prev_task, h] == 1:
                        prev_duration = self.tasks[prev_task]['duration'][h]
                        break
                
                prev_end_time = repaired_task_start[prev_task] + prev_duration
                min_required_start = prev_end_time + 40  # 最小要求：40分钟间隔
                
                # 获取当前任务的编码开始时间
                encoded_start = task_start[curr_task]
                
                # 只有当编码的开始时间违反最小约束时才修复
                if encoded_start < min_required_start:
                    # 违反最小间隔，需要修复
                    required_start = max(min_required_start, self.tasks[curr_task]['earliest_start'])
                    repaired_task_start[curr_task] = min(required_start, self.time_horizon - 1)
                else:
                    # 编码的间隔 >= 40分钟，保持原有间隔
                    # 但仍需确保满足任务的最早开始时间约束
                    required_start = max(encoded_start, self.tasks[curr_task]['earliest_start'])
                    repaired_task_start[curr_task] = min(required_start, self.time_horizon - 1)
        
        return repaired_task_start
    
    def _repair_solution(self, x):
        """
        修复解向量，确保所有值都在有效边界内
        """
        x_repaired = []
        for i in range(len(x)):
            val = x[i]
            lower = self.lower_bounds[i]
            upper = self.upper_bounds[i]
            
            # 将值限制在边界内并转换为整数
            val = max(lower, min(upper, int(round(val))))
            x_repaired.append(val)
        
        return x_repaired
    
    def _calculate_objectives(self, solution):
        """
        计算目标函数值 - 对应gurobi_multi的两个独立目标
        """
        z = solution['z']
        task_start = solution['task_start']
        
        # 目标1: 总能量消耗 (与gurobi_multi完全相同)
        total_energy = 0
        for i in range(self.num_tasks):
            for h in range(self.num_routes):
                if z[i, h] == 1:
                    total_energy += self.tasks[i]['soc_consumption'][h]
        
        # 目标2: 总延误时间 (与gurobi_multi完全相同)
        total_delay = 0
        for i in range(self.num_tasks):
            delay = max(0, task_start[i] - self.tasks[i]['earliest_start'])
            total_delay += delay
        
        return total_energy, total_delay
    
    def _check_constraints(self, solution):
        """检查约束违反情况"""
        violations = []
        penalty = 0.0
        
        y = solution['y']
        z = solution['z']
        task_start = solution['task_start']
        task_end = solution['task_end']
        chain_start = solution['chain_start']
        chain_end = solution['chain_end']
        b_chain_evtol = solution['b_chain_evtol']
        both_assigned = solution['both_assigned']
        
        # 2.1 任务串分配唯一性约束
        for c in range(self.num_chains):
            assignment_sum = sum(y[c, k, t] for k in range(self.num_evtols) for t in range(self.time_horizon))
            if abs(assignment_sum - 1.0) > 1e-6:
                violations.append(f"任务串{c}分配违反唯一性: {assignment_sum}")
                penalty += 1000
        
        # 2.2 航线选择唯一性约束
        for i in range(self.num_tasks):
            route_sum = sum(z[i, h] for h in range(self.num_routes))
            if abs(route_sum - 1.0) > 1e-6:
                violations.append(f"任务{i}航线选择违反唯一性: {route_sum}")
                penalty += 1000
        
        # 2.3 任务串开始时间约束 (检查一致性)
        for c in range(self.num_chains):
            calculated_start = sum(t * y[c, k, t] for k in range(self.num_evtols) for t in range(self.time_horizon))
            if abs(chain_start[c] - calculated_start) > 1e-6:
                violations.append(f"任务串{c}开始时间约束违反")
                penalty += 500
        
        # 2.4 任务串内任务的时间约束 - 严格的时序约束
        for c, chain in enumerate(self.task_chains):
            chain_start_time = chain_start[c]
            
            # 约束1: 第一个任务的开始时间必须等于任务串开始时间
            if len(chain) > 0:
                first_task = chain[0]
                if abs(task_start[first_task] - chain_start_time) > 1e-6:
                    violations.append(f"任务串{c}的第一个任务{first_task}开始时间({task_start[first_task]})不等于任务串开始时间({chain_start_time})")
                    penalty += 2000  # 严重违反，高惩罚
            
            # 约束2: 后续任务的开始时间必须 >= 前一个任务结束时间 + 40分钟最小间隔
            if len(chain) > 1:
                for i in range(len(chain) - 1):
                    curr_task = chain[i]
                    next_task = chain[i + 1]
                    min_required_gap = 40  # 最小间隔40分钟，允许更大间隔
                    min_start_time = task_end[curr_task] + min_required_gap
                    
                    if task_start[next_task] < min_start_time:
                        gap_violation = min_start_time - task_start[next_task]
                        violations.append(f"任务串{c}内任务{curr_task}->{next_task}间隔不足: 需要>={min_start_time}, 实际{task_start[next_task]}, 缺少{gap_violation}分钟")
                        penalty += 2000 + gap_violation * 10  # 基础惩罚 + 违反程度惩罚
        
        # 2.5 eVTOL冲突约束 - 检查同一eVTOL在同一时间不能执行多个任务
        for k in range(self.num_evtols):
            # 收集该eVTOL执行的所有任务及其时间
            evtol_tasks = []
            for c, chain in enumerate(self.task_chains):
                if b_chain_evtol[c, k] == 1:  # 该任务串分配给了eVTOL k
                    for task_id in chain:
                        evtol_tasks.append((task_id, task_start[task_id], task_end[task_id]))
            
            # 检查时间重叠
            for i in range(len(evtol_tasks)):
                for j in range(i + 1, len(evtol_tasks)):
                    task1_id, start1, end1 = evtol_tasks[i]
                    task2_id, start2, end2 = evtol_tasks[j]
                    
                    # 检查时间重叠 + 40分钟间隔（与任务串内间隔保持一致）
                    if not (end1 + 40 <= start2 or end2 + 40 <= start1):
                        violations.append(f"eVTOL{k}的任务{task1_id}和{task2_id}时间冲突")
                        penalty += 1200
        
        # 2.6 高度层防撞约束
        for i in range(self.num_tasks):
            for j in range(i + 1, self.num_tasks):
                for h in range(self.num_routes):
                    if z[i, h] == 1 and z[j, h] == 1:
                        # 两个任务使用相同航线，检查时间重叠
                        if not (task_end[i] <= task_start[j] or task_end[j] <= task_start[i]):
                            violations.append(f"任务{i}和{j}在航线{h}上时间重叠")
                            penalty += 1500
        
        # 2.7 任务串间隔约束
        for k in range(self.num_evtols):
            for c1 in range(self.num_chains):
                for c2 in range(c1 + 1, self.num_chains):
                    if both_assigned[c1, c2, k] == 1:
                        interval_satisfied = (chain_end[c1] + self.chain_interval_time <= chain_start[c2] or 
                                            chain_end[c2] + self.chain_interval_time <= chain_start[c1])
                        if not interval_satisfied:
                            violations.append(f"eVTOL{k}的任务串{c1}和{c2}间隔不足")
                            penalty += 1000
        
        # 2.8 任务时间窗约束
        for i in range(self.num_tasks):
            # 检查最早开始时间约束
            if task_start[i] < self.tasks[i]['earliest_start']:
                time_violation = self.tasks[i]['earliest_start'] - task_start[i]
                violations.append(f"任务{i}违反最早开始时间: 需要>={self.tasks[i]['earliest_start']}, 实际{task_start[i]}, 提前{time_violation}分钟")
                penalty += 1500 + time_violation * 20  # 基础惩罚 + 违反程度惩罚
            
            # 检查任务是否在时间范围内
            if task_start[i] >= self.time_horizon:
                violations.append(f"任务{i}开始时间({task_start[i]})超出时间范围({self.time_horizon})")
                penalty += 3000
            if task_end[i] >= self.time_horizon:
                violations.append(f"任务{i}结束时间({task_end[i]})超出时间范围({self.time_horizon})")
                penalty += 3000
        
        return violations, penalty
    
    def fitness(self, x):
        """
        计算适应度函数 - 多目标优化
        
        返回: [目标1, 目标2] = [总能耗, 总延误] + 约束惩罚
        对应gurobi_multi的epsilon约束方法中的两个独立目标
        """
        try:
            # 解码个体
            solution = self._decode_solution(x)
            if solution is None:
                return [50000.0, 50000.0]
            
            # 计算目标函数 (与gurobi_multi相同)
            total_energy, total_delay = self._calculate_objectives(solution)
            
            # 检查约束
            violations, penalty = self._check_constraints(solution)
            
            # 返回带惩罚的原始目标函数值 (无权重组合)
            objective1 = total_energy + penalty      # 目标1: 总能耗
            objective2 = total_delay + penalty       # 目标2: 总延误
            
            return [objective1, objective2]
            
        except Exception as e:
            print(f"适应度计算错误: {e}")
            return [50000.0, 50000.0]


def solve_rlts_nsga2_multi_objective(tasks, evtols, task_chains, time_horizon=720, 
                                    population_size=100, generations=200, verbose=True,
                                    tabu_search_frequency=10, tabu_search_intensity=0.3,
                                    enable_qlearning=True):
    """
    使用RLTS-NSGA-II算法求解eVTOL调度问题
    
    RLTS-NSGA-II (Reinforcement Learning and Tabu Search enhanced NSGA-II) 是基于NSGA-II的改进算法
    主要改进：
    1. 基于最拥挤关键路径的局部禁忌搜索策略 ✅ 已实现
    2. Q-learning实现的交叉率与变异率自适应调节 ✅ 已实现
    
    参数:
        tasks: 任务列表
        evtols: eVTOL列表  
        task_chains: 任务串列表
        time_horizon: 时间范围
        population_size: 种群大小
        generations: 进化代数
        verbose: 是否显示详细信息
        tabu_search_frequency: 禁忌搜索应用频率（每N代应用一次）
        tabu_search_intensity: 禁忌搜索强度（对种群中前X%的个体应用）
        enable_qlearning: 是否启用Q-learning参数自适应调节
    """
    
    if verbose:
        print(f"=== RLTS-NSGA-II 改进遗传算法多目标优化求解 (纯整数编码) ===")
    
    # 确保population_size符合要求
    if population_size < 8:
        population_size = 8
    if population_size % 4 != 0:
        population_size = ((population_size // 4) + 1) * 4
    
    try:
        # 创建问题实例
        problem = eVTOLSchedulingProblem(tasks, evtols, task_chains, time_horizon)
        
        # 创建PyGMO问题对象
        pg_problem = pg.problem(problem)
        
        # TODO: 这里将实现RLTS-NSGA-II的改进算法
        # 目前先使用NSGA-II作为基础，后续将添加改进
        
        # RLTS-NSGA-II - 基于NSGA-II进行改进，使用动态参数或默认参数
        # 初始参数（如果不使用Q-learning则使用这些默认值）
        initial_cr = 0.75
        initial_m = 15.0/problem.dimensions
        
        algo_obj = pg.nsga2(
            gen=1,  # 每次只进化1代
            cr=initial_cr,    # 交叉率（将动态调整）
            eta_c=8,    # 交叉分布指数
            m=initial_m,  # 变异率（将动态调整）
            eta_m=5     # 变异分布指数
        )
        
        # 使用 PyGMO 求解
        algo = pg.algorithm(algo_obj)
        
        # 创建种群
        pop = pg.population(pg_problem, population_size)
        
        # 初始化禁忌搜索管理器
        tabu_manager = TabuSearchManager(tabu_tenure=7, max_moves_per_iteration=3)
        
        # 初始化Q-learning参数控制器
        qlearning_controller = None
        if enable_qlearning:
            qlearning_controller = QLearningParameterController(max_generations=generations)
        
        if verbose:
            print(f"初始种群大小: {len(pop)}")
            print(f"决策变量维度: {problem.dimensions}")
            print(f"禁忌搜索频率: 每{tabu_search_frequency}代应用一次")
            print(f"禁忌搜索强度: 对前{tabu_search_intensity*100:.0f}%的个体应用")
            print(f"Q-learning参数调节: {'启用' if enable_qlearning else '禁用'}")
            print(f"开始RLTS-NSGA-II进化 {generations} 代...")
            print("=" * 80)
    
        # 记录进化过程数据
        evolution_data = {
            'generations': [],
            'pareto_count': [],
            'min_energy': [],
            'avg_energy': [],
            'min_delay': [],
            'avg_delay': [],
            'hypervolume': [],
            'pareto_fronts': []  # 存储每代的帕累托前沿
        }
        
        # 逐代进化并打印信息
        for gen in range(generations):
            # Q-learning动态参数调节
            if qlearning_controller is not None:
                # 获取当前种群适应度
                current_fitness = pop.get_f()
                
                # 获取动态调整的参数
                dynamic_cr, dynamic_m = qlearning_controller.get_parameters(
                    current_fitness, gen)
                
                # 创建新的算法实例（使用动态参数）
                algo_obj = pg.nsga2(
                    gen=1,
                    cr=dynamic_cr,
                    eta_c=8,
                    m=dynamic_m,
                    eta_m=5
                )
                algo = pg.algorithm(algo_obj)
                
                if verbose and gen % 10 == 0:  # 每10代显示一次参数信息
                    learning_stats = qlearning_controller.get_learning_stats()
                    print(f"    📊 第{gen+1}代Q-learning参数: cr={dynamic_cr:.3f}, m={dynamic_m:.3f}, "
                          f"状态={learning_stats['current_state']}, F值={learning_stats['previous_f_value']:.6f}")
            
            # 进化一代
            pop = algo.evolve(pop)
            
            # 应用禁忌搜索改进（每隔一定代数，但不在最后一代）
            if (gen + 1) % tabu_search_frequency == 0 and gen < generations - 1:
                if verbose:
                    print(f"    🔍 第{gen+1}代应用禁忌搜索改进...")
                
                # 获取当前种群
                current_individuals = pop.get_x()
                current_fitness = pop.get_f()
                
                # 选择前X%的个体进行禁忌搜索改进
                num_to_improve = max(1, int(len(current_individuals) * tabu_search_intensity))
                
                # 根据适应度排序选择要改进的个体（选择适应度较好的）
                fitness_scores = [f[0] + f[1] for f in current_fitness]  # 简单加权和
                sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
                
                improved_individuals = []
                improvement_count = 0
                
                for i in range(num_to_improve):
                    idx = sorted_indices[i]
                    original_individual = current_individuals[idx]
                    
                    # 应用禁忌搜索改进
                    improved_individual = tabu_manager.local_search_improvement(
                        original_individual, problem, max_iterations=3)
                    
                    # 检查是否有改进
                    original_fitness = problem.fitness(original_individual)
                    improved_fitness = problem.fitness(improved_individual)
                    
                    if tabu_manager._is_better_fitness(improved_fitness, original_fitness):
                        improved_individuals.append((idx, improved_individual))
                        improvement_count += 1
                
                # 将改进的个体重新插入种群
                if improved_individuals:
                    for idx, improved_individual in improved_individuals:
                        # 替换原个体
                        pop.set_x(idx, improved_individual)
                    
                    if verbose:
                        print(f"    ✅ 禁忌搜索改进了 {improvement_count}/{num_to_improve} 个个体")
                else:
                    if verbose:
                        print(f"    ⚠️  禁忌搜索未找到改进")
            
            
            if verbose:
                # 获取当前种群的适应度值
                fitness_values = pop.get_f()
                
                # 计算统计信息
                fitness1 = fitness_values[:, 0]  # 适应度1 (能耗+惩罚)
                fitness2 = fitness_values[:, 1]  # 适应度2 (延误+惩罚)
                
                min_fitness1 = np.min(fitness1)
                max_fitness1 = np.max(fitness1)
                avg_fitness1 = np.mean(fitness1)
                
                min_fitness2 = np.min(fitness2)
                max_fitness2 = np.max(fitness2)
                avg_fitness2 = np.mean(fitness2)
                
                # 先计算所有个体的真实目标值
                current_individuals = pop.get_x()
                all_real_objectives = []
                valid_indices = []
                
                for idx in range(len(current_individuals)):
                    individual = current_individuals[idx]
                    solution = problem._decode_solution(individual)
                    if solution is not None:
                        real_energy, real_delay = problem._calculate_objectives(solution)
                        all_real_objectives.append([real_energy, real_delay])
                        valid_indices.append(idx)
                
                # 基于真实目标值进行帕累托筛选 (用于显示和统计)
                if all_real_objectives:
                    real_objectives_array = np.array(all_real_objectives)
                    real_pareto_indices = pg.non_dominated_front_2d(real_objectives_array)
                    pareto_count = len(real_pareto_indices)
                    
                    # 获取帕累托前沿的真实目标值
                    pareto_real_objectives = [all_real_objectives[i] for i in real_pareto_indices]
                    pareto_front_points = [(obj[0], obj[1]) for obj in pareto_real_objectives]
                    
                    # 计算真实目标值的范围 (用于显示)
                    pareto_energies = [obj[0] for obj in pareto_real_objectives]
                    pareto_delays = [obj[1] for obj in pareto_real_objectives]
                    pareto_energy_range = f"{min(pareto_energies):.1f}-{max(pareto_energies):.1f}"
                    pareto_delay_range = f"{min(pareto_delays):.1f}-{max(pareto_delays):.1f}"
                    
                    # 计算超体积 (Hypervolume) - 使用真实目标值
                    ref_point = [max(pareto_energies) * 1.1, max(pareto_delays) * 1.1]
                    try:
                        hv = pg.hypervolume(real_objectives_array[real_pareto_indices])
                        hypervolume = hv.compute(ref_point)
                    except:
                        hypervolume = 0.0
                else:
                    pareto_count = 0
                    pareto_energy_range = "N/A"
                    pareto_delay_range = "N/A"
                    hypervolume = 0.0
                    pareto_front_points = []
                
                # 记录进化数据
                evolution_data['generations'].append(gen + 1)
                evolution_data['pareto_count'].append(pareto_count)
                evolution_data['min_energy'].append(min_fitness1)
                evolution_data['avg_energy'].append(avg_fitness1)
                evolution_data['min_delay'].append(min_fitness2)
                evolution_data['avg_delay'].append(avg_fitness2)
                evolution_data['hypervolume'].append(hypervolume)
                evolution_data['pareto_fronts'].append(pareto_front_points)
                
                # 打印当代信息
                print(f"第{gen+1:3d}代 | "
                      f"帕累托解: {pareto_count:2d} | "
                      f"适应度1: {min_fitness1:6.1f}-{max_fitness1:6.1f} (avg:{avg_fitness1:6.1f}) | "
                      f"适应度2: {min_fitness2:6.1f}-{max_fitness2:6.1f} (avg:{avg_fitness2:6.1f}) | "
                      f"前沿能耗: {pareto_energy_range} | "
                      f"前沿延误: {pareto_delay_range}")
                
                # 每10代或最后一代打印详细信息
                if (gen + 1) % 10 == 0 or gen == generations - 1:
                    print("-" * 80)
                    print(f"第{gen+1}代详细统计:")
                    print(f"  种群大小: {len(pop)}")
                    print(f"  帕累托前沿解数: {pareto_count}")
                    print(f"  适应度1统计: 最小={min_fitness1:.1f}, 最大={max_fitness1:.1f}, 平均={avg_fitness1:.1f}")
                    print(f"  适应度2统计: 最小={min_fitness2:.1f}, 最大={max_fitness2:.1f}, 平均={avg_fitness2:.1f}")
                    
                    if pareto_count > 0:
                        print(f"  帕累托前沿真实能耗范围: {pareto_energy_range}")
                        print(f"  帕累托前沿真实延误范围: {pareto_delay_range}")
                    
                    print("-" * 80)
        
        # 提取最终帕累托前沿 - 使用真实目标值筛选
        final_fitness = pop.get_f()
        final_individuals = pop.get_x()
        
        # 计算所有个体的真实目标值
        real_objectives = []
        valid_solutions = []
        for idx in range(len(final_individuals)):
            individual = final_individuals[idx]
            fitness = final_fitness[idx]
            
            solution = problem._decode_solution(individual)
            if solution is not None:
                total_energy, total_delay = problem._calculate_objectives(solution)
                real_objectives.append([total_energy, total_delay])
                valid_solutions.append({
                    'energy': total_energy,
                    'delay': total_delay,
                    'fitness': fitness,
                    'individual': individual,
                    'idx': idx
                })
        
        # 使用真实目标值进行帕累托前沿筛选
        if real_objectives:
            real_objectives_array = np.array(real_objectives)
            real_pareto_indices = pg.non_dominated_front_2d(real_objectives_array)
            pareto_front = [valid_solutions[i] for i in real_pareto_indices]
        else:
            pareto_front = []
        
        if verbose:
            print("\n🎉 RLTS-NSGA-II进化完成!")
            print(f"最终帕累托前沿解数量: {len(pareto_front)}")
            if pareto_front:
                energies = [sol['energy'] for sol in pareto_front]
                delays = [sol['delay'] for sol in pareto_front]
                print(f"最终能耗范围: {min(energies):.1f} - {max(energies):.1f}")
                print(f"最终延误范围: {min(delays):.1f} - {max(delays):.1f}")
                print("注: 进化过程和最终结果的帕累托前沿都基于真实目标值筛选，适应度仅用于进化选择")
                
                # 🔥 新增：详细打印每个帕累托前沿解
                print(f"\n📊 RLTS-NSGA-II 详细帕累托前沿解集:")
                print("=" * 80)
                print(f"{'解序号':<6} {'总能耗':<12} {'总延误':<12} {'适应度1':<12} {'适应度2':<12} {'编码'}")
                print("-" * 80)
                
                # 按能耗排序显示
                sorted_pareto = sorted(pareto_front, key=lambda x: x['energy'])
                for i, sol in enumerate(sorted_pareto, 1):
                    # 获取编码字符串（前10个基因）
                    encoding_preview = ', '.join([f'{int(x)}' for x in sol['individual'][:10]])
                    if len(sol['individual']) > 10:
                        encoding_preview += "..."
                    
                    print(f"{i:<6} {sol['energy']:<12.1f} {sol['delay']:<12.1f} "
                          f"{sol['fitness'][0]:<12.2f} {sol['fitness'][1]:<12.2f} [{encoding_preview}]")
                
                print("=" * 80)
                print(f"说明: 编码为决策变量序列，适应度包含惩罚项，真实目标值为纯目标函数值")
        
        # 选择一个前沿解进行可视化
        if pareto_front:
            # 选择延误最小的解进行可视化
            best_solution = min(pareto_front, key=lambda x: x['delay'])
            selected_schedule = _convert_rlts_nsga2_solution_to_schedule(best_solution['individual'], problem, tasks)
            
            if verbose and selected_schedule:
                print(f"\n选择前沿解进行可视化 (延误最小解: 能耗={best_solution['energy']:.1f}, 延误={best_solution['delay']:.1f})")
                # 打印用于生成调度图和调度表的编码
                encoding_str = ', '.join([f'{int(x)}' for x in best_solution['individual']])
                print(f"📊 RLTS-NSGA-II - 生成调度图和调度表的编码:")
                print(f"[{encoding_str}]")
                print("=" * 60)
                _visualize_rlts_nsga2_solution(selected_schedule, "RLTS-NSGA-II")
        
        # 🔥 新增：性能指标计算
        try:
            from performance_metrics import evaluate_algorithm_performance
            from data_definitions.tasks import DEFAULT_SCALE
            
            # 提取算法前沿用于指标计算
            algorithm_front = [(sol['energy'], sol['delay']) for sol in pareto_front]
            
            # 计算性能指标
            performance_result = evaluate_algorithm_performance(
                algorithm_pareto_front=algorithm_front,
                scale=DEFAULT_SCALE,
                algorithm_name="RLTS-NSGA-II"
            )
            
        except ImportError:
            print("⚠️  性能指标模块未找到，跳过指标计算")
            performance_result = None
        except Exception as e:
            print(f"⚠️  性能指标计算出错: {e}")
            performance_result = None
        
        return {
            'pareto_front': pareto_front,
            'problem': problem,
            'population': pop,
            'algorithm': algo,
            'evolution_data': evolution_data,
            'performance_metrics': performance_result
        }
        
    except Exception as e:
        print(f"RLTS-NSGA-II优化求解错误: {e}")
        return None


def _convert_rlts_nsga2_solution_to_schedule(individual, problem, tasks):
    """
    将RLTS-NSGA-II解转换为标准调度格式
    """
    try:
        solution = problem._decode_solution(individual)
        if not solution:
            return []
        
        schedule = []
        for i in range(problem.num_tasks):
            # 找到执行此任务的eVTOL
            evtol_id = None
            for c, chain in enumerate(problem.task_chains):
                if i in chain:
                    for k in range(problem.num_evtols):
                        for t in range(problem.time_horizon):
                            if solution['y'][c, k, t] == 1:
                                evtol_id = k
                                break
                    break
            
            # 找到选择的航线
            route_id = None
            for h in range(problem.num_routes):
                if solution['z'][i, h] == 1:
                    route_id = h
                    break
            
            if evtol_id is not None and route_id is not None:
                delay = max(0, solution['task_start'][i] - tasks[i]['earliest_start'])
                schedule.append({
                    "task_id": i,
                    "evtol_id": evtol_id,
                    "start_time": int(solution['task_start'][i]),
                    "end_time": int(solution['task_end'][i]),
                    "route": route_id,
                    "from": tasks[i]['from'],
                    "to": tasks[i]['to'],
                    "delay": delay
                })
        
        return schedule
    except Exception as e:
        print(f"转换RLTS-NSGA-II解失败: {e}")
        return []


def _visualize_rlts_nsga2_solution(schedule, algorithm_name="RLTS-NSGA-II"):
    """
    可视化RLTS-NSGA-II算法解
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from visualization import visualize_schedule_table, visualize_schedule_gantt, clean_algorithm_name_for_filename
    
    if schedule:
        clean_name = clean_algorithm_name_for_filename(algorithm_name)
        visualize_schedule_table(schedule, f"{algorithm_name} (延误最小解)", f"picture_result/evtol_schedule_table_{clean_name}_min_delay.png")
        visualize_schedule_gantt(schedule, f"{algorithm_name} (延误最小解)", f"picture_result/evtol_schedule_{clean_name}_min_delay.png")


def visualize_evolution_curves(evolution_data, save_path="picture_result/evolution_curves_rlts_nsga2.png"):
    """
    可视化RLTS-NSGA-II进化曲线
    
    参数:
        evolution_data: 进化过程数据
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'FangSong'
    
    generations = evolution_data['generations']
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 帕累托前沿解数量变化
    ax1.plot(generations, evolution_data['pareto_count'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('代数')
    ax1.set_ylabel('帕累托前沿解数量')
    ax1.set_title('帕累托前沿解数量进化曲线')
    ax1.grid(True, alpha=0.3)
    
    # 2. 适应度1指标进化 (能耗+惩罚)
    ax2.plot(generations, evolution_data['min_energy'], 'r-', linewidth=2, label='最小适应度1', marker='o', markersize=3)
    ax2.plot(generations, evolution_data['avg_energy'], 'g-', linewidth=2, label='平均适应度1', marker='s', markersize=3)
    ax2.set_xlabel('代数')
    ax2.set_ylabel('适应度1 (能耗+惩罚)')
    ax2.set_title('适应度1指标进化曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 适应度2指标进化 (延误+惩罚)
    ax3.plot(generations, evolution_data['min_delay'], 'purple', linewidth=2, label='最小适应度2', marker='o', markersize=3)
    ax3.plot(generations, evolution_data['avg_delay'], 'orange', linewidth=2, label='平均适应度2', marker='s', markersize=3)
    ax3.set_xlabel('代数')
    ax3.set_ylabel('适应度2 (延误+惩罚)')
    ax3.set_title('适应度2指标进化曲线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 超体积进化
    ax4.plot(generations, evolution_data['hypervolume'], 'brown', linewidth=2, marker='o', markersize=4)
    ax4.set_xlabel('代数')
    ax4.set_ylabel('超体积')
    ax4.set_title('帕累托前沿质量进化曲线 (超体积)')
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"进化曲线已保存到: {save_path}")


def visualize_pareto_front_evolution(evolution_data, save_path="picture_result/pareto_front_evolution_rlts_nsga2.png", 
                                   show_generations=[1, 10, 50, 100, -1]):
    """
    可视化帕累托前沿的进化过程
    
    参数:
        evolution_data: 进化过程数据
        save_path: 保存路径
        show_generations: 要显示的代数 (-1表示最后一代)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'FangSong'
    
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, gen_idx in enumerate(show_generations):
        if gen_idx == -1:
            gen_idx = len(evolution_data['pareto_fronts']) - 1
            label = f"第{len(evolution_data['pareto_fronts'])}代 (最终)"
        else:
            gen_idx = gen_idx - 1  # 转换为数组索引
            label = f"第{gen_idx + 1}代"
        
        if gen_idx < len(evolution_data['pareto_fronts']):
            pareto_points = evolution_data['pareto_fronts'][gen_idx]
            if pareto_points:
                energies, delays = zip(*pareto_points)
                plt.scatter(energies, delays, 
                          c=colors[i % len(colors)], 
                          marker=markers[i % len(markers)],
                          s=60, alpha=0.7, label=label,
                          edgecolors='black', linewidth=0.5)
    
    plt.xlabel('总能耗 (真实目标值)')
    plt.ylabel('总延误时间 (分钟, 真实目标值)')
    plt.title('RLTS-NSGA-II帕累托前沿进化过程 (基于真实目标值筛选)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"帕累托前沿进化图已保存到: {save_path}") 