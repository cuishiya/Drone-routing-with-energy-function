#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
四种能耗模型性能评估脚本

使用RMSE、MAE、R²等指标评估四种能耗模型的性能：
1. **物理模型 (Physical)**: 基于物理公式P(q) = k * (W + m + q)^(3/2)的理论模型 
2. 线性回归模型 (LinearRegressionEnergyModel) 
3. LightGBM树模型 (TreeBasedEnergyModel)
4. PyTorch深度学习模型 (DeepLearningEnergyModel)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import time
import os

from mtdrp_energy_model import (
    create_energy_model, DroneParameters, MTDRPInstance,
    Customer, Depot, NonlinearEnergyModel
)

class EnergyModelEvaluator:
    """能耗模型评估器"""
    
    def __init__(self, instance: MTDRPInstance):
        """
        初始化评估器
        
        参数:
            instance: MTDRP问题实例
        """
        self.instance = instance
        self.models = {}
        self.evaluation_results = {}
        
    def load_test_data_from_folder(self):
        """
        从test_data文件夹加载专门的测试数据
        
        返回:
            features: 特征矩阵 [distance, payload, wind_speed, wind_angle, temperature, humidity]
            targets: 目标值（真实的电池消耗数据）
        """
        print("正在加载test_data文件夹中的测试数据...")
        
        # 测试数据路径
        test_file_path = os.path.join("Drone_energy_dataset", "test_data", "flightRecord.xlsx")
        
        try:
            # 读取测试数据
            df = pd.read_excel(test_file_path)
            print(f"加载测试数据: {len(df)} 条记录")
            print(f"数据列: {list(df.columns)}")
            
            # 检查必要的列是否存在
            required_columns = ['Distance (m)', 'Battery Used (kWh)', 'Payload (kg)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"[ERROR] 缺少必要的列: {missing_columns}")
                print("[FATAL] 数据格式不正确，程序退出")
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            # 过滤有效数据
            valid_mask = (
                (df['Distance (m)'] > 0) & 
                (df['Battery Used (kWh)'] > 0) & 
                (df['Payload (kg)'] >= 0)
            )
            df_valid = df[valid_mask].copy()
            
            if len(df_valid) == 0:
                print("[ERROR] 没有有效的测试数据")
                print("[FATAL] 无有效数据，程序退出")
                raise ValueError("没有有效的测试数据")
            
            print(f"有效测试记录: {len(df_valid)} 条")
            
            # 计算实际飞行时间
            df_valid['Start Time'] = pd.to_datetime(df_valid['Start Time'])
            df_valid['End Time'] = pd.to_datetime(df_valid['End Time'])
            df_valid['Flight Time (s)'] = (df_valid['End Time'] - df_valid['Start Time']).dt.total_seconds()
            
            print(f"飞行时间统计:")
            print(f"  时间范围: {df_valid['Flight Time (s)'].min():.0f} - {df_valid['Flight Time (s)'].max():.0f} 秒")
            print(f"  平均飞行时间: {df_valid['Flight Time (s)'].mean():.0f} 秒")
            
            # 准备特征矩阵
            distances = df_valid['Distance (m)'].values
            payloads = df_valid['Payload (kg)'].values
            flight_times = df_valid['Flight Time (s)'].values  # 实际飞行时间
            # 转换为时间戳（秒）
            start_times = (df_valid['Start Time'] - pd.Timestamp('1970-01-01')).dt.total_seconds().values
            end_times = (df_valid['End Time'] - pd.Timestamp('1970-01-01')).dt.total_seconds().values
            
            # 处理环境特征
            # 风速
            wind_speeds = df_valid.get('Avg Wind Speed', pd.Series([5.0] * len(df_valid))).fillna(5.0).values
            
            # 风向角度 - 如果有Wind Direction列，转换为角度；否则使用默认值
            if 'Wind Direction' in df_valid.columns:
                # 假设Wind Direction是方向描述，转换为角度
                wind_directions = df_valid['Wind Direction'].fillna('N').values
                wind_angles = []
                direction_map = {
                    'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
                    'S': 180, 'SW': 225, 'W': 270, 'NW': 315
                }
                for direction in wind_directions:
                    wind_angles.append(direction_map.get(str(direction).upper(), 90))
                wind_angles = np.array(wind_angles, dtype=float)
            else:
                wind_angles = np.full(len(df_valid), 90.0)  # 默认90度
            
            # 温度和湿度
            temperatures = df_valid.get('Avg Temperature', pd.Series([25.0] * len(df_valid))).fillna(25.0).values
            humidities = df_valid.get('Avg Humidity', pd.Series([60.0] * len(df_valid))).fillna(60.0).values
            
            features = np.column_stack([
                distances, payloads, wind_speeds, wind_angles, temperatures, humidities
            ])
            
            # 目标值是真实的电池消耗
            targets = df_valid['Battery Used (kWh)'].values
            
            print(f"特征矩阵形状: {features.shape}")
            print(f"目标值范围: {targets.min():.6f} - {targets.max():.6f} kWh")
            print(f"平均电池消耗: {targets.mean():.6f} kWh")
            print(f"距离范围: {distances.min():.0f} - {distances.max():.0f} m")
            print(f"载荷范围: {payloads.min():.1f} - {payloads.max():.1f} kg")
            print(f"温度范围: {temperatures.min():.1f} - {temperatures.max():.1f} °C")
            print(f"湿度范围: {humidities.min():.1f} - {humidities.max():.1f} %")
            
            # 返回特征矩阵、目标值、飞行时间和时间戳
            return features, targets, flight_times, start_times, end_times
            
        except Exception as e:
            print(f"[ERROR] 读取测试数据文件时出错: {e}")
            print("[FATAL] 无法加载测试数据，程序退出")
            raise e
    
    
    def load_models(self):
        """加载四种能耗模型"""
        model_types = ["physical", "linear", "tree", "deep"]
        
        for model_type in model_types:
            try:
                print(f"正在加载 {model_type} 模型...")
                model = create_energy_model(model_type, self.instance)
                self.models[model_type] = model
                print(f"[OK] {model_type} 模型加载成功")
            except Exception as e:
                print(f"[ERROR] {model_type} 模型加载失败: {e}")
                self.models[model_type] = None
    
    def predict_energy(self, model, features, flight_times, start_times, end_times, model_name):
        """
        使用模型预测能耗
        
        参数:
            model: 能耗模型
            features: 特征矩阵 [distance, payload, wind_speed, wind_angle, temperature, humidity]
            flight_times: 实际飞行时间数组 [秒]
            start_times: 开始时间戳数组
            end_times: 结束时间戳数组
            model_name: 模型名称
            
        返回:
            预测的能耗数组
        """
        predictions = []
        
        for i, feature in enumerate(features):
            distance, payload = feature[0], feature[1]
            
            try:
                if model_name == "physical":
                    # 物理模型使用新接口: energy_consumption(payload, end_time, start_time)
                    energy = model.energy_consumption(payload, end_times[i], start_times[i])
                else:
                    # 其他模型使用标准接口
                    energy = model.energy_consumption(payload, distance)
                predictions.append(energy)
            except Exception as e:
                print(f"[WARNING] 预测失败: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """
        计算评估指标
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            
        返回:
            metrics: 评估指标字典
        """
        # 过滤掉无效预测
        valid_mask = (y_pred > 0) & (y_true > 0)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        if len(y_true_valid) == 0:
            return {
                'RMSE': float('inf'),
                'MAE': float('inf'),
                'R2': -float('inf'),
                'valid_samples': 0
            }
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        r2 = r2_score(y_true_valid, y_pred_valid)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'valid_samples': len(y_true_valid)
        }
        
        return metrics
    
    def evaluate_models(self, use_real_data: bool = True):
        """
        评估所有模型的性能
        
        参数:
            use_real_data: 是否使用真实数据，False则使用合成数据
        """
        print(f"\n{'='*80}")
        print("四种能耗模型性能评估")
        print(f"{'='*80}")
        
        # 加载测试数据
        if use_real_data:
            print("使用test_data文件夹中的测试数据进行评估...")
            features, targets, flight_times, start_times, end_times = self.load_test_data_from_folder()
            # 保存为实例变量，供绘图使用
            self.flight_times = flight_times
            self.start_times = start_times
            self.end_times = end_times
        else:
            print("[ERROR] 不支持合成数据模式")
            raise ValueError("仅支持真实数据评估模式")
        
        print(f"测试数据准备完成，样本数量: {len(targets)}")
        
        # 加载模型
        self.load_models()
        
        # 评估每个模型
        print(f"\n{'='*80}")
        print("模型性能评估结果")
        print(f"{'='*80}")
        
        results_data = []
        
        for model_name, model in self.models.items():
            if model is None:
                print(f"\n[SKIP] {model_name} 模型未加载，跳过评估")
                continue
            
            print(f"\n正在评估 {model_name} 模型...")
            
            # 记录预测时间
            start_time_eval = time.time()
            predictions = self.predict_energy(model, features, flight_times, start_times, end_times, model_name)
            prediction_time = time.time() - start_time_eval
            
            # 计算评估指标
            metrics = self.calculate_metrics(targets, predictions, model_name)
            metrics['prediction_time'] = prediction_time
            metrics['avg_time_per_sample'] = prediction_time / len(targets) * 1000  # ms
            
            self.evaluation_results[model_name] = metrics
            
            # 添加到结果数据
            results_data.append({
                '模型': model_name,
                'RMSE': f"{metrics['RMSE']:.6f}",
                'MAE': f"{metrics['MAE']:.6f}",
                'R2': f"{metrics['R2']:.6f}",
                '预测时间(s)': f"{metrics['prediction_time']:.3f}",
                '单样本时间(ms)': f"{metrics['avg_time_per_sample']:.3f}",
                '有效样本': metrics['valid_samples']
            })
            
            print(f"[OK] {model_name} 模型评估完成")
            print(f"  RMSE: {metrics['RMSE']:.6f}")
            print(f"  MAE:  {metrics['MAE']:.6f}")
            print(f"  R2:   {metrics['R2']:.6f}")
            print(f"  预测时间: {metrics['prediction_time']:.3f}s")
        
        # 创建结果表格
        if results_data:
            results_df = pd.DataFrame(results_data)
            print(f"\n{'='*80}")
            print("性能对比总结")
            print(f"{'='*80}")
            print(results_df.to_string(index=False))
            
            # 保存结果
            self.save_results(results_df, features, targets)
        
        return self.evaluation_results
    
    def save_results(self, results_df, features, targets):
        """
        保存评估结果
        
        参数:
            results_df: 结果DataFrame
            features: 测试特征
            targets: 测试目标值
        """
        # 确保result目录存在
        if not os.path.exists('result'):
            os.makedirs('result')
        
        # 保存评估结果表格
        results_df.to_csv('result/energy_models_evaluation.csv', index=False, encoding='utf-8-sig')
        print(f"\n评估结果已保存为: result/energy_models_evaluation.csv")
        
        # 绘制性能对比图
        self.plot_performance_comparison()
        
        # 绘制预测vs真值散点图
        self.plot_prediction_scatter(features, targets)
    
    def plot_performance_comparison(self):
        """绘制性能对比图 - SCI论文风格"""
        if not self.evaluation_results:
            return
        
        try:
            # SCI论文风格设置
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'font.family': 'Times New Roman',
                'font.size': 11,
                'axes.labelsize': 12,
                'axes.titlesize': 13,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'axes.linewidth': 1.2,
                'axes.edgecolor': '#333333',
                'grid.alpha': 0.3,
                'grid.linestyle': '--',
            })
            
            # 准备数据
            model_names = list(self.evaluation_results.keys())
            # 模型名称美化
            display_names = {
                'physical': 'Physical',
                'linear': 'Linear Reg.',
                'tree': 'LightGBM',
                'deep': 'Deep Learning'
            }
            labels = [display_names.get(name, name) for name in model_names]
            
            rmse_values = [self.evaluation_results[name]['RMSE'] for name in model_names]
            mae_values = [self.evaluation_results[name]['MAE'] for name in model_names]
            r2_values = [self.evaluation_results[name]['R2'] for name in model_names]
            
            # SCI配色方案 (Nature风格)
            colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3'][:len(model_names)]
            
            # 创建图形
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # 通用柱状图样式
            bar_width = 0.6
            edge_color = '#333333'
            
            # (a) RMSE对比
            ax1 = axes[0]
            bars1 = ax1.bar(labels, rmse_values, width=bar_width, color=colors, 
                           edgecolor=edge_color, linewidth=1.2, alpha=0.85)
            ax1.set_ylabel('RMSE (kWh)', fontweight='bold')
            ax1.set_title('(a) RMSE Comparison', fontweight='bold', pad=10)
            ax1.set_ylim(0, max(rmse_values) * 1.25)
            ax1.tick_params(axis='x', rotation=30)
            # 添加数值标签
            for bar, value in zip(bars1, rmse_values):
                ax1.annotate(f'{value:.4f}', 
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # (b) MAE对比
            ax2 = axes[1]
            bars2 = ax2.bar(labels, mae_values, width=bar_width, color=colors,
                           edgecolor=edge_color, linewidth=1.2, alpha=0.85)
            ax2.set_ylabel('MAE (kWh)', fontweight='bold')
            ax2.set_title('(b) MAE Comparison', fontweight='bold', pad=10)
            ax2.set_ylim(0, max(mae_values) * 1.25)
            ax2.tick_params(axis='x', rotation=30)
            for bar, value in zip(bars2, mae_values):
                ax2.annotate(f'{value:.4f}',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # (c) R² 对比
            ax3 = axes[2]
            bars3 = ax3.bar(labels, r2_values, width=bar_width, color=colors,
                           edgecolor=edge_color, linewidth=1.2, alpha=0.85)
            ax3.set_ylabel('R² Score', fontweight='bold')
            ax3.set_title('(c) R² Comparison', fontweight='bold', pad=10)
            # R²可能有负值，设置合适的范围
            min_r2 = min(r2_values)
            max_r2 = max(r2_values)
            if min_r2 < 0:
                ax3.set_ylim(min_r2 * 1.2, max_r2 * 1.15)
                ax3.axhline(y=0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.5)
            else:
                ax3.set_ylim(0, max_r2 * 1.15)
            ax3.tick_params(axis='x', rotation=30)
            for bar, value in zip(bars3, r2_values):
                offset = 5 if value >= 0 else -15
                va = 'bottom' if value >= 0 else 'top'
                ax3.annotate(f'{value:.4f}',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, offset), textcoords='offset points',
                            ha='center', va=va, fontsize=9, fontweight='bold')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            
            plt.tight_layout(pad=2.0)
            plt.savefig('result/energy_models_performance_comparison.png', 
                       dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.show()
            
            # 恢复默认设置
            plt.rcParams.update(plt.rcParamsDefault)
            matplotlib.rcParams['font.family'] = 'SimHei'
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            print(f"性能对比图已保存为: result/energy_models_performance_comparison.png")
            
        except Exception as e:
            print(f"[ERROR] 绘制性能对比图失败: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_prediction_scatter(self, features, targets):
        """绘制预测vs真值散点图 - SCI论文风格"""
        if not self.evaluation_results:
            return
        
        try:
            # SCI论文风格设置
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'font.family': 'Times New Roman',
                'font.size': 11,
                'axes.labelsize': 12,
                'axes.titlesize': 13,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'axes.linewidth': 1.2,
                'axes.edgecolor': '#333333',
            })
            
            n_models = len([m for m in self.models.values() if m is not None])
            if n_models == 0:
                return
            
            # 模型名称美化
            display_names = {
                'physical': 'Physical Model',
                'linear': 'Linear Regression',
                'tree': 'LightGBM',
                'deep': 'Deep Learning'
            }
            
            # SCI配色方案
            colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 9))
            axes = axes.flatten()
            
            subplot_labels = ['(a)', '(b)', '(c)', '(d)']
            
            plot_idx = 0
            for model_name, model in self.models.items():
                if model is None or plot_idx >= 4:
                    continue
                
                # 获取预测值
                predictions = self.predict_energy(
                    model, features, 
                    getattr(self, 'flight_times', np.zeros(len(targets))),
                    getattr(self, 'start_times', np.zeros(len(targets))),
                    getattr(self, 'end_times', np.zeros(len(targets))),
                    model_name
                )
                
                # 过滤有效值
                valid_mask = (predictions > 0) & (targets > 0)
                valid_targets = targets[valid_mask]
                valid_predictions = predictions[valid_mask]
                
                if len(valid_targets) == 0:
                    continue
                
                ax = axes[plot_idx]
                
                # 绘制散点图 - 使用更专业的样式
                scatter = ax.scatter(valid_targets, valid_predictions, 
                                    alpha=0.5, color=colors[plot_idx], 
                                    s=25, edgecolors='white', linewidth=0.3)
                
                # 绘制理想线 (y=x) - 使用红色虚线
                min_val = min(valid_targets.min(), valid_predictions.min())
                max_val = max(valid_targets.max(), valid_predictions.max())
                margin = (max_val - min_val) * 0.05
                line_range = [min_val - margin, max_val + margin]
                ax.plot(line_range, line_range, 'r--', linewidth=1.5, 
                       label='Ideal (y=x)', alpha=0.8)
                
                # 设置坐标轴
                ax.set_xlim(line_range)
                ax.set_ylim(line_range)
                ax.set_xlabel('Actual Energy (kWh)', fontweight='bold')
                ax.set_ylabel('Predicted Energy (kWh)', fontweight='bold')
                
                # 标题
                title_name = display_names.get(model_name, model_name)
                ax.set_title(f'{subplot_labels[plot_idx]} {title_name}', 
                            fontweight='bold', pad=10, loc='left')
                
                # 添加统计信息框
                r2 = self.evaluation_results[model_name]['R2']
                rmse = self.evaluation_results[model_name]['RMSE']
                mae = self.evaluation_results[model_name]['MAE']
                
                stats_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
                props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='#666666', alpha=0.9, linewidth=1)
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=props, family='Times New Roman')
                
                # 添加样本数量
                n_samples = len(valid_targets)
                ax.text(0.95, 0.05, f'n = {n_samples}', transform=ax.transAxes,
                       fontsize=9, ha='right', va='bottom',
                       family='Times New Roman', style='italic')
                
                # 移除上边和右边的边框
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # 设置网格
                ax.grid(True, linestyle='--', alpha=0.3, color='gray')
                ax.set_axisbelow(True)
                
                plot_idx += 1
            
            # 隐藏多余的子图
            for i in range(plot_idx, 4):
                axes[i].set_visible(False)
            
            plt.tight_layout(pad=2.5)
            plt.savefig('result/energy_models_prediction_scatter.png', 
                       dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.show()
            
            # 恢复默认设置
            plt.rcParams.update(plt.rcParamsDefault)
            matplotlib.rcParams['font.family'] = 'SimHei'
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            print(f"预测散点图已保存为: result/energy_models_prediction_scatter.png")
            
        except Exception as e:
            print(f"[ERROR] 绘制预测散点图失败: {e}")
            import traceback
            traceback.print_exc()

def create_test_instance():
    """创建测试用的MTDRP实例"""
    depot = Depot(0, 5000.0, 5000.0)
    customers = [
        Customer(1, 4000.0, 4000.0, 2.0, 0.0, 120.0, 5.0),
        Customer(2, 6000.0, 6000.0, 3.0, 0.0, 120.0, 5.0),
        Customer(3, 3000.0, 7000.0, 1.5, 0.0, 120.0, 5.0),
    ]
    drone_params = DroneParameters()
    
    instance = MTDRPInstance(
        name="evaluation_instance",
        depot=depot,
        customers=customers,
        drone_params=drone_params,
        num_drones=3
    )
    
    return instance

def main():
    """主函数"""
    print("四种能耗模型性能评估程序")
    print("=" * 80)
    print("评估指标: RMSE, MAE, R2")
    print("=" * 80)
    
    # 创建测试实例
    instance = create_test_instance()
    print(f"测试实例: {instance.name}")
    
    # 创建评估器
    evaluator = EnergyModelEvaluator(instance)
    
    # 执行评估（使用真实数据）
    results = evaluator.evaluate_models(use_real_data=True)
    
    # 输出最终总结
    print(f"\n{'='*80}")
    print("评估完成！")
    print(f"{'='*80}")
    
    if results:
        # 找出最佳模型
        valid_results = {k: v for k, v in results.items() if v['valid_samples'] > 0}
        
        if valid_results:
            best_rmse_model = min(valid_results.keys(), key=lambda x: valid_results[x]['RMSE'])
            best_r2_model = max(valid_results.keys(), key=lambda x: valid_results[x]['R2'])
            
            print(f"最佳RMSE模型: {best_rmse_model} (RMSE: {valid_results[best_rmse_model]['RMSE']:.6f})")
            print(f"最佳R2模型: {best_r2_model} (R2: {valid_results[best_r2_model]['R2']:.6f})")
            
            print(f"\n结果文件:")
            print(f"- 评估结果表格: result/energy_models_evaluation.csv")
            print(f"- 性能对比图: result/energy_models_performance_comparison.png")
            print(f"- 预测散点图: result/energy_models_prediction_scatter.png")
        else:
            print("没有有效的评估结果")
    else:
        print("评估失败，请检查模型配置")

if __name__ == "__main__":
    main()
