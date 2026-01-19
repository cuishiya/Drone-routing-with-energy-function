#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
瞬时功率预测模型评估脚本

评估四种功率预测模型的性能:
1. PhysicalPowerModel - 基于物理公式
2. TreePowerModel - 基于LightGBM
3. DeepPowerModel - 基于PyTorch
4. LinearPowerModel - 基于线性回归

使用flightTrajectory数据进行评估
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import matplotlib.patches as mpatches

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================================
# 出版级图表配置 (Publication-Quality Plot Configuration)
# ============================================================================

def setup_publication_style():
    """
    设置出版级图表样式
    - 使用Times New Roman字体
    - 刻度线朝内
    - 四边显示刻度，仅左下显示数值
    - 无网格或极淡网格
    """
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'mathtext.fontset': 'stix',
        
        # 坐标轴设置
        'axes.linewidth': 1.0,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.labelweight': 'normal',
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        
        # 刻度设置 - 朝内
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.top': True,
        'xtick.bottom': True,
        'ytick.left': True,
        'ytick.right': True,
        
        # 图例设置
        'legend.fontsize': 9,
        'legend.frameon': False,
        'legend.loc': 'best',
        
        # 线条设置
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        
        # 图像设置
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # 其他
        'axes.unicode_minus': False,
    })


# Nature/Science 风格配色方案 (柔和且对比度高)
PUBLICATION_COLORS = {
    'blue': '#4878D0',      # 柔和蓝
    'orange': '#EE854A',    # 柔和橙
    'green': '#6ACC64',     # 柔和绿
    'red': '#D65F5F',       # 柔和红
    'purple': '#956CB4',    # 柔和紫
    'brown': '#8C613C',     # 柔和棕
    'pink': '#DC7EC0',      # 柔和粉
    'gray': '#797979',      # 中灰
    'yellow': '#D5BB67',    # 柔和黄
    'cyan': '#82C6E2',      # 柔和青
    'black': '#2D2D2D',     # 深灰黑
}

# 模型配色映射
MODEL_COLORS = {
    'tree': PUBLICATION_COLORS['green'],
    'deep': PUBLICATION_COLORS['blue'],
    'linear': PUBLICATION_COLORS['orange'],
    'physical': PUBLICATION_COLORS['gray'],
    'actual': PUBLICATION_COLORS['black'],
}

# 模型显示名称
MODEL_DISPLAY_NAMES = {
    'tree': 'LightGBM',
    'deep': 'Deep Learning',
    'linear': 'Linear Regression',
    'physical': 'Physical Model',
}

# 线型样式
MODEL_LINESTYLES = {
    'tree': '-',
    'deep': '--',
    'linear': '-.',
    'physical': ':',
    'actual': '-',
}

from mtdrp_energy_model import (
    create_power_model, PhysicalPowerModel, TreePowerModel,
    DeepPowerModel, LinearPowerModel, DroneParameters
)


class PowerModelEvaluator:
    """瞬时功率模型评估器"""
    
    def __init__(self):
        self.test_data = None
        self.models = {}
        self.results = {}
        
    def load_test_data(self, test_path: str = "Drone_energy_dataset/test_data/flightTrajectory.xlsx",
                       record_path: str = "Drone_energy_dataset/test_data/flightRecord.xlsx"):
        """
        加载测试数据，并通过Order ID关联flightRecord获取载荷
        """
        print("=" * 60)
        print("加载测试数据")
        print("=" * 60)
        
        if not os.path.exists(test_path):
            # 尝试使用训练数据的一部分作为测试
            alt_paths = [
                ("Drone_energy_dataset/UAS04028624/flightTrajectory.xlsx",
                 "Drone_energy_dataset/UAS04028624/flightRecord.xlsx"),
                ("Drone_energy_dataset/UAS04028648/flightTrajectory.xlsx",
                 "Drone_energy_dataset/UAS04028648/flightRecord.xlsx")
            ]
            for traj_path, rec_path in alt_paths:
                if os.path.exists(traj_path):
                    test_path = traj_path
                    record_path = rec_path
                    print(f"使用替代数据源: {test_path}")
                    break
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"测试数据文件不存在: {test_path}")
        
        print(f"加载轨迹数据: {test_path}")
        df = pd.read_excel(test_path)
        print(f"原始记录: {len(df)} 条")
        
        # 加载flightRecord获取载荷数据
        payload_map = {}
        if record_path and os.path.exists(record_path):
            print(f"加载航次数据: {record_path}")
            df_record = pd.read_excel(record_path)
            payload_map = df_record.set_index('Order ID')['Payload (kg)'].to_dict()
            print(f"航次记录: {len(df_record)} 条")
        
        # 通过Order ID关联载荷数据
        if 'Order ID' in df.columns and payload_map:
            df['payload'] = df['Order ID'].map(payload_map)
        else:
            df['payload'] = 0.0  # 默认载荷为0
        
        # 过滤有效数据
        valid_mask = (
            (df['Height'] > 0) &
            (df['GS (m/s)'] >= 0) &
            (df['Voltage'] > 0) &
            (df['Current'] > 0) &
            (df['Wind Speed'] >= 0) &
            (df['Temperature'] > -50) &
            (df['Humidity'] >= 0) & (df['Humidity'] <= 100) &
            (df['payload'].notna())
        )
        df_valid = df[valid_mask].copy()
        
        # 数据预处理
        df_valid['height'] = df_valid['Height'] / 10.0
        df_valid['VS'] = df_valid['VS (m/s)']
        df_valid['GS'] = df_valid['GS (m/s)']
        df_valid['wind_speed'] = df_valid['Wind Speed'] / 10.0
        df_valid['temperature'] = df_valid['Temperature']
        df_valid['humidity'] = df_valid['Humidity']
        
        # 计算风向夹角
        course = df_valid['Course'] / 10.0
        wind_direct = df_valid['Wind Direct']
        angle_diff = np.abs(course - wind_direct)
        df_valid['wind_angle'] = np.minimum(angle_diff, 360 - angle_diff)
        
        # 计算真实功率 [W]
        df_valid['true_power'] = (df_valid['Voltage'] / 1000.0) * (df_valid['Current'] / 1000.0)
        
        # 采样以加快评估速度（如果数据量太大）
        if len(df_valid) > 50000:
            df_valid = df_valid.sample(n=50000, random_state=42)
            print(f"采样后记录: {len(df_valid)} 条")
        
        self.test_data = df_valid
        print(f"有效测试记录: {len(self.test_data)} 条")
        print(f"真实功率范围: {self.test_data['true_power'].min():.2f} - {self.test_data['true_power'].max():.2f} W")
        print(f"平均功率: {self.test_data['true_power'].mean():.2f} W")
        print(f"载荷范围: {self.test_data['payload'].min():.2f} - {self.test_data['payload'].max():.2f} kg")
        
        return self.test_data
    
    def load_models(self):
        """
        加载所有功率预测模型
        """
        print("\n" + "=" * 60)
        print("加载功率预测模型")
        print("=" * 60)
        
        # 物理模型
        try:
            self.models['physical'] = PhysicalPowerModel(DroneParameters())
            print("[OK] PhysicalPowerModel 加载成功")
        except Exception as e:
            print(f"[ERROR] PhysicalPowerModel 加载失败: {e}")
        
        # 树模型
        try:
            self.models['tree'] = TreePowerModel()
            if self.models['tree'].model is not None:
                print("[OK] TreePowerModel 加载成功")
            else:
                print("[WARNING] TreePowerModel 模型文件未找到")
        except Exception as e:
            print(f"[ERROR] TreePowerModel 加载失败: {e}")
        
        # 深度学习模型
        try:
            self.models['deep'] = DeepPowerModel()
            if self.models['deep'].model is not None:
                print("[OK] DeepPowerModel 加载成功")
            else:
                print("[WARNING] DeepPowerModel 模型文件未找到")
        except Exception as e:
            print(f"[ERROR] DeepPowerModel 加载失败: {e}")
        
        # 线性回归模型
        try:
            self.models['linear'] = LinearPowerModel()
            if self.models['linear'].model is not None:
                print("[OK] LinearPowerModel 加载成功")
            else:
                print("[WARNING] LinearPowerModel 模型文件未找到")
        except Exception as e:
            print(f"[ERROR] LinearPowerModel 加载失败: {e}")
        
        print(f"\n成功加载 {len(self.models)} 个模型")
        return self.models
    
    def evaluate_models(self):
        """
        评估所有模型
        """
        print("\n" + "=" * 60)
        print("模型评估")
        print("=" * 60)
        
        if self.test_data is None:
            raise ValueError("请先加载测试数据")
        
        true_power = self.test_data['true_power'].values
        
        for model_name, model in self.models.items():
            print(f"\n评估 {model_name} 模型...")
            
            try:
                # 预测功率
                predictions = []
                for _, row in self.test_data.iterrows():
                    # 物理模型只需要payload参数，其他模型需要全部特征（包括payload）
                    if model_name == 'physical':
                        pred = model.predict_power(payload=row['payload'])
                    else:
                        pred = model.predict_power(
                            height=row['height'],
                            VS=row['VS'],
                            GS=row['GS'],
                            wind_speed=row['wind_speed'],
                            temperature=row['temperature'],
                            humidity=row['humidity'],
                            wind_angle=row['wind_angle'],
                            payload=row['payload']
                        )
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                
                # 计算评估指标
                rmse = np.sqrt(mean_squared_error(true_power, predictions))
                mae = mean_absolute_error(true_power, predictions)
                r2 = r2_score(true_power, predictions)
                
                self.results[model_name] = {
                    'predictions': predictions,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                }
                
                print(f"  RMSE: {rmse:.4f} W")
                print(f"  MAE:  {mae:.4f} W")
                print(f"  R2:   {r2:.6f}")
                
            except Exception as e:
                print(f"  [ERROR] 评估失败: {e}")
                self.results[model_name] = {'error': str(e)}
        
        return self.results
    
    def print_summary(self):
        """
        打印评估结果汇总
        """
        print("\n" + "=" * 60)
        print("评估结果汇总")
        print("=" * 60)
        
        print(f"\n{'模型':<12} {'RMSE (W)':<12} {'MAE (W)':<12} {'R2':<12}")
        print("-" * 48)
        
        # 按R²排序
        sorted_results = sorted(
            [(name, res) for name, res in self.results.items() if 'R2' in res],
            key=lambda x: x[1]['R2'],
            reverse=True
        )
        
        for model_name, res in sorted_results:
            print(f"{model_name:<12} {res['RMSE']:<12.4f} {res['MAE']:<12.4f} {res['R2']:<12.6f}")
        
        if sorted_results:
            best_model = sorted_results[0][0]
            print(f"\n最佳模型: {best_model} (R2 = {sorted_results[0][1]['R2']:.6f})")
    
    def plot_results(self, save_path: str = 'result/power_models_evaluation.png'):
        """
        绘制评估结果图 - 出版级质量
        """
        print("\n生成出版级评估图表...")
        
        # 设置出版级样式
        setup_publication_style()
        
        # 排除物理模型，只保留数据驱动模型
        valid_results = [(name, res) for name, res in self.results.items() 
                        if 'predictions' in res and name != 'physical']
        
        if not valid_results:
            print("[WARNING] 没有有效的评估结果可绘制")
            return
        
        # 按R2排序
        valid_results = sorted(valid_results, key=lambda x: x[1]['R2'], reverse=True)
        n_models = len(valid_results)
        
        # 黄金分割比例，适合双栏排版
        fig_width = 3.5 * n_models  # 每列约3.5英寸
        fig_height = 6.5  # 高度
        fig, axes = plt.subplots(2, n_models, figsize=(fig_width, fig_height))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        true_power = self.test_data['true_power'].values
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        
        for i, (model_name, res) in enumerate(valid_results):
            predictions = res['predictions']
            color = MODEL_COLORS.get(model_name, PUBLICATION_COLORS['gray'])
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            
            # ========== 上行: 散点图 ==========
            ax1 = axes[0, i]
            
            # 绘制散点 - 使用半透明小点
            ax1.scatter(true_power, predictions, 
                       alpha=0.15, s=3, c=color, 
                       edgecolors='none', rasterized=True)
            
            # 绘制理想线 (y=x)
            lims = [min(true_power.min(), predictions.min()),
                   max(true_power.max(), predictions.max())]
            ax1.plot(lims, lims, '-', color=PUBLICATION_COLORS['red'], 
                    linewidth=1.5, alpha=0.8, zorder=10)
            
            # 设置坐标轴范围
            margin = (lims[1] - lims[0]) * 0.02
            ax1.set_xlim(lims[0] - margin, lims[1] + margin)
            ax1.set_ylim(lims[0] - margin, lims[1] + margin)
            
            # 标签
            ax1.set_xlabel('Actual Power (W)')
            ax1.set_ylabel('Predicted Power (W)')
            
            # 标题 - 左上角子图标签
            ax1.text(-0.15, 1.05, subplot_labels[i], transform=ax1.transAxes,
                    fontsize=11, fontweight='bold', va='bottom')
            ax1.set_title(f'{display_name}', fontsize=10, pad=8)
            
            # 添加统计信息框 - 右下角
            stats_text = f'R² = {res["R2"]:.4f}\nRMSE = {res["RMSE"]:.1f} W'
            ax1.text(0.95, 0.05, stats_text, transform=ax1.transAxes,
                    fontsize=8, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='none', alpha=0.8))
            
            # 添加次要刻度
            ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
            
            # ========== 下行: 残差分布 ==========
            ax2 = axes[1, i]
            residuals = true_power - predictions
            
            # 绘制直方图 - 使用柔和颜色
            n_bins = 60
            counts, bins, patches = ax2.hist(residuals, bins=n_bins, 
                                            color=color, alpha=0.7,
                                            edgecolor='white', linewidth=0.3)
            
            # 添加垂直参考线 (x=0)
            ax2.axvline(x=0, color=PUBLICATION_COLORS['red'], 
                       linestyle='--', linewidth=1.2, alpha=0.8)
            
            # 标签
            ax2.set_xlabel('Residual (W)')
            ax2.set_ylabel('Frequency')
            
            # 子图标签
            ax2.text(-0.15, 1.05, subplot_labels[i + n_models], transform=ax2.transAxes,
                    fontsize=11, fontweight='bold', va='bottom')
            
            # 添加统计信息
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            stats_text2 = f'μ = {mean_res:.1f}\nσ = {std_res:.1f}'
            ax2.text(0.95, 0.95, stats_text2, transform=ax2.transAxes,
                    fontsize=8, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='none', alpha=0.8))
            
            # 添加次要刻度
            ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        plt.tight_layout(pad=1.5)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"评估图表已保存到: {save_path}")
    
    def plot_flight_power_curve(self, order_id: str = None, save_path: str = 'result/flight_power_curve.png'):
        """
        绘制单个航次的功率曲线 - 出版级质量
        
        参数:
            order_id: 指定航次的Order ID，如果为None则自动选择第一个
            save_path: 图表保存路径
        """
        # 设置出版级样式
        setup_publication_style()
        
        if self.test_data is None:
            print("[WARNING] 没有测试数据")
            return
        
        # 获取可用的Order ID列表
        if 'Order ID' not in self.test_data.columns:
            print("[WARNING] 测试数据中没有Order ID列")
            return
        
        available_orders = self.test_data['Order ID'].unique()
        print(f"\n可用航次数: {len(available_orders)}")
        
        # 选择航次
        if order_id is None:
            order_id = available_orders[0]
            print(f"自动选择第一个航次: {order_id}")
        elif order_id not in available_orders:
            print(f"[WARNING] 指定的Order ID不存在，使用第一个航次")
            order_id = available_orders[0]
        
        # 提取该航次的数据
        flight_data = self.test_data[self.test_data['Order ID'] == order_id].copy()
        flight_data = flight_data.sort_values('Time Stamp').reset_index(drop=True)
        
        print(f"航次 {order_id} 数据点数: {len(flight_data)}")
        
        # 计算相对时间（秒）
        if 'Time Stamp' in flight_data.columns:
            time_stamps = pd.to_datetime(flight_data['Time Stamp'])
            flight_data['time_seconds'] = (time_stamps - time_stamps.iloc[0]).dt.total_seconds()
        else:
            flight_data['time_seconds'] = range(len(flight_data))
        
        # 获取真实功率
        true_power = flight_data['true_power'].values
        time_axis = flight_data['time_seconds'].values
        
        # 在开头和结尾添加0点
        time_axis_ext = np.concatenate([[time_axis[0] - 5], time_axis, [time_axis[-1] + 5]])
        true_power_ext = np.concatenate([[0], true_power, [0]])
        
        # 计算各模型的预测功率
        model_predictions = {}
        for model_name, model in self.models.items():
            if model_name == 'physical':
                continue
            predictions = []
            for _, row in flight_data.iterrows():
                pred = model.predict_power(
                    height=row['height'],
                    VS=row['VS'],
                    GS=row['GS'],
                    wind_speed=row['wind_speed'],
                    temperature=row['temperature'],
                    humidity=row['humidity'],
                    wind_angle=row['wind_angle'],
                    payload=row['payload']
                )
                predictions.append(pred)
            predictions = np.concatenate([[0], predictions, [0]])
            model_predictions[model_name] = predictions
        
        # ============================================================
        # 出版级图表绘制
        # ============================================================
        
        # 黄金分割比例: 1.618:1，适合双栏排版
        fig_width = 7.0  # 英寸 (双栏宽度约7英寸)
        fig_height = fig_width / 1.618
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # 绘制真实功率 - 主曲线
        ax.plot(time_axis_ext, true_power_ext, 
               color=PUBLICATION_COLORS['black'], 
               linewidth=2.2, 
               label='Actual Power',
               zorder=10)
        
        # 绘制各模型预测
        for model_name, predictions in model_predictions.items():
            color = MODEL_COLORS.get(model_name, PUBLICATION_COLORS['gray'])
            linestyle = MODEL_LINESTYLES.get(model_name, '--')
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            
            ax.plot(time_axis_ext, predictions,
                   color=color,
                   linestyle=linestyle,
                   linewidth=1.8,
                   label=display_name,
                   alpha=0.85,
                   zorder=5)
        
        # 设置坐标轴
        ax.set_xlabel('Flight Time (s)')
        ax.set_ylabel('Power (W)')
        
        # 设置坐标轴范围
        ax.set_xlim(time_axis_ext[0] - 2, time_axis_ext[-1] + 2)
        y_min = 0
        y_max = max(true_power_ext.max(), max([p.max() for p in model_predictions.values()])) * 1.08
        ax.set_ylim(y_min, y_max)
        
        # 添加次要刻度
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        # 图例 - 无边框，放在右上角不遮挡数据
        legend = ax.legend(loc='upper right', frameon=False, 
                          fontsize=9, ncol=1,
                          handlelength=2.5, handletextpad=0.5)
        
        
        plt.tight_layout(pad=0.5)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"航次功率曲线已保存到: {save_path}")
        
        # 打印统计信息
        print(f"\n航次统计:")
        print(f"  飞行时长: {time_axis[-1]:.1f} 秒")
        print(f"  真实功率: {true_power.min():.1f} - {true_power.max():.1f} W, 平均: {true_power.mean():.1f} W")
        print(f"  载荷: {flight_data['payload'].min():.2f} - {flight_data['payload'].max():.2f} kg")
        print(f"  地速: {flight_data['GS'].min():.1f} - {flight_data['GS'].max():.1f} m/s")
        
        # 能耗对比
        print(f"\n能耗对比:")
        print(f"  真实总能耗: {total_energy_true:.2f} Wh")
        for model_name, predictions in model_predictions.items():
            total_energy_pred = np.sum(predictions) / 3600
            error_percent = abs(total_energy_pred - total_energy_true) / total_energy_true * 100
            print(f"  {model_name}预测能耗: {total_energy_pred:.2f} Wh (误差: {error_percent:.2f}%)")
    
    def save_results(self, save_path: str = 'result/power_models_evaluation.csv'):
        """
        保存评估结果到CSV
        """
        results_data = []
        for model_name, res in self.results.items():
            if 'R2' in res:
                results_data.append({
                    '模型': model_name,
                    'RMSE': res['RMSE'],
                    'MAE': res['MAE'],
                    'R2': res['R2']
                })
        
        if results_data:
            df = pd.DataFrame(results_data)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"评估结果已保存到: {save_path}")
    
    def save_predictions(self, save_path: str = 'result/power_predictions_comparison.csv'):
        """
        保存实际功率与预测功率对比数据到CSV
        """
        if self.test_data is None:
            print("[WARNING] 没有测试数据")
            return
        
        # 创建对比数据框
        comparison_df = pd.DataFrame()
        comparison_df['实际功率(W)'] = self.test_data['true_power'].values
        comparison_df['高度(m)'] = self.test_data['height'].values
        comparison_df['竖直速度(m/s)'] = self.test_data['VS'].values
        comparison_df['地速(m/s)'] = self.test_data['GS'].values
        comparison_df['风速(m/s)'] = self.test_data['wind_speed'].values
        comparison_df['温度(C)'] = self.test_data['temperature'].values
        comparison_df['湿度(%)'] = self.test_data['humidity'].values
        comparison_df['风向夹角(度)'] = self.test_data['wind_angle'].values
        comparison_df['载荷(kg)'] = self.test_data['payload'].values
        
        # 添加各模型预测值
        for model_name, res in self.results.items():
            if 'predictions' in res:
                comparison_df[f'{model_name}_预测功率(W)'] = res['predictions']
                comparison_df[f'{model_name}_误差(W)'] = self.test_data['true_power'].values - res['predictions']
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        comparison_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"预测对比数据已保存到: {save_path}")
        print(f"共 {len(comparison_df)} 条记录")
    
    def plot_model_comparison_bar(self, save_path: str = 'result/power_models_bar_comparison.png'):
        """
        绘制模型性能对比柱状图 - 出版级质量
        """
        setup_publication_style()
        
        # 排除物理模型，只保留数据驱动模型
        valid_results = [(name, res) for name, res in self.results.items() 
                        if 'R2' in res and name != 'physical']
        if not valid_results:
            print("[WARNING] 没有有效的评估结果")
            return
        
        # 按R2排序
        valid_results = sorted(valid_results, key=lambda x: x[1]['R2'], reverse=True)
        
        model_names = [MODEL_DISPLAY_NAMES.get(name, name) for name, _ in valid_results]
        rmse_values = [res['RMSE'] for _, res in valid_results]
        r2_values = [res['R2'] for _, res in valid_results]
        colors = [MODEL_COLORS.get(name, PUBLICATION_COLORS['gray']) for name, _ in valid_results]
        
        # 创建双子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))
        
        x = np.arange(len(model_names))
        bar_width = 0.6
        
        # ========== 左图: RMSE ==========
        bars1 = ax1.bar(x, rmse_values, bar_width, color=colors, 
                       edgecolor='white', linewidth=0.8, alpha=0.85)
        ax1.set_ylabel('RMSE (W)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
        ax1.set_ylim(0, max(rmse_values) * 1.2)
        
        # 添加数值标签
        for bar, val in zip(bars1, rmse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.02,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes,
                fontsize=11, fontweight='bold')
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        # ========== 右图: R² ==========
        bars2 = ax2.bar(x, r2_values, bar_width, color=colors,
                       edgecolor='white', linewidth=0.8, alpha=0.85)
        ax2.set_ylabel('R² Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
        
        # R²可能有负值
        min_r2 = min(r2_values)
        if min_r2 < 0:
            ax2.set_ylim(min_r2 * 1.2, 1.05)
            ax2.axhline(y=0, color=PUBLICATION_COLORS['gray'], 
                       linestyle='-', linewidth=0.5, alpha=0.5)
        else:
            ax2.set_ylim(0, 1.05)
        
        # 添加数值标签
        for bar, val in zip(bars2, r2_values):
            offset = max(r2_values) * 0.02 if val >= 0 else -max(r2_values) * 0.08
            va = 'bottom' if val >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    f'{val:.3f}', ha='center', va=va, fontsize=8)
        
        ax2.text(-0.15, 1.05, '(b)', transform=ax2.transAxes,
                fontsize=11, fontweight='bold')
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        plt.tight_layout(pad=1.0)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"模型对比柱状图已保存到: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("瞬时功率预测模型评估程序")
    print("=" * 60)
    
    try:
        evaluator = PowerModelEvaluator()
        
        # 加载测试数据
        evaluator.load_test_data()
        
        # 加载模型
        evaluator.load_models()
        
        # 评估模型
        evaluator.evaluate_models()
        
        # 打印汇总
        evaluator.print_summary()
        
        # 绘制结果 (出版级质量)
        evaluator.plot_results()
        
        # 绘制模型性能对比柱状图 (出版级质量)
        evaluator.plot_model_comparison_bar()
        
        # 保存结果
        evaluator.save_results()
        
        # 保存预测对比数据
        evaluator.save_predictions()
        
        # 绘制单个航次的功率曲线 (出版级质量)
        evaluator.plot_flight_power_curve()
        
        print("\n" + "=" * 60)
        print("评估完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
