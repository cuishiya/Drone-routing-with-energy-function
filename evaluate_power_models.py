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
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        绘制评估结果图
        """
        print("\n生成评估图表...")
        
        valid_results = [(name, res) for name, res in self.results.items() if 'predictions' in res]
        
        if not valid_results:
            print("[WARNING] 没有有效的评估结果可绘制")
            return
        
        n_models = len(valid_results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        true_power = self.test_data['true_power'].values
        
        for i, (model_name, res) in enumerate(valid_results):
            predictions = res['predictions']
            
            # 散点图
            ax1 = axes[0, i]
            ax1.scatter(true_power, predictions, alpha=0.3, s=1)
            ax1.plot([true_power.min(), true_power.max()], 
                    [true_power.min(), true_power.max()], 'r--', lw=2)
            ax1.set_xlabel('真实功率 (W)')
            ax1.set_ylabel('预测功率 (W)')
            ax1.set_title(f'{model_name}\nR2 = {res["R2"]:.4f}')
            ax1.grid(True, alpha=0.3)
            
            # 残差分布
            ax2 = axes[1, i]
            residuals = true_power - predictions
            ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('残差 (W)')
            ax2.set_ylabel('频次')
            ax2.set_title(f'{model_name} 残差分布')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"评估图表已保存到: {save_path}")
    
    def plot_flight_power_curve(self, order_id: str = None, save_path: str = 'result/flight_power_curve.png'):
        """
        绘制单个航次的功率曲线（横轴时间，纵轴功率）
        
        参数:
            order_id: 指定航次的Order ID，如果为None则自动选择第一个
            save_path: 图表保存路径
        """
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
            # 转换时间戳为相对秒数
            time_stamps = pd.to_datetime(flight_data['Time Stamp'])
            flight_data['time_seconds'] = (time_stamps - time_stamps.iloc[0]).dt.total_seconds()
        else:
            # 如果没有时间戳，使用索引作为时间（假设每秒一条记录）
            flight_data['time_seconds'] = range(len(flight_data))
        
        # 获取真实功率
        true_power = flight_data['true_power'].values
        time_axis = flight_data['time_seconds'].values
        
        # 在开头和结尾添加0点（表示起飞前和降落后功率为0）
        time_axis = np.concatenate([[time_axis[0] - 5], time_axis, [time_axis[-1] + 5]])
        true_power = np.concatenate([[0], true_power, [0]])
        
        # 计算各模型的预测功率（排除physical模型）
        model_predictions = {}
        for model_name, model in self.models.items():
            if model_name == 'physical':
                continue  # 跳过物理模型
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
            # 在开头和结尾添加0点
            predictions = np.concatenate([[0], predictions, [0]])
            model_predictions[model_name] = predictions
        
        # 绘制功率曲线（单图）
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        # 功率随时间变化
        ax1.plot(time_axis, true_power, 'k-', linewidth=2, label='真实功率', alpha=0.8)
        
        colors = {'tree': 'green', 'deep': 'blue', 'linear': 'orange'}
        for model_name, predictions in model_predictions.items():
            color = colors.get(model_name, 'gray')
            ax1.plot(time_axis, predictions, '--', linewidth=1.5, label=f'{model_name}预测', 
                    color=color, alpha=0.7)
        
        ax1.set_xlabel('飞行时间 (秒)', fontsize=12)
        ax1.set_ylabel('功率 (W)', fontsize=12)
        ax1.set_title(f'航次功率曲线 - {order_id}', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"航次功率曲线已保存到: {save_path}")
        
        # 打印该航次的统计信息
        print(f"\n航次统计:")
        print(f"  飞行时长: {time_axis[-1]:.1f} 秒")
        print(f"  真实功率: {true_power.min():.1f} - {true_power.max():.1f} W, 平均: {true_power.mean():.1f} W")
        print(f"  载荷: {flight_data['payload'].min():.2f} - {flight_data['payload'].max():.2f} kg")
        print(f"  地速: {flight_data['GS'].min():.1f} - {flight_data['GS'].max():.1f} m/s")
        
        # 计算总能耗
        total_energy_true = np.sum(true_power) / 3600 / 1000  # kWh
        print(f"\n能耗对比:")
        print(f"  真实总能耗: {total_energy_true*1000:.2f} Wh")
        for model_name, predictions in model_predictions.items():
            total_energy_pred = np.sum(predictions) / 3600 / 1000
            error_percent = abs(total_energy_pred - total_energy_true) / total_energy_true * 100
            print(f"  {model_name}预测能耗: {total_energy_pred*1000:.2f} Wh (误差: {error_percent:.2f}%)")
    
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
        
        # 绘制结果
        evaluator.plot_results()
        
        # 保存结果
        evaluator.save_results()
        
        # 保存预测对比数据
        evaluator.save_predictions()
        
        # 绘制单个航次的功率曲线
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
