#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
显示不同模型预测值与真实值对比的脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

from mtdrp_energy_model import create_energy_model, MTDRPInstance, DroneParameters, Depot, Customer

class PredictionViewer:
    """预测值查看器"""
    
    def __init__(self):
        # 创建一个简单的实例用于模型初始化
        self.instance = MTDRPInstance(
            name="prediction_viewer",
            depot=Depot(),
            customers=[Customer(1, 0, 0, 1.0, 0, 100, 5)],
            drone_params=DroneParameters(),
            num_drones=1
        )
        self.models = {}
        
    def load_test_data(self):
        """加载测试数据"""
        print("正在加载测试数据...")
        
        test_file_path = "Drone_energy_dataset/test_data/flightRecord.xlsx"
        df = pd.read_excel(test_file_path)
        print(f"加载测试数据: {len(df)} 条记录")
        
        # 过滤有效数据
        valid_mask = (
            (df['Distance (m)'] > 0) & 
            (df['Battery Used (kWh)'] > 0) & 
            (df['Payload (kg)'] >= 0)
        )
        df_valid = df[valid_mask].copy()
        print(f"有效测试记录: {len(df_valid)} 条")
        
        # 计算实际飞行时间
        df_valid['Start Time'] = pd.to_datetime(df_valid['Start Time'])
        df_valid['End Time'] = pd.to_datetime(df_valid['End Time'])
        df_valid['Flight Time (s)'] = (df_valid['End Time'] - df_valid['Start Time']).dt.total_seconds()
        
        # 转换为时间戳（秒）
        start_times = (df_valid['Start Time'] - pd.Timestamp('1970-01-01')).dt.total_seconds().values
        end_times = (df_valid['End Time'] - pd.Timestamp('1970-01-01')).dt.total_seconds().values
        
        # 准备特征和目标
        distances = df_valid['Distance (m)'].values
        payloads = df_valid['Payload (kg)'].values
        flight_times = df_valid['Flight Time (s)'].values
        targets = df_valid['Battery Used (kWh)'].values
        
        # 处理环境特征
        wind_speeds = df_valid.get('Avg Wind Speed', pd.Series([5.0] * len(df_valid))).fillna(5.0).values
        wind_angles = np.full(len(df_valid), 90.0)  # 默认90度
        temperatures = df_valid.get('Avg Temperature', pd.Series([25.0] * len(df_valid))).fillna(25.0).values
        humidities = df_valid.get('Avg Humidity', pd.Series([60.0] * len(df_valid))).fillna(60.0).values
        
        features = np.column_stack([
            distances, payloads, wind_speeds, wind_angles, temperatures, humidities
        ])
        
        return features, targets, flight_times, start_times, end_times, df_valid
    
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
    
    def predict_all_models(self, features, flight_times, start_times, end_times):
        """使用所有模型进行预测"""
        predictions = {}
        
        for model_name, model in self.models.items():
            if model is None:
                print(f"[SKIP] {model_name} 模型未加载，跳过预测")
                continue
            
            print(f"正在使用 {model_name} 模型进行预测...")
            model_predictions = []
            
            for i, feature in enumerate(features):
                distance, payload = feature[0], feature[1]
                
                try:
                    if model_name == "physical":
                        # 物理模型使用新接口
                        energy = model.energy_consumption(payload, end_times[i], start_times[i])
                    else:
                        # 其他模型使用标准接口
                        energy = model.energy_consumption(payload, distance)
                    model_predictions.append(energy)
                except Exception as e:
                    print(f"[WARNING] {model_name} 模型预测失败: {e}")
                    model_predictions.append(0.0)
            
            predictions[model_name] = np.array(model_predictions)
        
        return predictions
    
    def show_predictions(self):
        """显示预测结果对比"""
        # 加载数据和模型
        features, targets, flight_times, start_times, end_times, df_valid = self.load_test_data()
        self.load_models()
        
        # 获取所有模型的预测
        predictions = self.predict_all_models(features, flight_times, start_times, end_times)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame()
        result_df['样本ID'] = range(1, len(targets) + 1)
        result_df['距离(m)'] = features[:, 0]
        result_df['载荷(kg)'] = features[:, 1]
        result_df['飞行时间(s)'] = flight_times
        result_df['真实能耗(kWh)'] = targets
        
        # 添加各模型预测值
        for model_name, pred_values in predictions.items():
            result_df[f'{model_name}_预测(kWh)'] = pred_values
            result_df[f'{model_name}_误差'] = pred_values - targets
            result_df[f'{model_name}_相对误差(%)'] = ((pred_values - targets) / targets) * 100
        
        # 显示前20条记录
        print("\n" + "="*120)
        print("前20条样本的预测结果对比")
        print("="*120)
        
        # 选择要显示的列
        display_columns = ['样本ID', '距离(m)', '载荷(kg)', '飞行时间(s)', '真实能耗(kWh)']
        for model_name in predictions.keys():
            display_columns.extend([f'{model_name}_预测(kWh)', f'{model_name}_误差'])
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.6f}'.format)
        
        print(result_df[display_columns].head(20))
        
        # 保存完整结果到文件
        output_file = "result/model_predictions_comparison.csv"
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n完整预测结果已保存到: {output_file}")
        
        # 显示统计信息
        print("\n" + "="*80)
        print("各模型预测统计信息")
        print("="*80)
        
        stats_df = pd.DataFrame()
        stats_df['模型'] = ['真实值'] + list(predictions.keys())
        
        # 计算统计指标
        mean_values = [targets.mean()]
        std_values = [targets.std()]
        min_values = [targets.min()]
        max_values = [targets.max()]
        
        for model_name, pred_values in predictions.items():
            mean_values.append(pred_values.mean())
            std_values.append(pred_values.std())
            min_values.append(pred_values.min())
            max_values.append(pred_values.max())
        
        stats_df['平均值(kWh)'] = mean_values
        stats_df['标准差(kWh)'] = std_values
        stats_df['最小值(kWh)'] = min_values
        stats_df['最大值(kWh)'] = max_values
        
        print(stats_df.to_string(index=False, float_format='%.6f'))
        
        # 显示误差统计
        print("\n" + "="*80)
        print("各模型误差统计")
        print("="*80)
        
        error_stats = []
        for model_name, pred_values in predictions.items():
            errors = pred_values - targets
            abs_errors = np.abs(errors)
            rel_errors = np.abs(errors / targets) * 100
            
            error_stats.append({
                '模型': model_name,
                '平均误差(kWh)': errors.mean(),
                '平均绝对误差(kWh)': abs_errors.mean(),
                '平均相对误差(%)': rel_errors.mean(),
                '最大绝对误差(kWh)': abs_errors.max(),
                '最大相对误差(%)': rel_errors.max()
            })
        
        error_df = pd.DataFrame(error_stats)
        print(error_df.to_string(index=False, float_format='%.6f'))
        
        return result_df

def main():
    """主函数"""
    viewer = PredictionViewer()
    result_df = viewer.show_predictions()
    
    print(f"\n预测对比完成！")
    print(f"详细结果已保存到: result/model_predictions_comparison.csv")

if __name__ == "__main__":
    main()
