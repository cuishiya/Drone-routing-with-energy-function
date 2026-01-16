#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于flightTrajectory数据训练瞬时功率预测的线性回归模型

输入特征:
- height: 高度 [m]
- VS: 竖直速度 [m/s]
- GS: 地速 [m/s]
- windSpeed: 风速 [m/s]
- temperature: 温度 [°C]
- humidity: 湿度 [%]
- wind_angle: 风向夹角 [度] (由windDirect和course计算)

输出:
- 瞬时功率 [W] (由 Voltage × Current 计算)
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PowerLinearModelTrainer:
    """瞬时功率线性回归模型训练器"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.training_data = None
        self.feature_names = ['height', 'VS', 'GS', 'wind_speed', 'temperature', 'humidity', 'wind_angle', 'payload']
        
    def load_trajectory_data(self):
        """
        从flightTrajectory文件加载训练数据，并通过Order ID关联flightRecord获取载荷
        """
        print("正在加载flightTrajectory训练数据...")
        
        # 数据源路径 (trajectory和record成对)
        data_sources = [
            ("Drone_energy_dataset/UAS04028624/flightTrajectory.xlsx", 
             "Drone_energy_dataset/UAS04028624/flightRecord.xlsx"),
            ("Drone_energy_dataset/UAS04028648/flightTrajectory.xlsx",
             "Drone_energy_dataset/UAS04028648/flightRecord.xlsx")
        ]
        
        all_data = []
        
        for traj_source, record_source in data_sources:
            if os.path.exists(traj_source) and os.path.exists(record_source):
                try:
                    print(f"加载 {traj_source}...")
                    df_traj = pd.read_excel(traj_source)
                    print(f"  轨迹记录: {len(df_traj)} 条")
                    
                    # 加载flightRecord获取载荷数据
                    print(f"加载 {record_source}...")
                    df_record = pd.read_excel(record_source)
                    print(f"  航次记录: {len(df_record)} 条")
                    
                    # 创建Order ID到Payload的映射
                    payload_map = df_record.set_index('Order ID')['Payload (kg)'].to_dict()
                    
                    required_cols = ['Order ID', 'Height', 'VS (m/s)', 'GS (m/s)', 'Course', 
                                   'Wind Speed', 'Wind Direct', 'Temperature', 'Humidity',
                                   'Voltage', 'Current']
                    
                    if all(col in df_traj.columns for col in required_cols):
                        # 通过Order ID关联载荷数据
                        df_traj['payload'] = df_traj['Order ID'].map(payload_map)
                        
                        # 过滤有效数据
                        valid_mask = (
                            (df_traj['Height'] > 0) &
                            (df_traj['GS (m/s)'] >= 0) &
                            (df_traj['Voltage'] > 0) &
                            (df_traj['Current'] > 0) &
                            (df_traj['Wind Speed'] >= 0) &
                            (df_traj['Temperature'] > -50) &
                            (df_traj['Humidity'] >= 0) & (df_traj['Humidity'] <= 100) &
                            (df_traj['payload'].notna())
                        )
                        df_valid = df_traj[valid_mask].copy()
                        
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
                        
                        # 载荷已经是kg单位
                        # df_valid['payload'] 已在前面通过map获取
                        
                        # 计算瞬时功率 [W]
                        df_valid['power'] = (df_valid['Voltage'] / 1000.0) * (df_valid['Current'] / 1000.0)
                        
                        all_data.append(df_valid)
                        print(f"  有效记录: {len(df_valid)} 条")
                        print(f"  载荷范围: {df_valid['payload'].min():.2f} - {df_valid['payload'].max():.2f} kg")
                    else:
                        missing = [col for col in required_cols if col not in df_traj.columns]
                        print(f"  [WARNING] 缺少列: {missing}")
                        
                except Exception as e:
                    print(f"  [ERROR] 读取失败: {e}")
            else:
                print(f"  [WARNING] 文件不存在: {traj_source} 或 {record_source}")
        
        if not all_data:
            raise ValueError("没有加载到任何训练数据")
        
        self.training_data = pd.concat(all_data, ignore_index=True)
        print(f"\n总训练数据: {len(self.training_data)} 条记录")
        
        return self.training_data
    
    def prepare_features(self):
        """准备特征和目标变量"""
        if self.training_data is None:
            raise ValueError("请先加载训练数据")
        
        print("\n准备特征矩阵...")
        
        feature_cols = ['height', 'VS', 'GS', 'wind_speed', 'temperature', 'humidity', 'wind_angle', 'payload']
        
        X = self.training_data[feature_cols].values
        y = self.training_data['power'].values
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")
        print(f"\n特征统计:")
        for i, name in enumerate(self.feature_names):
            print(f"  {name}: {X[:, i].min():.2f} - {X[:, i].max():.2f}")
        print(f"\n目标变量统计:")
        print(f"  功率范围: {y.min():.2f} - {y.max():.2f} W")
        print(f"  平均功率: {y.mean():.2f} W")
        
        return X, y
    
    def build_model(self, X, y, test_size=0.2, random_state=42, use_polynomial=True, degree=2):
        """
        构建和训练线性回归模型
        """
        print("\n构建线性回归瞬时功率预测模型...")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        # 创建模型管道
        if use_polynomial:
            print(f"使用多项式特征 (degree={degree})")
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('regressor', Ridge(alpha=1.0))  # 使用Ridge回归防止过拟合
            ])
        else:
            print("使用简单线性回归")
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
        
        # 训练模型
        print("开始训练模型...")
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # 评估模型
        self._evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
        
        return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
    
    def _evaluate_model(self, y_train, y_pred_train, y_test, y_pred_test):
        """
        评估模型性能
        """
        print("\n" + "=" * 50)
        print("模型评估结果")
        print("=" * 50)
        
        # 训练集指标
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        print(f"训练集:")
        print(f"  RMSE: {train_rmse:.4f} W")
        print(f"  MAE:  {train_mae:.4f} W")
        print(f"  R2:   {train_r2:.6f}")
        
        # 测试集指标
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n测试集:")
        print(f"  RMSE: {test_rmse:.4f} W")
        print(f"  MAE:  {test_mae:.4f} W")
        print(f"  R2:   {test_r2:.6f}")
        
        # 显示模型截距和系数
        regressor = self.model.named_steps['regressor']
        if hasattr(regressor, 'coef_') and hasattr(regressor, 'intercept_'):
            print(f"\n" + "=" * 50)
            print("模型参数")
            print("=" * 50)
            print(f"截距 (intercept): {regressor.intercept_:.6f}")
            print(f"\n系数 (coefficients):")
            
            # 获取多项式特征名称
            if 'poly' in self.model.named_steps:
                poly = self.model.named_steps['poly']
                feature_names = poly.get_feature_names_out(self.feature_names)
            else:
                feature_names = self.feature_names
            
            print(f"  共 {len(regressor.coef_)} 个系数")
            print(f"\n  {'特征名称':<30} {'系数值':<15}")
            print(f"  {'-'*45}")
            
            # 按系数绝对值排序，显示最重要的特征
            coef_importance = sorted(zip(feature_names, regressor.coef_), 
                                    key=lambda x: abs(x[1]), reverse=True)
            
            # 显示前20个最重要的系数
            for name, coef in coef_importance[:20]:
                print(f"  {name:<30} {coef:>15.6f}")
            
            if len(coef_importance) > 20:
                print(f"  ... (还有 {len(coef_importance) - 20} 个系数未显示)")
    
    def save_model(self, model_path='result/power_linear_model.pkl'):
        """
        保存训练好的模型
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\n模型已保存到: {model_path}")
    
    def plot_results(self, y_test, y_pred_test):
        """
        绘制训练结果
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 预测vs真实值散点图
        ax1 = axes[0]
        ax1.scatter(y_test, y_pred_test, alpha=0.3, s=1)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('真实功率 (W)')
        ax1.set_ylabel('预测功率 (W)')
        ax1.set_title('线性回归瞬时功率预测效果')
        ax1.grid(True, alpha=0.3)
        
        # 残差分布
        ax2 = axes[1]
        residuals = y_test - y_pred_test
        ax2.hist(residuals, bins=100, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('残差 (W)')
        ax2.set_ylabel('频次')
        ax2.set_title('残差分布')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('result/power_linear_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("结果图已保存到: result/power_linear_model_results.png")


def main():
    """主函数"""
    print("=" * 60)
    print("线性回归瞬时功率预测模型训练程序")
    print("=" * 60)
    
    try:
        trainer = PowerLinearModelTrainer()
        trainer.load_trajectory_data()
        X, y = trainer.prepare_features()
        
        # 训练模型 (使用多项式特征)
        results = trainer.build_model(X, y, use_polynomial=True, degree=2)
        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = results
        
        trainer.save_model()
        trainer.plot_results(y_test, y_pred_test)
        
        print("\n" + "=" * 60)
        print("瞬时功率线性回归模型训练完成！")
        print("=" * 60)
        print("\n模型使用说明:")
        print("输入特征: height(m), VS(m/s), GS(m/s), wind_speed(m/s), temperature(°C), humidity(%), wind_angle(°)")
        print("输出: 瞬时功率 (W)")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
