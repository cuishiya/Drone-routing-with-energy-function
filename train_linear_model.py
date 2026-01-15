#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
线性回归模型训练脚本

使用真实飞行数据训练线性回归模型，获得最优参数
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

class LinearModelTrainer:
    """线性回归模型训练器"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.training_data = None
        
    def load_training_data(self):
        """加载训练数据"""
        print("正在加载训练数据...")
        
        # 从多个数据源加载数据
        data_sources = [
            "Drone_energy_dataset/UAS04028624/flightRecord.xlsx",
            "Drone_energy_dataset/UAS04028648/flightRecord.xlsx", 
            "Drone_energy_dataset/UAS04143500/flightRecord.xlsx"
        ]
        
        all_data = []
        
        for source in data_sources:
            if os.path.exists(source):
                try:
                    df = pd.read_excel(source)
                    print(f"加载 {source}: {len(df)} 条记录")
                    
                    # 检查必要列
                    required_cols = ['Distance (m)', 'Battery Used (kWh)', 'Payload (kg)']
                    if all(col in df.columns for col in required_cols):
                        # 过滤有效数据
                        valid_mask = (
                            (df['Distance (m)'] > 0) & 
                            (df['Battery Used (kWh)'] > 0) & 
                            (df['Payload (kg)'] >= 0)
                        )
                        df_valid = df[valid_mask]
                        all_data.append(df_valid)
                        print(f"  有效记录: {len(df_valid)} 条")
                    else:
                        print(f"  [WARNING] 缺少必要列")
                        
                except Exception as e:
                    print(f"  [ERROR] 读取失败: {e}")
            else:
                print(f"  [WARNING] 文件不存在: {source}")
        
        if not all_data:
            raise ValueError("没有加载到任何训练数据")
        
        # 合并数据
        self.training_data = pd.concat(all_data, ignore_index=True)
        print(f"总训练数据: {len(self.training_data)} 条记录")
        
        return self.training_data
    
    def prepare_features(self):
        """准备特征和目标变量"""
        if self.training_data is None:
            raise ValueError("请先加载训练数据")
        
        # 特征: 距离和载荷
        X = self.training_data[['Distance (m)', 'Payload (kg)']].values
        
        # 目标: 电池消耗
        y = self.training_data['Battery Used (kWh)'].values
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")
        print(f"特征统计:")
        print(f"  距离范围: {X[:, 0].min():.0f} - {X[:, 0].max():.0f} m")
        print(f"  载荷范围: {X[:, 1].min():.1f} - {X[:, 1].max():.1f} kg")
        print(f"目标变量统计:")
        print(f"  能耗范围: {y.min():.6f} - {y.max():.6f} kWh")
        print(f"  平均能耗: {y.mean():.6f} kWh")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """训练线性回归模型"""
        print("\n开始训练线性回归模型...")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        # 特征标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"标准化参数:")
        print(f"  均值: {self.scaler.mean_}")
        print(f"  标准差: {self.scaler.scale_}")
        
        # 训练线性回归模型
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
        
        # 预测
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # 评估
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"\n模型训练结果:")
        print(f"  训练集 RMSE: {train_rmse:.6f}")
        print(f"  测试集 RMSE: {test_rmse:.6f}")
        print(f"  训练集 R2: {train_r2:.6f}")
        print(f"  测试集 R2: {test_r2:.6f}")
        
        # 输出模型参数
        print(f"\n学习到的模型参数:")
        print(f"  距离系数: {self.model.coef_[0]:.8f}")
        print(f"  载荷系数: {self.model.coef_[1]:.8f}")
        print(f"  截距: {self.model.intercept_:.8f}")
        
        # 计算原始特征空间的系数（未标准化）
        coef_distance_raw = self.model.coef_[0] / self.scaler.scale_[0]
        coef_payload_raw = self.model.coef_[1] / self.scaler.scale_[1]
        intercept_raw = (self.model.intercept_ - 
                        np.dot(self.model.coef_, self.scaler.mean_ / self.scaler.scale_))
        
        print(f"\n原始特征空间的参数:")
        print(f"  距离系数: {coef_distance_raw:.8f} kWh/m")
        print(f"  载荷系数: {coef_payload_raw:.8f} kWh/kg")
        print(f"  截距: {intercept_raw:.8f} kWh")
        
        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred
    
    def save_model(self, model_path='result/linear_regression_model.pkl'):
        """保存训练好的模型"""
        if self.model is None or self.scaler is None:
            raise ValueError("模型尚未训练，无法保存")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型和标准化器
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'training_info': {
                'n_samples': len(self.training_data),
                'feature_names': ['Distance (m)', 'Payload (kg)'],
                'target_name': 'Battery Used (kWh)',
                'coef_distance': self.model.coef_[0],
                'coef_payload': self.model.coef_[1],
                'intercept': self.model.intercept_
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n模型已保存到: {model_path}")
    
    def plot_results(self, y_test, y_test_pred):
        """绘制预测结果"""
        plt.figure(figsize=(12, 5))
        
        # 预测vs真实值散点图
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_test_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('真实能耗 (kWh)')
        plt.ylabel('预测能耗 (kWh)')
        plt.title('线性回归模型预测效果')
        plt.grid(True, alpha=0.3)
        
        # 残差图
        plt.subplot(1, 2, 2)
        residuals = y_test - y_test_pred
        plt.scatter(y_test_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测能耗 (kWh)')
        plt.ylabel('残差 (kWh)')
        plt.title('残差分析')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('result/linear_regression_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("训练结果图已保存到: result/linear_regression_training_results.png")

def main():
    """主函数"""
    print("线性回归模型训练程序")
    print("=" * 50)
    
    try:
        # 创建训练器
        trainer = LinearModelTrainer()
        
        # 加载数据
        trainer.load_training_data()
        
        # 准备特征
        X, y = trainer.prepare_features()
        
        # 训练模型
        X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = trainer.train_model(X, y)
        
        # 保存模型
        trainer.save_model()
        
        # 绘制结果
        trainer.plot_results(y_test, y_test_pred)
        
        print("\n" + "=" * 50)
        print("线性回归模型训练完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"训练过程中出错: {e}")

if __name__ == "__main__":
    main()
