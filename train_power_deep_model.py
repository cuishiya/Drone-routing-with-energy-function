#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于flightTrajectory数据训练瞬时功率预测的PyTorch深度学习模型

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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# PyTorch相关导入
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    print("[OK] PyTorch可用")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("[ERROR] PyTorch未安装，无法训练深度学习模型")


class PowerNet(nn.Module):
    """
    基于PyTorch的瞬时功率预测神经网络
    简化架构，适合大数据量训练
    """
    def __init__(self, input_size=8, hidden_sizes=[64, 32], dropout_rate=0.1):
        super(PowerNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 隐藏层
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if i == 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.ReLU())  # 确保功率非负
        
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


class PowerDeepModelTrainer:
    """PyTorch深度学习瞬时功率模型训练器"""
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.training_data = None
        self.training_history = {'train_loss': [], 'val_loss': []}
        self.feature_names = ['height', 'VS', 'GS', 'wind_speed', 'temperature', 'humidity', 'wind_angle', 'payload']
        
        print(f"使用设备: {self.device}")
    
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
    
    def create_model(self, input_size=7):
        """创建深度学习模型"""
        self.model = PowerNet(
            input_size=input_size,
            hidden_sizes=[64, 32],
            dropout_rate=0.1
        ).to(self.device)
        
        print(f"\n模型结构:")
        print(self.model)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"总参数数量: {total_params:,}")
        
        return self.model
    
    def train_model(self, X, y, test_size=0.2, val_size=0.1, random_state=42,
                   batch_size=256, epochs=100, learning_rate=0.005, patience=15):
        """训练深度学习模型"""
        print("\n开始训练PyTorch瞬时功率预测模型...")
        
        # 分割数据集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"测试集大小: {len(X_test)}")
        
        # 特征标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 创建模型
        self.create_model(input_size=X.shape[1])
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\n开始训练，共 {epochs} 个epoch...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
            
            if patience_counter >= patience:
                print(f"早停触发，在第 {epoch+1} 个epoch停止训练")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 最终评估
        self.model.eval()
        with torch.no_grad():
            y_train_pred = self.model(X_train_tensor).cpu().numpy().flatten()
            y_val_pred = self.model(X_val_tensor).cpu().numpy().flatten()
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            y_test_pred = self.model(X_test_tensor).cpu().numpy().flatten()
        
        # 计算评估指标
        print(f"\n模型训练结果:")
        print(f"  训练集 RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f} W, R2: {r2_score(y_train, y_train_pred):.6f}")
        print(f"  验证集 RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred)):.4f} W, R2: {r2_score(y_val, y_val_pred):.6f}")
        print(f"  测试集 RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f} W, R2: {r2_score(y_test, y_test_pred):.6f}")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, 
                y_train_pred, y_val_pred, y_test_pred)
    
    def save_model(self, model_path='result/power_pytorch_model.pth'):
        """保存训练好的模型"""
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save(self.model.state_dict(), model_path)
        print(f"\n模型已保存到: {model_path}")
        
        scaler_path = model_path.replace('.pth', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"标准化器已保存到: {scaler_path}")
    
    def plot_results(self, y_test, y_test_pred):
        """绘制训练结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 训练历史
        ax1 = axes[0]
        ax1.plot(self.training_history['train_loss'], label='训练损失', alpha=0.8)
        ax1.plot(self.training_history['val_loss'], label='验证损失', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练历史')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 预测vs真实值散点图
        ax2 = axes[1]
        ax2.scatter(y_test, y_test_pred, alpha=0.3, s=1)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax2.set_xlabel('真实功率 (W)')
        ax2.set_ylabel('预测功率 (W)')
        ax2.set_title('深度学习瞬时功率预测效果')
        ax2.grid(True, alpha=0.3)
        
        # 残差分布
        ax3 = axes[2]
        residuals = y_test - y_test_pred
        ax3.hist(residuals, bins=100, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('残差 (W)')
        ax3.set_ylabel('频次')
        ax3.set_title('残差分布')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('result/power_deep_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("结果图已保存到: result/power_deep_model_results.png")


def main():
    """主函数"""
    if not PYTORCH_AVAILABLE:
        print("PyTorch未安装，无法训练深度学习模型")
        return
    
    print("=" * 60)
    print("PyTorch瞬时功率预测模型训练程序")
    print("=" * 60)
    
    try:
        trainer = PowerDeepModelTrainer()
        trainer.load_trajectory_data()
        X, y = trainer.prepare_features()
        
        results = trainer.train_model(
            X, y,
            batch_size=256,
            epochs=100,
            learning_rate=0.005,
            patience=15
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test, y_train_pred, y_val_pred, y_test_pred = results
        
        trainer.save_model()
        trainer.plot_results(y_test, y_test_pred)
        
        print("\n" + "=" * 60)
        print("瞬时功率深度学习模型训练完成！")
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
