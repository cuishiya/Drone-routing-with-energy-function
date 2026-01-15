#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch深度学习模型训练脚本

使用真实飞行数据训练深度神经网络模型，预测无人机能耗
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

class EnergyNet(nn.Module):
    """
    基于PyTorch的能耗预测神经网络
    """
    def __init__(self, input_size=6, hidden_sizes=[32, 16], dropout_rate=0.1):
        super(EnergyNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 隐藏层 - 简化为2层
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            # 只在第一层使用dropout，减少正则化强度
            if i == 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.ReLU())  # 确保输出非负
        
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

class DeepModelTrainer:
    """PyTorch深度学习模型训练器"""
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.training_data = None
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        print(f"使用设备: {self.device}")
        
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
        
        # 特征: 距离、载荷、风速、风向、温度、湿度
        distances = self.training_data['Distance (m)'].values
        payloads = self.training_data['Payload (kg)'].values
        
        # 处理环境特征，如果不存在则使用默认值
        wind_speeds = self.training_data.get('Avg Wind Speed', pd.Series([5.0] * len(self.training_data))).fillna(5.0).values
        
        # 风向处理
        if 'Wind Direction' in self.training_data.columns:
            wind_directions = self.training_data['Wind Direction'].fillna('N').values
            wind_angles = []
            direction_map = {
                'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
                'S': 180, 'SW': 225, 'W': 270, 'NW': 315
            }
            for direction in wind_directions:
                wind_angles.append(direction_map.get(str(direction).upper(), 90))
            wind_angles = np.array(wind_angles, dtype=float)
        else:
            wind_angles = np.full(len(self.training_data), 90.0)
        
        temperatures = self.training_data.get('Avg Temperature', pd.Series([25.0] * len(self.training_data))).fillna(25.0).values
        humidities = self.training_data.get('Avg Humidity', pd.Series([60.0] * len(self.training_data))).fillna(60.0).values
        
        X = np.column_stack([
            distances, payloads, wind_speeds, wind_angles, temperatures, humidities
        ])
        
        # 目标: 电池消耗
        y = self.training_data['Battery Used (kWh)'].values
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")
        print(f"特征统计:")
        feature_names = ['距离(m)', '载荷(kg)', '风速(m/s)', '风向(°)', '温度(°C)', '湿度(%)']
        for i, name in enumerate(feature_names):
            print(f"  {name}: {X[:, i].min():.2f} - {X[:, i].max():.2f}")
        print(f"目标变量统计:")
        print(f"  能耗范围: {y.min():.6f} - {y.max():.6f} kWh")
        print(f"  平均能耗: {y.mean():.6f} kWh")
        
        return X, y
    
    def create_model(self, input_size=6):
        """创建深度学习模型"""
        self.model = EnergyNet(
            input_size=input_size,
            hidden_sizes=[32, 16],  # 简化为2层隐藏层
            dropout_rate=0.1        # 降低dropout率
        ).to(self.device)
        
        print(f"模型结构:")
        print(self.model)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        return self.model
    
    def train_model(self, X, y, test_size=0.2, val_size=0.1, random_state=42,
                   batch_size=32, epochs=200, learning_rate=0.001, patience=20):
        """训练深度学习模型"""
        print("\n开始训练PyTorch深度学习模型...")
        
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
        
        print(f"标准化参数:")
        print(f"  均值: {self.scaler.mean_}")
        print(f"  标准差: {self.scaler.scale_}")
        
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
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 增加正则化
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)  # 更保守的学习率调度
        
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
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 20 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.2e}")
            
            # 早停
            if patience_counter >= patience:
                print(f"早停触发，在第 {epoch+1} 个epoch停止训练")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"恢复最佳模型，验证损失: {best_val_loss:.6f}")
        
        # 最终评估
        self.model.eval()
        with torch.no_grad():
            # 训练集预测
            y_train_pred = self.model(X_train_tensor).cpu().numpy().flatten()
            # 验证集预测
            y_val_pred = self.model(X_val_tensor).cpu().numpy().flatten()
            # 测试集预测
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            y_test_pred = self.model(X_test_tensor).cpu().numpy().flatten()
        
        # 计算评估指标
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"\n模型训练结果:")
        print(f"  训练集 RMSE: {train_rmse:.6f}, R2: {train_r2:.6f}")
        print(f"  验证集 RMSE: {val_rmse:.6f}, R2: {val_r2:.6f}")
        print(f"  测试集 RMSE: {test_rmse:.6f}, R2: {test_r2:.6f}")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, 
                y_train_pred, y_val_pred, y_test_pred)
    
    def save_model(self, model_path='result/pytorch_energy_model.pth'):
        """保存训练好的模型"""
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型状态字典
        torch.save(self.model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")
        
        # 保存标准化器
        scaler_path = model_path.replace('.pth', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"标准化器已保存到: {scaler_path}")
        
        # 保存训练历史
        history_path = model_path.replace('.pth', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        print(f"训练历史已保存到: {history_path}")
    
    def plot_results(self, y_test, y_test_pred):
        """绘制训练结果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 训练历史
        ax1.plot(self.training_history['train_loss'], label='训练损失', alpha=0.8)
        ax1.plot(self.training_history['val_loss'], label='验证损失', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练历史')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 预测vs真实值散点图
        ax2.scatter(y_test, y_test_pred, alpha=0.6)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax2.set_xlabel('真实能耗 (kWh)')
        ax2.set_ylabel('预测能耗 (kWh)')
        ax2.set_title('深度学习模型预测效果')
        ax2.grid(True, alpha=0.3)
        
        # 残差图
        residuals = y_test - y_test_pred
        ax3.scatter(y_test_pred, residuals, alpha=0.6)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('预测能耗 (kWh)')
        ax3.set_ylabel('残差 (kWh)')
        ax3.set_title('残差分析')
        ax3.grid(True, alpha=0.3)
        
        # 残差分布
        ax4.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('残差 (kWh)')
        ax4.set_ylabel('频次')
        ax4.set_title('残差分布')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('result/deep_learning_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("训练结果图已保存到: result/deep_learning_training_results.png")

def main():
    """主函数"""
    if not PYTORCH_AVAILABLE:
        print("PyTorch未安装，无法训练深度学习模型")
        return
    
    print("PyTorch深度学习模型训练程序")
    print("=" * 50)
    
    try:
        # 创建训练器
        trainer = DeepModelTrainer()
        
        # 加载数据
        trainer.load_training_data()
        
        # 准备特征
        X, y = trainer.prepare_features()
        
        # 训练模型
        results = trainer.train_model(
            X, y,
            batch_size=16,      # 减小批次大小
            epochs=300,         # 增加训练轮数
            learning_rate=0.005, # 提高学习率
            patience=30         # 增加早停耐心
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test, y_train_pred, y_val_pred, y_test_pred = results
        
        # 保存模型
        trainer.save_model()
        
        # 绘制结果
        trainer.plot_results(y_test, y_test_pred)
        
        print("\n" + "=" * 50)
        print("PyTorch深度学习模型训练完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
