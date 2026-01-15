import pandas as pd
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
# import seaborn as sns  # 暂时注释掉，后续需要时再安装

class DroneEnergyModel:
    def __init__(self):
        """
        初始化无人机能耗模型
        """
        self.model = None
        self.feature_names = ['distance', 'payload', 'wind_speed', 'wind_angle', 'temperature', 'humidity']
        self.data = None
        
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        计算两点间的方位角（飞行方向）
        参数:
            lat1, lon1: 起点纬度、经度
            lat2, lon2: 终点纬度、经度
        返回:
            方位角（度数，0-360）
        """
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        
        bearing = atan2(y, x)
        bearing = degrees(bearing)
        bearing = (bearing + 360) % 360  # 转换为0-360度
        
        return bearing
    
    def calculate_wind_angle(self, flight_bearing, wind_direction):
        """
        计算风向与飞行方向的夹角
        参数:
            flight_bearing: 飞行方位角（度）
            wind_direction: 风向（度，0-360，风的来向方向，0°为正北，90°为正东）
        返回:
            夹角（度，0-180）
        """
        # 将风向转换为风去的方向（与飞行方向一致的定义）
        wind_to_direction = (wind_direction + 180) % 360
        
        # 计算夹角
        angle_diff = abs(flight_bearing - wind_to_direction)
        
        # 确保夹角在0-180度之间
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        return angle_diff
    
    def load_and_preprocess_data(self):
        """
        加载并预处理飞行数据
        """
        print("开始加载和预处理数据...")
        
        # 定义数据路径
        base_path = "Drone_energy_dataset"
        drone_models = ["UAS04028624", "UAS04028648"]
        
        all_data = []
        
        for model in drone_models:
            file_path = os.path.join(base_path, model, "flightRecord.xlsx")
            
            try:
                df = pd.read_excel(file_path)
                df['drone_model'] = model
                all_data.append(df)
                print(f"成功加载 {model} 数据: {df.shape[0]} 条记录")
                
            except Exception as e:
                print(f"加载 {model} 数据时出错: {e}")
        
        # 合并所有数据
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"总共加载数据: {self.data.shape[0]} 条记录")
            
            # 开始特征预处理
            print("开始特征预处理...")
            
            # 检查必要的列是否存在
            required_cols = ['Start Latitude', 'Start Longitude', 'End Latitude', 'End Longitude',
                            'Distance (m)', 'Battery Used (kWh)', 'Payload (kg)', 
                            'Avg Wind Speed', 'Wind Direction', 'Avg Temperature', 'Avg Humidity']
            
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                print(f"警告：缺少以下列: {missing_cols}")
            
            # 计算飞行方位角
            print("计算飞行方位角...")
            self.data['flight_bearing'] = self.data.apply(
                lambda row: self.calculate_bearing(
                    row['Start Latitude'], row['Start Longitude'],
                    row['End Latitude'], row['End Longitude']
                ), axis=1
            )
            
            # 计算风向与飞行方向的夹角
            print("计算风向与飞行方向的夹角...")
            self.data['wind_angle'] = self.data.apply(
                lambda row: self.calculate_wind_angle(
                    row['flight_bearing'], row['Wind Direction']
                ) if pd.notna(row['Wind Direction']) else np.nan, axis=1
            )
            
            # 重命名列以便于使用
            column_mapping = {
                'Distance (m)': 'distance',
                'Battery Used (kWh)': 'battery_used',
                'Payload (kg)': 'payload',
                'Avg Wind Speed': 'wind_speed',
                'Avg Temperature': 'temperature',
                'Avg Humidity': 'humidity'
            }
            
            self.data = self.data.rename(columns=column_mapping)
            
            # 数据清洗
            print("开始数据清洗...")
            
            initial_count = len(self.data)
            
            # 移除目标变量为空或异常的记录
            self.data = self.data.dropna(subset=['battery_used'])
            self.data = self.data[self.data['battery_used'] > 0]
            
            # 移除距离异常的记录
            self.data = self.data[self.data['distance'] > 0]
            
            # 移除特征值异常的记录
            for feature in ['wind_speed', 'temperature', 'humidity']:
                if feature in self.data.columns:
                    # 移除明显异常的值
                    self.data = self.data[self.data[feature] >= 0]
            
            # 移除风向夹角为空的记录
            self.data = self.data.dropna(subset=['wind_angle'])
            
            final_count = len(self.data)
            removed_count = initial_count - final_count
            
            print(f"数据清洗完成: 移除了 {removed_count} 条异常记录，剩余 {final_count} 条记录")
            
            # 显示清洗后的数据统计
            print("\n清洗后的数据统计:")
            feature_cols = ['distance', 'payload', 'wind_speed', 'wind_angle', 'temperature', 'humidity', 'battery_used']
            available_cols = [col for col in feature_cols if col in self.data.columns]
            print(self.data[available_cols].describe())
            
            print("特征预处理完成")
            
        else:
            raise ValueError("未能加载任何数据")
    
        
    def prepare_features(self):
        """
        准备模型训练的特征和目标变量
        """
        print("准备模型特征...")
        
        # 确保所有需要的特征都存在
        required_features = ['distance', 'payload', 'wind_speed', 'wind_angle', 'temperature', 'humidity']
        available_features = [f for f in required_features if f in self.data.columns]
        
        if len(available_features) != len(required_features):
            missing = set(required_features) - set(available_features)
            print(f"警告：缺少特征: {missing}")
        
        # 准备特征矩阵和目标变量
        X = self.data[available_features].copy()
        y = self.data['battery_used'].copy()
        
        # 检查特征的分布
        print("\n特征统计信息:")
        print(X.describe())
        
        print(f"\n目标变量统计:")
        print(f"电池消耗范围: {y.min():.4f} - {y.max():.4f} kWh")
        print(f"平均电池消耗: {y.mean():.4f} kWh")
        
        return X, y, available_features
        
    
    def build_model(self, X, y, test_size=0.2, random_state=42):
        """
        构建和训练LightGBM模型
        """
        print("构建LightGBM模型...")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        # 设置LightGBM参数 - 进一步优化版本
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 50,        # 进一步增加叶子数量
            'max_depth': 8,          # 增加深度以捕获更复杂的模式
            'learning_rate': 0.03,   # 进一步降低学习率
            'feature_fraction': 0.85, # 稍微增加特征使用比例
            'bagging_fraction': 0.85, # 稍微增加数据使用比例
            'bagging_freq': 3,       # 更频繁的bagging
            'min_data_in_leaf': 15,  # 稍微减少最小叶子数据量
            'min_child_weight': 0.01, # 调整最小权重
            'reg_alpha': 0.05,       # 减少L1正则化
            'reg_lambda': 0.05,      # 减少L2正则化
            'min_split_gain': 0.01,  # 添加最小分裂增益
            'subsample_for_bin': 200000, # 增加采样数量
            'cat_smooth': 10,        # 类别特征平滑
            'verbose': 0,
            'random_state': random_state
        }
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 训练模型
        print("开始训练模型...")
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'eval'],
            num_boost_round=2000,   # 进一步增加训练轮数
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=200)]
        )
        
        # 预测，用于评估模型
        y_pred_train = self.model.predict(X_train, num_iteration=self.model.best_iteration)
        y_pred_test = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        
        # 评估模型
        self._evaluate_model(y_train, y_pred_train, y_test, y_pred_test, X.columns)
        
        return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
    
    def _evaluate_model(self, y_train, y_pred_train, y_test, y_pred_test, feature_names):
        """
        评估模型性能
        """
        print("\n" + "="*50)
        print("模型评估结果")
        print("="*50)
        
        # 训练集评估
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        print(f"训练集性能:")
        print(f"  RMSE: {train_rmse:.6f}")
        print(f"  MAE:  {train_mae:.6f}")
        print(f"  R2:   {train_r2:.6f}")
        
        # 测试集评估
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n测试集性能:")
        print(f"  RMSE: {test_rmse:.6f}")
        print(f"  MAE:  {test_mae:.6f}")
        print(f"  R2:   {test_r2:.6f}")
        
        # 特征重要性
        print(f"\n特征重要性:")
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for feature, imp in feature_importance:
            print(f"  {feature}: {imp:.2f}")
    
    def plot_results(self, y_test, y_pred_test):
        """
        绘制预测结果
        """
        plt.figure(figsize=(12, 5))
        
        # 预测值 vs 真实值散点图
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_test, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('真实值 (kWh)')
        plt.ylabel('预测值 (kWh)')
        plt.title('预测值 vs 真实值')
        plt.grid(True, alpha=0.3)
        
        # 残差图
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred_test
        plt.scatter(y_pred_test, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值 (kWh)')
        plt.ylabel('残差 (kWh)')
        plt.title('残差分布')
        plt.ylim(-1, 1)  # 设置y轴范围为-1到1
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('result/drone_energy_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("结果图表已保存为 'result/drone_energy_model_results.png'")
    
    def save_model(self, filename='drone_energy_model.txt'):
        """
        保存训练好的模型
        """
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        self.model.save_model(filename)
        print(f"模型已保存为: {filename}")
    

if __name__ == "__main__":
    # 创建模型实例并运行完整流程
    model = DroneEnergyModel()
    
    # 1. 加载和预处理数据
    model.load_and_preprocess_data()
    
    # 2. 准备特征
    X, y, feature_names = model.prepare_features()
    
    # 3. 构建和训练模型
    X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = model.build_model(X, y)
    
    # 4. 绘制结果
    model.plot_results(y_test, y_pred_test)  
    
    # 5. 保存模型
    model.save_model('result/drone_energy_lgb_model.txt')
    
    print("\n" + "="*50)
    print("模型训练完成！")
    print("="*50)
    print("模型文件已保存为: result/drone_energy_lgb_model.txt")
    print("可视化结果已保存为: result/drone_energy_model_results.png")
    print("\n使用说明:")
    print("1. 模型输入特征: 距离(m), 载荷(kg), 风速(m/s), 风向夹角(°), 温度(°C), 湿度(%)")
    print("2. 模型输出: 电池消耗(kWh)")
