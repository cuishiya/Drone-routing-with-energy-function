import pandas as pd
import os

def read_flight_records():
    """
    读取三个无人机型号的飞行记录数据
    """
    # 定义数据路径
    base_path = "Drone_energy_dataset"
    drone_models = ["UAS04028624", "UAS04028648", "UAS04143500"]
    
    flight_records = {}
    
    for model in drone_models:
        file_path = os.path.join(base_path, model, "flightRecord.xlsx")
        
        try:
            # 读取Excel文件
            df = pd.read_excel(file_path)
            flight_records[model] = df
            
            print(f"\n=== {model} 飞行记录 ===")
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print("\n前5行数据:")
            print(df.head())
            print("\n数据统计信息:")
            print(df.describe())
            
        except Exception as e:
            print(f"读取 {model} 文件时出错: {e}")
    
    return flight_records

def detailed_analysis(flight_records):
    """
    对飞行记录进行详细分析
    """
    print("\n" + "="*60)
    print("详细数据分析")
    print("="*60)
    
    for model, df in flight_records.items():
        print(f"\n### {model} 详细分析 ###")
        
        # 基本信息
        print(f"总飞行次数: {len(df)}")
        print(f"数据时间范围: {df['Start Time'].min()} 到 {df['End Time'].max()}")
        
        # 飞行距离分析
        if 'Distance (m)' in df.columns:
            print(f"平均飞行距离: {df['Distance (m)'].mean():.2f} 米")
            print(f"最大飞行距离: {df['Distance (m)'].max():.2f} 米")
            print(f"最小飞行距离: {df['Distance (m)'].min():.2f} 米")
        
        # 电池使用分析
        if 'Battery Used (kWh)' in df.columns:
            print(f"平均电池消耗: {df['Battery Used (kWh)'].mean():.4f} kWh")
            print(f"最大电池消耗: {df['Battery Used (kWh)'].max():.4f} kWh")
            print(f"总电池消耗: {df['Battery Used (kWh)'].sum():.4f} kWh")
        
        # 载荷分析
        if 'Payload (kg)' in df.columns:
            print(f"平均载荷: {df['Payload (kg)'].mean():.2f} kg")
            print(f"最大载荷: {df['Payload (kg)'].max():.2f} kg")
        
        # 环境条件分析
        if 'Avg Temperature' in df.columns:
            print(f"平均温度: {df['Avg Temperature'].mean():.1f}°C")
        if 'Avg Humidity' in df.columns:
            print(f"平均湿度: {df['Avg Humidity'].mean():.1f}%")
        if 'Avg Wind Speed' in df.columns:
            print(f"平均风速: {df['Avg Wind Speed'].mean():.1f}")

def compare_models(flight_records):
    """
    比较不同无人机型号的性能
    """
    print("\n" + "="*60)
    print("无人机型号对比分析")
    print("="*60)
    
    comparison_data = {}
    
    for model, df in flight_records.items():
        comparison_data[model] = {
            '飞行次数': len(df),
            '平均飞行距离(m)': df['Distance (m)'].mean() if 'Distance (m)' in df.columns else 0,
            '平均电池消耗(kWh)': df['Battery Used (kWh)'].mean() if 'Battery Used (kWh)' in df.columns else 0,
            '平均载荷(kg)': df['Payload (kg)'].mean() if 'Payload (kg)' in df.columns else 0,
            '能效比(m/kWh)': (df['Distance (m)'].mean() / df['Battery Used (kWh)'].mean()) if 'Distance (m)' in df.columns and 'Battery Used (kWh)' in df.columns else 0
        }
    
    # 创建对比表格
    comparison_df = pd.DataFrame(comparison_data).T
    print("\n对比表格:")
    print(comparison_df.round(2))

if __name__ == "__main__":
    records = read_flight_records()
    if records:
        detailed_analysis(records)
        compare_models(records)
