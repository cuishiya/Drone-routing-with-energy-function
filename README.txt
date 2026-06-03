================================================================================
  多行程无人机路径规划与能量预测系统
  Multi-Trip Drone Routing Problem with Energy Function
================================================================================

一、项目概述
-----------
本项目基于论文 "Drone routing with energy function: Formulation and exact algorithm"
(Cheng et al., 2020)，实现了多行程无人机路径规划问题（MTDRP）的建模与求解。

核心思路：
  1. 使用深度学习时序模型（LSTM-Transformer）预测无人机瞬时功率
  2. 基于航迹积分计算弧能耗（上升→巡航→下降三段式航迹）
  3. 采用改进的 RLTS-NSGA-II 多目标优化算法求解路径规划

二、问题定义
-----------
场景：一队同质无人机从中央仓库（Depot）出发，为一组客户提供配送服务。
每架无人机可执行多个行程（Multi-Trip），行程之间可更换电池。

约束条件：
  - 载重约束：无人机任何时候携带货物不超过最大载重 Q
  - 能量约束：单次行程累计能耗不超过电池容量 sigma
  - 时间窗约束：客户有最早/最晚服务时间 [e_i, l_i]
  - 访问约束：每个客户必须且仅被服务一次

优化目标（多目标）：
  - 目标1：最小化总飞行距离
  - 目标2：最小化总能量消耗

三、能耗预测模型
---------------
3.1 航迹模式
  弧飞行采用三段式航迹：垂直上升 → 水平巡航 → 垂直下降
  对整条航迹逐秒采样，生成时序特征序列，输入深度学习模型预测瞬时功率，
  再积分得到弧总能耗。

3.2 模型输入特征（8个特征）
  - Height: 高度 [m]
  - VS: 竖直速度 [m/s]
  - GS: 地速 [m/s]
  - Wind Speed: 风速 [m/s]
  - Temperature: 温度 [°C]
  - Humidity: 湿度 [%]
  - Wind Angle: 风向夹角 [度]
  - Payload: 当前载重 [kg]

3.3 模型输出
  - 瞬时功率 [W]

3.4 支持的时序深度学习模型

  模型                   RMSE (W)    MAE (W)     R²        MAPE (%)  参数量
  --------------------------------------------------------------------------
  Bi-LSTM                397.78      291.08      0.8287    9.05      996,866
  GRU Seq2Seq            402.50      295.57      0.8247    7.91      994,817
  LSTM Seq2Seq           412.86      305.24      0.8155    9.22      1,326,337
  Transformer            418.51      292.74      0.8104    7.47      802,433
  LSTM-Transformer       当前主模型（融合LSTM时序建模+Transformer全局注意力）
  LSTM-FC                全连接预测头变体
  TCN                    时序卷积网络
  Informer               长序列预测模型

  当前求解器默认使用 LSTM-Transformer 模型（result/power_lstm_transformer_model.pth）

四、求解算法：RLTS-NSGA-II
-------------------------
基于 NSGA-II 多目标遗传算法，集成以下改进：
  - Q-Learning 自适应参数调节：动态调整交叉率 CR 和变异率 M
  - 反应式禁忌搜索（RLTS）：周期性对精英个体进行局部搜索改进
  - 多行程时间衔接：跟踪每架无人机的可用时间，实现行程间电池更换
  - 载重流显式记录：记录每条弧的实时载重 q_ij
  - 能量累计检查：使用 f_i 变量逐节点检查能量约束

编码方案：
  [customer_order[0..n-1], drone_assignment[0..n-1], trip_assignment[0..n-1]]
  维度 = 客户数 × 3

五、项目文件结构
---------------

核心求解模块：
  mtdrp_energy_model.py      MTDRP问题建模、LSTM-Transformer功率模型、解评估
  mtdrp_rlts_nsga2.py         RLTS-NSGA-II求解器主程序（直接运行）

模型训练脚本（model_training/）：
  train_lstm_transformer_model.py   LSTM-Transformer混合模型训练
  train_bilstm_model.py             Bi-LSTM模型训练
  train_gru_model.py                GRU Seq2Seq模型训练
  train_lstm_seq2seq_model.py       LSTM Seq2Seq模型训练
  train_transformer_model.py        Transformer模型训练
  train_lstm_fc_model.py            LSTM-FC模型训练
  train_tcn_model.py                TCN模型训练
  train_informer_model.py           Informer模型训练

模型评估脚本（model_evaluation/）：
  evaluate_all_models.py            统一评估所有模型
  ablation_feature_groups.py        特征消融实验

训练好的模型文件（result/）：
  power_lstm_transformer_model.pth  当前主模型（LSTM-Transformer）
  power_bilstm_v2_model.pth         Bi-LSTM v2
  power_gru_model.pth               GRU Seq2Seq
  power_lstm_seq2seq_model.pth      LSTM Seq2Seq
  power_transformer_model.pth       Transformer
  power_lstm_fc_model.pth           LSTM-FC
  power_tcn_model.pth               TCN
  power_informer_model.pth          Informer
  各模型对应的 _scalers.pkl         特征标准化参数

数据目录：
  Drone_energy_dataset/             原始飞行轨迹数据（UAS04028624/48）
  标准算例/                         标准测试算例（含 test_small_20.dat）
  Cheng_Instances/                  Cheng et al. 论文原始算例
  深圳光明实例算例/                 深圳光明区顺丰站点真实案例

六、快速开始
-----------

6.1 环境要求
  - Python 3.11+
  - PyTorch, pygmo, numpy, matplotlib, scikit-learn
  - 推荐使用 Anaconda 虚拟环境

6.2 运行求解器
  # 激活虚拟环境后执行：
  python mtdrp_rlts_nsga2.py

  默认使用 test_small_20.dat（20客户/5无人机）测试实例。
  求解参数可在主函数中调整：population_size, generations, tabu_frequency 等。

6.3 切换功率模型
  修改 mtdrp_rlts_nsga2.py 中的 POWER_MODEL_TYPE 变量：
    "lstm_transformer"  → LSTM-Transformer（默认）
    "bilstm"            → Bi-LSTM
    "gru"               → GRU Seq2Seq
    "lstm"              → LSTM Seq2Seq
    "transformer"       → Transformer

6.4 训练新模型
  cd model_training
  python train_lstm_transformer_model.py

七、输出结果
-----------
运行完成后输出：
  - 控制台：每代耗时、帕累托前沿规模、最优解详情
  - result/mtdrp_rlts_nsga2_evolution.png  进化曲线（成本/能耗/帕累托规模）
  - result/mtdrp_rlts_nsga2_routes.png     最优路径可视化

八、关键改进点（相对原始论文）
-----------------------------
  1. 能耗模型：从简化物理公式升级为 LSTM-Transformer 深度学习预测
  2. 多行程支持：允许无人机执行多次行程，行程间可更换电池
  3. 时间窗约束：新增客户时间窗 [e_i, l_i] 约束
  4. 多目标优化：同时优化总距离和总能耗
  5. Q-Learning 自适应：动态调整遗传算法参数
  6. 真实气象数据：集成温度、湿度、风速、风向等低空气象因素


