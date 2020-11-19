## 贪吃蛇AI

### 项目简介
本项目是基于深度强化学习（DDQN）实现的一个贪吃蛇AI

### Installation
1. 新建Anaconda虚拟环境

   ```
   conda create -n snake python=3.7 -y
   ```
   
2. 激活环境

   ```
   conda activate snake
   ```

3. 安装pytorch, torchvision

   ```
   conda install pytorch torchvision -c pytorch
   ```

4. 安装依赖包

   ```
   pip install -r requirements.txt
   ```

5. 运行

   ```
   python agent_play.py
   ```
   

### 文件目录结构
```
├── README.md
├── agent_play.py       // AI运行贪吃蛇
├── configs.py          // 配置文件
├── human_play.py       // 手动玩蛇
├── img                 // 存放可视化结果（忽略）
├── model
│   ├── __init__.py
│   └── dqn.py          // DQN网络结构
├── objects.py          // 场景物体
├── requirements.txt
├── train.py            // 训练脚本
├── utils
│   ├── __init__.py
│   ├── image_transform.py    // 图像变换
│   ├── memory.py       // 记忆单元
│   └── utils.py        // 一些必要的工具函数
├── visualize.py        // 可视化
└── weights             // 存放模型参数
    ├── dqn2_reduced.pt
    ├── dqn3_reduced.pt
    └── dqn_reduced.pt
```




