# CG_CC — 裂解炉乙烯收率机器学习建模

用多种机器学习 / 深度学习方法，对裂解炉运行数据进行回归建模，预测乙烯收率（PE）。

## 目录结构

```
CG_CC/
├─ data/        # 数据集
│  └─ 00FurnaceCleanData.csv
├─ src/         # 核心建模脚本（编号体现实验顺序）
├─ models/      # 训练产物：模型、标准化器、最优超参数
├─ figures/     # 预测对比图（old/ 为旧版本图）
├─ sandbox/     # 与主项目无关的实验脚本
├─ docs/        # 设计 / 说明文档
├─ requirements.txt
└─ README.md
```

## 数据说明

`data/00FurnaceCleanData.csv`（约 33,000 行）为清洗后的裂解炉运行数据：

- **特征 X**：第 1–19 列（`P, I, N, A, IBP, D10~D90, FBP, DEN, FEED, DS, DSHC, HKPRESSURE, HKTEMP, COT, COP` 等工艺参数与原料性质）
- **目标 y**：第 26 列 `PE`（乙烯收率，约 0.5 量级）
- 因目标值过小，建模时统一乘以系数 `n`（100 或 1000），出图前再还原

通用流程：`读数据 → StandardScaler 标准化 → train_test_split(7:3) → 训练 → r2_score 评估 → 绘制“测量值 vs 预测值”对比图`。

## 建模脚本

| 脚本 | 方法 | 框架 |
|------|------|------|
| `src/01SVR.py` | 支持向量回归（手动调参） | scikit-learn |
| `src/02SVR_GridSearchCV.py` | SVR + 网格搜索 | scikit-learn |
| `src/04KNR_GridSearchCV.py` | K 近邻回归 + 网格搜索 | scikit-learn |
| `src/05DTR_GridSearchCV.py` | 决策树 + AdaBoost + 网格搜索 | scikit-learn |
| `src/07MLP_GridSearchCV.py` | 多层感知机 + 网格搜索 | scikit-learn |
| `src/09Keras_MLP.py` | MLP 全连接网络 | TensorFlow / Keras |
| `src/10Keras_RNN.py` | 多层 LSTM 循环网络 | TensorFlow / Keras |

## 运行方法

脚本使用相对路径，**需在项目根目录运行**：

```bash
pip install -r requirements.txt
python src/01SVR.py        # 其他脚本同理
```

输出：模型与超参数写入 `models/`，预测对比图写入 `figures/`。

## sandbox/

与主项目无关的练手 / 测试脚本，需额外依赖：

- `BingDwenDwen.py` — turtle 绘制冰墩墩（无额外依赖）
- `HUAWEI_YUN_MQTT_PUB.py` — MQTT 推送测试（需 `paho-mqtt`）
- `GPU_Print.py` / `tensorflowtest.py` — TensorFlow 环境与 MNIST 测试（需 `tensorflow-datasets`）

## 注意事项（旧版 API）

脚本编写于 2021 年，部分 API 在新版库中已弃用，若用新版库直接运行会报错：

- Keras：`keras.optimizers.Adam(lr=...)` → 新版改为 `learning_rate=`
- scikit-learn：`DecisionTreeRegressor(criterion='mse'/'mae')` → 新版改为 `'squared_error'`/`'absolute_error'`

建议固定旧版本运行，或后续按上述提示升级代码。
