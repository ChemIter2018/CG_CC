# CG_CC — 裂解炉乙烯收率机器学习建模

用多种机器学习 / 深度学习方法，对裂解炉运行数据进行回归建模，预测乙烯收率（PE）。

## 目录结构

```
CG_CC/
├─ data/        # 数据集
│  └─ 00FurnaceCleanData.csv
├─ ml_lab/      # ★ Agent-loop 实验框架（当前工作，SOTA 模型）
├─ configs/     # ml_lab 实验配置 (YAML)
├─ experiments/ # ml_lab 运行产物：leaderboard.csv、每模型图与结果
├─ scripts/     # 测试 / 分析脚本 (run_tests.sh, leakage_comparison.py)
├─ tests/       # ml_lab 单元测试 (pytest)
├─ src/         # 旧建模脚本（2021，编号体现实验顺序）
├─ models/      # 旧脚本训练产物
├─ figures/     # 旧脚本预测对比图（old/ 为更早版本）
├─ sandbox/     # 与主项目无关的实验脚本
├─ docs/        # 设计 / 说明文档（含 ml_lab-report.md）
├─ requirements.txt
└─ README.md
```

## ml_lab —— Agent-loop 实验框架（当前工作）

`ml_lab/` 是一个可复用、配置驱动的实验循环框架，用当前 SOTA 表格回归模型做 **PE 软测量**（用 19 个工艺变量估计当前 PE），并修正了旧脚本的数据泄漏问题。

**关键发现**：旧脚本 ~0.9 的 R² 主要是**数据泄漏假象**。数据是连续时序、相邻行近重复；随机划分会把近重复行同时放进训练/测试集。同一 CatBoost 模型下——随机划分 R²=0.903 vs 按时间划分 R²=0.040（高估 0.86）。诚实划分下 PE 难以外推预测，其中 **FT-Transformer 表现最好**。完整结论见 [`docs/ml_lab-report.md`](docs/ml_lab-report.md)。

**特性**：
- 统一模型接口：XGBoost / LightGBM / CatBoost / FT-Transformer / AutoGluon（+ 旧模型重测）
- 按时间划分 + embargo，scaler 仅训练集拟合（杜绝泄漏）
- Optuna + TimeSeriesSplit 调参；R²/RMSE/MAE 排行榜与图
- **每模型独立子进程运行**：隔离 torch/GBDT 的 OpenMP 冲突，且单模型崩溃不中断整轮

**运行**：

```bash
# 主环境
conda create -n ml_lab python=3.11
conda run -n ml_lab pip install -r ml_lab/requirements.txt
brew install libomp                       # xgboost/lightgbm 需要 (macOS)

python -m ml_lab.run --config configs/pe_soft_sensor.yaml \
    --models xgboost,lightgbm,catboost,ft_transformer

# AutoGluon 在独立环境（见 ml_lab/requirements.txt 注释），通过 --model-python 调用
# 测试（torch 与 GBDT 需分进程跑）
PYTHON=$(conda run -n ml_lab which python) ./scripts/run_tests.sh
```

---

## 旧建模脚本（src/，2021）

下面是项目最初的脚本，作为对照保留（注意：它们使用随机划分，R² 偏乐观，见上文泄漏说明）。

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
