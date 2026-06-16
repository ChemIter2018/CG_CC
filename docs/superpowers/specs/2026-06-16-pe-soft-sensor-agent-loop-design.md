# 设计规格：裂解炉 PE 软测量 — Agent-Loop 实验框架（ml_lab）

日期：2026-06-16
状态：已批准，进入实现

## 1. 背景与目标

项目数据 `data/00FurnaceCleanData.csv`（约 33,000 行）为裂解炉连续运行读数。目标是构建一个**可复用、配置驱动的实验循环框架** `ml_lab`，用 SOTA 表格回归模型实现 **PE（乙烯收率）软测量**：用当前 19 个可测工艺变量实时估计当前 PE。

**交付物**
- `ml_lab` 框架（数据/划分/模型/调参/评估/排行榜/绘图）
- 全模型真实对比排行榜（含旧模型重测 + AutoGluon 基线）
- 测量 vs 预测图、最优模型存档
- 简短报告（含 agent-loop 迭代记录与结论）

**成功标准**
- 在**按时间划分的留出测试集**上给出诚实 R²/RMSE/MAE
- 量化旧脚本"随机划分 + 全量拟合 scaler"造成的精度高估
- 框架可通过一行 CLI 复现全部实验

## 2. 数据与正确性修复

- **特征**：第 1–19 列（`P, I, N, A, IBP, D10, D30, D50, D70, D90, FBP, DEN, FEED, DS, DSHC, HKPRESSURE, HKTEMP, COT, COP`）
- **目标**：`PE`（最后一列）
- **显式排除** `H2, CH4, C2H4, C2H6, C3H6, C3H8`（下游产物组成，做特征会信息泄漏）
- **划分**：按时间先后 train/val/test = 70/15/15；段间可加 `embargo` 间隔行，缓解相邻近重复行的边界泄漏
- **缩放**：`StandardScaler` 仅在训练集拟合，应用到 val/test（旧代码在划分前对全量拟合，已修正）；树模型免缩放
- 去除旧的"目标 ×100/1000" 数值技巧，直接预测 PE，所有指标用原单位

## 3. 架构与模块

```
ml_lab/
├─ __init__.py
├─ config.py              # dataclass 配置 + 从 YAML 加载
├─ data.py                # 读 CSV、特征/目标定义、缩放
├─ splits.py              # 按时间划分(+embargo)、TimeSeriesSplit CV
├─ metrics.py             # R² / RMSE / MAE
├─ models/
│  ├─ base.py             # BaseRegressor 接口
│  ├─ gbdt.py             # XGBoost / LightGBM / CatBoost
│  ├─ ft_transformer.py   # FT-Transformer (rtdl，支持 MPS)
│  ├─ autogluon_model.py  # AutoGluon TabularPredictor 包装
│  └─ legacy.py           # 旧 SVR/KNR/DTR/MLP 在新划分下重测
├─ tuning.py              # Optuna 调参 (TimeSeriesSplit CV)
├─ experiment.py          # 单次实验编排: 调参→训练→评估→存档→写排行榜
├─ leaderboard.py         # 排行榜 CSV/JSON + 排名
├─ plots.py               # 测量 vs 预测图（沿用现有 seaborn 风格）
└─ run.py                 # CLI 入口
configs/
└─ pe_soft_sensor.yaml    # 实验配置
experiments/
└─ runs/<timestamp>_<model>/   # 每次运行产物：params.json, metrics.json, model, plot.png
└─ leaderboard.csv             # 汇总排行榜
```

### 模块职责与接口

- **config.py**：`ExperimentConfig`（数据路径、特征列、目标、划分比例、embargo、模型列表、每模型调参 trial 数与时间上限、随机种子）。支持 YAML 覆盖。
- **data.py**：`load_xy(cfg) -> (X: DataFrame[19], y: Series)`；`fit_scaler(X_train)`、`apply_scaler`。
- **splits.py**：`chronological_split(n, ratios, embargo) -> (train_idx, val_idx, test_idx)`；`time_series_cv(n_splits)`。保证无重叠、按时间序。
- **metrics.py**：`regression_metrics(y_true, y_pred) -> {r2, rmse, mae}`。
- **models/base.py**：`BaseRegressor`：属性 `name`；方法 `fit(X_tr, y_tr, X_val, y_val)`、`predict(X)`、`save(path)`、`load(path)`（classmethod）、`default_search_space(trial)`（供 Optuna）、`build(params)`。统一让循环以同一方式处理所有模型。
- **gbdt.py**：三个子类，验证集早停。
- **ft_transformer.py**：rtdl FT-Transformer；设备优先 MPS→CPU；验证集早停。
- **autogluon_model.py**：`TabularPredictor`（preset 与 time_limit 可配），作为强自动集成基线。
- **legacy.py**：在修正划分下重训旧 SVR/KNR/DTR/MLP，给可比基线。
- **tuning.py**：`tune(model_cls, X, y, cv, n_trials, timeout) -> best_params`，目标为 CV 平均 R²。
- **experiment.py**：`run_experiment(model_name, cfg) -> metrics`：调参→用最优参在 train(+val) 重训→在 test 评估→存档（参数/指标/模型/图）→写排行榜；整体 try/except，单模型失败不影响 sweep。
- **leaderboard.py**：`append(record)`、`to_table()`，按 R² 降序。
- **plots.py**：`measured_vs_predicted(y_true, y_pred, title, out_path)`。
- **run.py**：`python -m ml_lab.run --config configs/pe_soft_sensor.yaml --models all|<names>`。

## 4. Agent loop 两层

- **内层（自动）**：`run.py` 遍历候选模型 + Optuna 搜索 → 写排行榜，可无头复现。
- **外层（agent 驱动）**：读排行榜与诊断 → 决定下一步（加滞后/滑窗特征、调搜索空间、换模型、处理过/欠拟合）→ 每轮验证后再迭代，记录到报告。不做自主自改 AutoML，保持人在环、可控、有上限。

## 5. 环境

新建隔离环境（conda/venv，python≥3.10）：pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, torch, rtdl_revisiting_models, optuna, autogluon.tabular, matplotlib, seaborn, pyyaml。输出 `ml_lab/requirements.txt`。
注意：AutoGluon 体积大；若安装失败或过慢，将其降级为"可选模型"，sweep 自动跳过并在排行榜标注，不阻塞主流程。

## 6. 评估协议

- 主指标 R²（对齐历史），加 RMSE、MAE（原单位）
- 旧模型同时报"随机划分 vs 时间划分"两套数，量化高估
- 朴素基线（预测训练集均值）作参照下界

## 7. 错误处理 / 可靠性

- 每模型独立 try/except：失败记录错误并继续，排行榜标注 `failed`
- 固定随机种子；记录库版本到运行产物
- 调参与 AutoGluon 均设 trial 数 / 时间上限，控制总耗时

## 8. 测试策略（TDD）

- 单元测试：
  - 划分无重叠且严格按时间序；embargo 生效
  - scaler 仅在训练集拟合（统计量等于训练集统计量）
  - metrics 在解析可算的玩具数据上正确
  - `BaseRegressor` 契约：一个最小 dummy 模型 fit/predict 形状正确
  - leaderboard 追加与按 R² 排序
- **冒烟测试**：在 ~2000 行子样本、每模型 2 个 trial 上跑通整条链路，再跑全量。

## 9. 明确不做（YAGNI）

多目标（其他收率）、自主自改 AutoML、Web UI / MLflow 服务。

## 10. 诚实风险提示

在约 3.3 万行表格上，FT-Transformer/AutoGluon 未必超过调好的 GBDT；"最先进架构"≠一定最优，框架客观呈现。完整 AutoGluon + Optuna sweep 可能较耗时，将限时并在报告说明。

## 11. 实现顺序（高层）

1. 环境与依赖可用性验证（网络/安装能力），失败则给出降级方案
2. 脚手架 + 配置 + 数据/划分/metrics（含单元测试）
3. BaseRegressor + GBDT 三件套 + 调参 + 实验编排 + 排行榜 + 绘图
4. 冒烟测试（子样本）跑通端到端
5. FT-Transformer、legacy 重测、AutoGluon 基线
6. 全量运行、生成排行榜与图
7. 外层 agent-loop 迭代（特征工程/搜索空间）若干轮
8. 报告 + README 更新 + 提交并推送 CG_CC
