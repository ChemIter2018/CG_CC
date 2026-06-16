# ml_lab 结果报告 — 裂解炉 PE 软测量

日期：2026-06-16

## TL;DR

- 旧脚本报告的 ~0.9 的 R² **主要是数据泄漏假象**。数据是连续运行的时序读数，相邻行近乎重复；随机划分把这些近重复行同时放进训练集和测试集，等于"考过的题再考一遍"。
- **泄漏量化**（CatBoost 默认参数，同一模型）：
  - 随机划分（旧做法）：**R²=0.903**，RMSE=0.0035
  - 按时间划分（诚实）：**R²=0.040**，RMSE=0.0126
  - 随机划分把 R² 高估了 **0.86**
- 在诚实的按时间划分下，用当前 19 个工艺变量"外推"预测 PE 很难。**FT-Transformer（深度表格 SOTA）表现最好**。
- 注意：R² 低 ≠ 绝对误差大。RMSE ≈ 0.012（PE≈0.5，约 2.4% 相对误差）——模型在绝对量级上预测尚可，只是难以解释测试时段那点很小的方差。

## 诚实排行榜（按时间划分 test 集，PE 原单位）

| model | R² | RMSE | MAE | 说明 |
|-------|-----|------|-----|------|
| ft_transformer | **0.196** | 0.0115 | 0.0096 | 深度表格 SOTA，最佳 |
| catboost | 0.122 | 0.0120 | 0.0096 | GBDT 最佳 |
| xgboost | 0.047 | 0.0125 | 0.0103 | |
| lightgbm | −0.027 | 0.0130 | 0.0108 | |
| autogluon | −0.152 | 0.0138 | 0.0117 | 自动集成；内部随机/CV 验证不适配时间外推，诚实划分下最差 |

实时结果以 `experiments/leaderboard.csv` 为准。

**值得注意**：AutoGluon（best_quality，10 分钟预算）在诚实的时间划分上反而最差（R²=−0.152）。原因正是本报告的主题——它靠内部随机/交叉验证选模型与集成，等于在"插值"目标上调优，对"时间外推"反而过拟合；而 FT-Transformer 用显式的时间验证集早停，外推泛化最好。这再次印证：**评估划分方式比模型本身更决定结果**。

## 方法

- **任务**：软测量——用当前 19 个工艺变量（CSV 第 1–19 列）估计当前 PE。显式排除下游产物列（H2…C3H8，会泄漏）。
- **划分**：按时间先后 70/15/15 + 段间 embargo 50 行；`StandardScaler` 仅在训练集拟合；目标用原单位（去掉旧的 ×100/1000 技巧）。
- **调参**：Optuna + TimeSeriesSplit 交叉验证（GBDT 30 trials，FT-Transformer 5 trials）。
- **指标**：R²（对齐历史）+ RMSE + MAE。
- **运行**：每个模型在独立子进程中训练（原生库隔离 + 崩溃隔离）。

## Agent loop 迭代记录

- **Iteration 0（基线）**：4 个 SOTA 模型在诚实划分上跑通 → R² 普遍很低（最佳 0.12–0.20）。
- **反思**：与旧脚本 ~0.9 严重矛盾 → 假设根因是随机划分泄漏。
- **验证**：`scripts/leakage_comparison.py` → 随机 0.903 vs 时间 0.040，**确认泄漏**。
- **观察**：FT-Transformer 在诚实划分下最佳（0.196），略胜调好的 GBDT。

### 过程中系统性调试修复的工程问题
1. **torch 与 GBDT 的 OpenMP 冲突**（pip 轮子各自携带不同 libomp，同进程做并行运算 → 段错误）→ 改为**每模型子进程隔离**（同时实现真正的崩溃隔离，单模型崩溃不再中断整轮）。
2. **optuna 与 AutoGluon 环境耦合**（run_one 顶层导入 optuna，AutoGluon 环境无 optuna）→ 加 `tunable` 标志 + optuna **惰性导入**解耦。
3. **AutoGluon DyStack 冲突**（best_quality 的动态堆叠 + 传入 tuning_data → "Learner is already fit"）→ 不传 tuning_data、`dynamic_stacking=False`。

## 结论与建议（后续可迭代方向）

1. 现有 19 个工艺变量对 PE 的"未来时段外推"预测力有限——这是诚实结论，不是 bug。
2. R² 低主要因测试时段 PE 方差极小；可同时关注 RMSE/MAE（绝对误差）。
3. 后续值得尝试：
   - **特征工程**：加入工艺变量的滞后/滑窗统计（对软测量是合法输入）。
   - **分组/分块评估**：按运行工况（如清洗周期）分块，避免相邻行泄漏的同时衡量插值能力。
   - **重新明确部署场景**：若软测量是"连续在线插值"而非"远期外推"，评估方式应相应调整——但务必继续避免相邻近重复行的泄漏。

## 复现

```bash
# 主环境（GBDT + FT-Transformer）
conda create -n ml_lab python=3.11
conda run -n ml_lab pip install -r ml_lab/requirements.txt
brew install libomp            # xgboost/lightgbm 需要

python -m ml_lab.run --config configs/pe_soft_sensor.yaml --models xgboost,lightgbm,catboost,ft_transformer

# AutoGluon（独立环境）
conda create -n ml_lab_ag python=3.11
conda run -n ml_lab_ag pip install "autogluon.tabular[all]" pyyaml
python -m ml_lab.run --config configs/pe_soft_sensor.yaml --models autogluon \
  --model-python autogluon=$(conda run -n ml_lab_ag which python)

# 泄漏对比
PYTHONPATH=. python scripts/leakage_comparison.py
```
