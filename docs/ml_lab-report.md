# ml_lab 结果报告 — 裂解炉 PE 软测量

日期：2026-06-17（修订版；订正了 06-16 版关于"R² 为何低"的两条结论，并新增在线 walk-forward 评估）

## TL;DR

- 旧脚本 ~0.9 的 R² **主要是数据泄漏假象**：数据是连续时序、相邻行近乎重复，随机划分把近重复行同时放进训练/测试集（"考过的题再考一遍"）。泄漏量化（CatBoost 同模型）：随机划分 R²=0.903 vs 按时间划分 R²=0.040，高估约 **0.86**。（持续性 y[t]=y[t−1] 的 R²=0.93，印证相邻行多么近似。）
- **R² 低与否，取决于"问"哪个问题**：
  - **远期零反馈外推**（训练前 70% → 预测末 15%、全程无 PE 反馈）：最难的设定，R² 仅 **0.05–0.20**。这是 06-16 版报告的全部内容。
  - **在线可重标定**（部署中可周期拿到 PE 化验值、滚动重训）：用 **walk-forward 滚动评估**，诚实 R² 回到 **~0.56–0.64**（xgboost，每 250 行重标定一次）。✅ 软测量通常是这种场景。
- **订正两条 06-16 结论（被数据推翻）**：
  1. ~~"R² 低主要因测试段 PE 方差极小"~~ —— **不成立**：测试段 std=0.0128，比全序列 0.0112 还**大**。真正原因是**目标本身近常数**（std/mean≈2%）叠加**训练→测试工况漂移**（预测训练均值在测试段 R²=−0.40）。
  2. ~~"19 个工艺变量预测力有限"~~ —— **不成立**：在重标定下，仅用工艺变量的模型 R²=0.64，且明显高于持续性基线（0.55），说明工艺变量确实携带 PE 的同时刻信息。之前弱，纯粹是"模型冻结在旧工况、跨漂移外推"的假象。
- R² 低 ≠ 绝对误差大：各模型 RMSE≈0.008–0.013（PE≈0.5，相对误差约 1.6–2.5%）。

## R² 为什么低 —— 根因拆解（数据实测）

| 诊断 | 数值 | 含义 |
|------|------|------|
| PE std / mean | **2.18%**（范围仅 0.480–0.550） | 目标近常数；R²=1−SSE/Var(y) 的分母极小 → 天生苛刻 |
| 测试段 std vs 全序列 std | 0.0128 vs 0.0112 | 测试段方差**不**小 → 订正旧说法 |
| 预测"训练均值"在测试段 | **R²=−0.40** | 训练/测试 PE 水平不在同一档（工况漂移） |
| 预测"最后训练值" | R²=−1.41 | 漂移很实 |
| 持续性 y[t]=y[t−1] | **R²=0.93** | PE 变化极慢；有近期反馈则天花板很高 |
| Ridge 时间划分 / 随机划分 | **−3.0 / +0.61** | 同模型同特征，再次坐实泄漏 |

把 PE 按时间切 25 块观察：它在炉役内**缓慢漂移**（0.51→峰 0.535→谷 0.496→回升）。训练集落在偏高平台、测试集落在偏低恢复段——这是跨工况的概念漂移，不是噪声，也正是单次划分 R² 低的根因。

## 两种评估口径的结果

### A. 远期零反馈外推（单次时间划分 70/15/15，最难）

| model | R² | RMSE | MAE | 说明 |
|-------|-----|------|-----|------|
| ft_transformer | **0.196** | 0.0115 | 0.0096 | 深度表格 SOTA，本口径最佳 |
| catboost | 0.122 | 0.0120 | 0.0096 | GBDT 最佳 |
| xgboost | 0.047 | 0.0125 | 0.0103 | |
| lightgbm | −0.027 | 0.0130 | 0.0108 | |
| autogluon | −0.152 | 0.0138 | 0.0117 | 自动集成；内部随机/CV 验证不适配时间外推，本口径最差 |

实时结果以 `experiments/leaderboard.csv` 为准。该口径衡量"用旧工况外推未来"，对**可重标定**的部署过于悲观，**不应**作为部署能力的结论。

**值得注意**：AutoGluon（best_quality）在诚实时间划分上反而最差，因为它靠内部随机/CV 选模型与集成，等于在"插值"目标上调优，对"时间外推"过拟合；FT-Transformer 用显式时间验证集早停，外推泛化最好。**评估划分方式比模型本身更决定结果。**

### B. 在线可重标定（滚动 walk-forward，贴近真实部署）

做法：从初始 50% 起，每 `interval` 行重训一次、预测下一块，段间 embargo 50 行，严格时序、不泄漏。代码：`ml_lab/splits.py:walk_forward_split` + `experiment.py:run_walk_forward`，开关 `eval: walk_forward`（`configs/pe_walkforward.yaml`）。

诊断代理（sklearn HistGBR，固定参，全量数据）：

| 重标定间隔 interval | 仅工艺变量 R² | RMSE | hold-last-PE 基线 R² |
|------|------|------|------|
| **250 行** | **0.636** | 0.0076 | 0.552 |
| 500 行 | 0.578 | 0.0082 | 0.416 |
| 1000 行 | 0.461 | 0.0092 | 0.096 |
| 2000 行 | 0.227 | 0.0111 | 0.246 |

ml_lab 正式实跑（xgboost，全量数据，n_trials=3，interval=250）：**R²=0.557，RMSE=0.0084，MAE=0.0063**，与代理同量级。R² 随重标定间隔单调下降——这就是上线要定的工作点：**多久拿到一次化验/重训一次，直接决定可达精度**。

> 注：特征工程（工艺变量滞后/滑窗，见 `experiments_features/`）在单次划分下几乎没用（xgboost 0.047→0.0495）。真正的杠杆是**重标定**，不是堆特征。

## 方法

- **任务**：软测量——用当前 19 个工艺变量（CSV 第 1–19 列）估计当前 PE。显式排除下游产物列（H2…C3H8，会泄漏）。
- **评估**：两种口径——单次时间划分（`eval: single`，70/15/15 + embargo 50）与滚动 walk-forward（`eval: walk_forward`）。两者均"训练集拟合 `StandardScaler`、目标用原单位"。walk-forward 在**初始窗调参一次**、逐折**重训到完整轮数**、各块预测 pooled 后统一计分。
- **调参**：Optuna + TimeSeriesSplit 交叉验证（GBDT 30 trials，FT-Transformer 5 trials）。
- **指标**：R²（对齐历史）+ RMSE + MAE。
- **运行**：每个模型在独立子进程中训练（原生库隔离 + 崩溃隔离）。

## Agent loop 迭代记录

- **Iteration 0（基线）**：4 个 SOTA 模型在单次诚实划分上跑通 → R² 普遍很低（最佳 0.12–0.20）。
- **反思 + 验证**：与旧脚本 ~0.9 严重矛盾 → 假设根因是随机划分泄漏 → `scripts/leakage_comparison.py` 得随机 0.903 vs 时间 0.040，**确认泄漏**。
- **Iteration 1（评估口径）**：意识到单次划分 = 远期零反馈外推，与"可重标定"的真实部署不符 → 实现 walk-forward 滚动评估 → 诚实 R² 回到 0.56–0.64。
- **同时订正根因**：实测发现"测试段方差小"不成立、"工艺变量预测力有限"不成立（见上文根因拆解）。

### 过程中系统性调试修复的工程问题
1. **torch 与 GBDT 的 OpenMP 冲突**（pip 轮子各自携带不同 libomp，同进程并行运算 → 段错误）→ 改为**每模型子进程隔离**（兼得崩溃隔离）。
2. **optuna 与 AutoGluon 环境耦合**（run_one 顶层导入 optuna，AutoGluon 环境无 optuna）→ 加 `tunable` 标志 + optuna **惰性导入**解耦。
3. **AutoGluon DyStack 冲突**（best_quality 动态堆叠 + 传入 tuning_data → "Learner is already fit"）→ 不传 tuning_data、`dynamic_stacking=False`。
4. **walk-forward 早停在漂移数据上欠拟合**：首版给每折切"最近 15% 验证尾"做早停，全量 R² 崩到 −0.011。对照实验：早停**关** R²=0.608 vs 早停**开** R²=0.087（median best_iter≈26）。根因：PE 漂移使"最近尾巴"成另一工况，早停几十棵树就停 → 欠拟合塌成均值预测。→ 每折**拟合到完整轮数**（`X_val=None`，模型大小由那一次调参决定），并加 spy 模型回归测试钉死该契约。

## 结论与建议

1. **现有 19 个工艺变量对 PE 是有预测力的**：可重标定/滚动评估下 R²≈0.56–0.64，且胜过持续性基线（0.55）。06-16 版"预测力有限"的结论**已被推翻**——那只是"冻结旧工况外推"的评估假象。
2. **R² 低的真因是目标近常数（std/mean≈2%）+ 工况漂移**，不是"测试段方差小"。务必同时看 RMSE/MAE（绝对误差）。
3. **按真实部署选评估口径**：
   - 可周期拿到化验值 → 用 **walk-forward**，并据"多久能重标定一次"选定 `interval` 工作点。
   - 必须远期零反馈外推 → 接受低 R²、以 RMSE/MAE 为主，并尝试加**物理漂移协变量**（距上次清焦时间、累计进料、炉役进度）。
4. 别追旧论文的 0.9——那是随机划分泄漏（同模型随机 0.61 vs 时间 −3.0）。

## 复现

```bash
# 主环境（GBDT + FT-Transformer）
conda create -n ml_lab python=3.11
conda run -n ml_lab pip install -r ml_lab/requirements.txt
brew install libomp            # xgboost/lightgbm 需要

# A. 单次时间划分（远期外推口径）
python -m ml_lab.run --config configs/pe_soft_sensor.yaml --models xgboost,lightgbm,catboost,ft_transformer

# B. 在线场景（滚动 walk-forward）
python -m ml_lab.run --config configs/pe_walkforward.yaml --models xgboost,lightgbm,catboost
# 改 configs/pe_walkforward.yaml 的 walk_forward.interval（250/500/1000/2000）即可画"重标定频率 vs R²"曲线

# AutoGluon（独立环境）
conda create -n ml_lab_ag python=3.11
conda run -n ml_lab_ag pip install "autogluon.tabular[all]" pyyaml
python -m ml_lab.run --config configs/pe_soft_sensor.yaml --models autogluon \
  --model-python autogluon=$(conda run -n ml_lab_ag which python)

# 泄漏对比
PYTHONPATH=. python scripts/leakage_comparison.py
```
