# 仓库重组设计 — CG / CG_CC

日期：2026-06-16

## 目标
对裂解炉乙烯收率建模项目进行目录归类整理，为后续开发提供清晰结构，并推送到新建的私有仓库 `CG_CC`。

## 背景
项目原为扁平结构：7 个核心 ML 脚本、数据 CSV、保存的模型、图片以及若干与项目无关的实验脚本混在根目录。脚本使用硬编码的相对路径（假定从项目根运行）。

## 目标结构（标准化分层）
```
CG_CC/
├─ data/        # 数据集
│  └─ 00FurnaceCleanData.csv
├─ src/         # 核心建模脚本（保留 01/02/.. 编号以体现实验顺序）
│  ├─ 01SVR.py
│  ├─ 02SVR_GridSearchCV.py
│  ├─ 04KNR_GridSearchCV.py
│  ├─ 05DTR_GridSearchCV.py
│  ├─ 07MLP_GridSearchCV.py
│  ├─ 09Keras_MLP.py
│  └─ 10Keras_RNN.py
├─ models/      # 训练产物（原 Saved_Model/）
├─ figures/     # 预测对比图（原 Pictures/ + 01SVR_PE.png；old/ 为旧图）
├─ sandbox/     # 与主项目无关的实验脚本
├─ docs/        # 设计文档
├─ README.md
├─ requirements.txt
└─ .gitignore
```

## 关键决策
- **保留编号文件名**：`01/02/04/05/07/09/10` 体现实验顺序，重命名会丢失语义。
- **保持"从项目根运行"约定**：脚本内相对路径相应更新（`data/` 读、`models/` 与 `figures/` 写），运行方式 `python src/xxx.py`。
- **无关脚本归入 `sandbox/`**：BingDwenDwen（turtle 绘图）、HUAWEI_YUN_MQTT_PUB、GPU_Print、tensorflowtest。
- **新增 README 与 requirements** 作为后续开发入口。

## 路径修改映射（每个核心脚本）
- `"00FurnaceCleanData.csv"` → `"data/00FurnaceCleanData.csv"`
- `Saved_Model/...` → `models/...`
- `Pictures/...` → `figures/...`
- `01SVR.py`：`'01SVR_PE.png'` → `'figures/01SVR_PE.png'`

## 验证
- 全部脚本 `py_compile` 语法检查。
- 确认新路径（`data/...csv`、`models/`、`figures/`）存在。
- 不实际跑训练（过重，如 09 跑 30000 轮）；运行结果以后续实跑为准。

## 收尾
所有移动用 `git mv` 保留历史；提交后推送到 `cg_cc` 远程（CG_CC 私有仓库）。
