# G4 Confirmation 冻结协议 (R2 Remediation)

**冻结日期**: 2026-07-14  
**冻结版本**: R2 frozen confirmation set  
**种子**: 20260714

---

## 1. 父序列选择 (Frozen Parents)

### 1.1 入选标准

从原始MPRA数据表(DATA-Table_S2__MPRA_dataset.txt)中选择，必须同时满足：

1. **Split**: 仅test split（chr7, chr13）
2. **序列长度**: 严格200nt
3. **项目来源**: data_project == "CRE"
4. **质量过滤**: 所有三个细胞系的lfcSE最大值 < 1.0（严格小于，不包含等于）
5. **近重复排除**: 排除near_duplicate_audit.json中标记的test-train冲突ID
6. **G0-G3排除**: 排除之前G0/G1/G2/G3已使用过的父序列（validation 24条 + G3 test 24条，共48条）

### 1.2 排序与选择

- 符合条件的序列按**exact_blake2b_128**哈希值排序，其次按ID字典序排序（与derive.py逻辑一致）
- 排除G0-G3用过的48条后，取前**96条**作为最终CONFIRMATION PARENTS
- 所有96条必须位于chr7或chr13上

### 1.3 输出文件

- `checkpoints/FROZEN_PARENTS.tsv.gz`: 包含source_row, IDs, chr, split, sequence, K562_log2FC, HepG2_log2FC, SKNSH_log2FC
- `checkpoints/G0_G3_PARENT_DENYLIST.json`: G0-G3用过的父序列ID列表（48条）

---

## 2. 实验设计参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_parents | 96 | 父序列数量 |
| targets | K562, HepG2, SKNSH | 三个目标细胞系 |
| budgets | 1, 5, 10, 20 | 编辑预算（突变数量） |
| methods | random_matched, greedy_malinois, safeedit_consensus | 三种编辑方法 |
| 预期总行数 | **3456** | 96×3×4×3，硬约束 |
| beam_width | 24 | SafeEdit consensus束搜索宽度 |
| seed | 20260714 | 主随机种子 |
| primary batch size | 128 | Malinois预测批大小 |
| reviewer batch size | 1024 | CNN集成reviewer批大小 |
| device | cuda | GPU计算 |

---

## 3. 校准规则 (Calibration)

### 3.1 校准数据来源

- **仅使用validation split的random edits**进行阈值校准
- validation父序列：从pilot_30k_5k_5k.tsv.gz中取前24条validation CRE（与G3一致）

### 3.2 阈值计算

使用validation random edits的分位数作为固定阈值，**不进行×1.5松弛**：

- strand_disagreement_max: 95%分位数
- reviewer_uncertainty_max: 95%分位数
- naturalness_delta_min: 5%分位数（下界）
- absolute_gc_delta_max: 95%分位数
- max_homopolymer ≤ 6（硬约束）

---

## 4. 统计检验 (Statistical Tests)

### 4.1 McNemar精确检验

- 使用`scipy.stats.binomtest`默认**双侧检验**
- **不进行额外的×2 p值调整**（修正原代码可能的问题）
- 基于不一致对(b,c)进行计算

### 4.2 Bootstrap置信区间

- 迭代次数：**100,000次**
- 重采样方式：**parent-cluster resampling**（按父序列聚类重采样，而非逐行）
- 置信水平：95% CI (2.5%, 97.5%)
- Bootstrap种子：20260713

---

## 5. Pareto最优选择

### 5.1 优化目标

同时最大化两个目标：
1. **primary_margin_gain**: Malinois primary模型的特异性边际增益
2. **reviewer_margin_gain**: CNN集成reviewer的特异性边际增益

### 5.2 近Pareto容忍

满足以下任一条件即为near-Pareto：
- 相对容差：≤ 5%
- 绝对容差：≤ 0.1边际单位

---

## 6. Tier分类逻辑

三个独立列，不合并：

1. **audit_status**: 审计通过/失败
2. **design_method**: 设计方法标识
3. **priority_tier**: 优先级分层

**Tier A标准**（全部满足）：
- method = safeedit_consensus
- audit_pass = true
- no_synth_risk = true
- pareto或near_pareto = true
- core_quality_met = true

---

## 7. 不可行序列处理 (Infeasible)

对于每个(parent_id, target_cell, budget, method)元组：
- **必须输出记录**，即使无法找到满足约束的编辑方案
- 不可行时：
  - design_status = "infeasible"
  - accepted = false
  - 其他必要字段填充占位值
- **不允许缺失任何元组**，最终行数必须严格等于3456

---

## 8. 输出验证 (G4 Acceptor)

最终输出必须通过以下验证：

1. **行数检查**: 正好3456行
2. **元组完整性**: 所有96×3×4×3 = 3456个预期(parent_id, target_cell, budget, method)元组都存在
3. **无重复**: 没有重复的元组
4. **无多余**: 没有超出预期的额外行

---

## 9. 随机数控制

所有随机过程使用固定种子：

| 过程 | 种子 |
|------|------|
| 主种子 | 20260714 |
| validation random paths | seed + 10,000 |
| test random paths | seed + 20,000 |
| Bootstrap | 20260713 |

---

## 10. 文件清单

| 文件 | 说明 |
|------|------|
| checkpoints/FROZEN_PARENTS.tsv.gz | 96条冻结父序列 |
| checkpoints/G0_G3_PARENT_DENYLIST.json | G0-G3排除列表 |
| checkpoints/FROZEN_CONFIG.yaml | 冻结配置参数 |
| checkpoints/FROZEN_PROTOCOL.md | 本文档 |
| src/safeedit_cre/g4_confirmation_frozen.py | 冻结版G4脚本 |
| scripts/prepare_frozen_confirmation.py | 父序列准备脚本 |

---

**本协议一旦冻结，所有参数不得修改。**
