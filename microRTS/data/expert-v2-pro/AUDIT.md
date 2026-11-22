# MicroRTS 专家轨迹数据质量审计报告

## 1. 质量门禁总览

| 检查项 | 状态 | 详情 |
| --- | :---: | --- |
| `schema_validity` | **PASS** | schema_errors=0 |
| `token_length_budget` | **PASS** | train p50=474, p95=898, max=1024, gt_2048=0 |
| `bot_skill_match_fidelity` | **PASS** | wait_fallback_rate=0.00%, wait_rate=7.37% |
| `tactical_diversity` | **PASS** | skills=7 (['attack', 'harvest', 'produce', 'move', 'wait', 'return', 'build']), barracks=True, combat_units=True |
| `player_symmetry` | **PASS** | P0=12158 (49.5%), P1=12391 (50.5%) |
| `duplicate_control` | **PASS** | dup_row_fraction=0.1%, conflict_groups=29 |
| `train_val_isolation` | **PASS** | shared_maps=0, shared_contexts=0, shared_prompts=0 |

## 2. 数据集规模与切分隔离

| 指标 | 训练集 (train) | 验证集 (val, 未见地图) |
| --- | ---: | ---: |
| 决策行数 | 24549 | 6045 |
| 对局数 | 362 | 103 |
| 唯一上下文数 | 16811 | 4581 |
| 唯一完整题面数 | 24520 | 6041 |
| Schema 校验错误 | 0 | 0 |
| 平均候选数 K | 3.92 (2–8) | 4.61 (2–8) |
| Wait 占比 | 7.37% | 3.72% |
| Wait 回退失配率 | 0.00% | 0.00% |
| 选项0位置占比 (打乱前) | 30.4% | 25.1% |
| 重复题面冗余率 | 0.1% | 0.1% |
| 同题不同标冲突组 | 29 | 4 |
| Token P50 / P95 / Max | 474 / 898 / 1024 | 417 / 539 / 673 |
| Token >512 / >1024 / >2048 | 10851 / 0 / 0 | 619 / 0 / 0 |

- **Train / Val 隔离**：共享地图 `0` 张，共享上下文 `0` 个，共享完整题面 `0` 个。

## 3. 技能、兵种与匹配质量分布（训练集）

**技能分布**：`{"attack": 8298, "harvest": 7482, "produce": 3818, "move": 2019, "wait": 1810, "return": 677, "build": 445}`  
**兵种分布**：`{"Worker": 17694, "Base": 3527, "Ranged": 1233, "Heavy": 955, "Barracks": 805, "Light": 335}`  
**动作匹配方式**：`{"exact": 18794, "semantic": 3945, "wait_native": 1810}`  
**教师分布**：`{"ObiBotKenobi": 6762, "TMA": 5533, "CoacAI": 5384, "Mayari": 3990, "Droplet": 1639, "StrategyTactics": 1241}`  
**对手分布**：`{"WorkerRush": 5239, "HeavyRush": 5099, "LightRush": 4537, "RangedRush": 3934, "CoacAI": 2565, "ObiBotKenobi": 1152, "TMA": 1108, "Mayari": 915}`  
**阵营分布**：`{"1": 12391, "0": 12158}`  
**时间阶段 (tick)**：`{"0-99": 2353, "100-299": 8378, "300-699": 9107, "700+": 4711}`

## 4. 分地图规模与 Token 长度（训练集）

| 地图 | 行数 | P50 tok | P95 tok | Max tok | >512 | >1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `basesWorkers10x10.xml` | 1213 | 319 | 406 | 472 | 0 | 0 |
| `FourBasesWorkers12x12.xml` | 6793 | 815 | 961 | 1024 | 6745 | 0 |
| `basesWorkers12x12A.xml` | 2867 | 446 | 605 | 751 | 634 | 0 |
| `basesWorkers12x12C.xml` | 1112 | 374 | 495 | 604 | 40 | 0 |
| `FourBasesWorkers8x8.xml` | 1846 | 690 | 824 | 926 | 1838 | 0 |
| `basesWorkers8x8A.xml` | 988 | 313 | 388 | 437 | 0 | 0 |
| `basesWorkers8x8B.xml` | 1041 | 313 | 381 | 437 | 0 | 0 |
| `basesWorkers8x8D.xml` | 958 | 334 | 416 | 461 | 0 | 0 |
| `basesWorkers8x8Obstacle.xml` | 848 | 357 | 466 | 606 | 21 | 0 |
| `NoWhereToRun9x8.xml` | 2782 | 388 | 500 | 575 | 83 | 0 |
| `TwoBasesBarracks16x16.xml` | 1170 | 577 | 754 | 881 | 938 | 0 |
| `basesWorkers16x16A.xml` | 2931 | 458 | 559 | 673 | 552 | 0 |

## 5. 典型同题多标签冲突样例（策略多样性/歧义）

- `FourBasesWorkers8x8.xml` t=50 — Best skill for Base U68 at (5,5) to win this match? → `{'U68: produce Worker at (4,5)': 1, 'Wait 16t': 1}`
- `FourBasesWorkers8x8.xml` t=50 — Best skill for Base U68 at (5,5) to win this match? → `{'U68: produce Worker at (6,5)': 1, 'Wait 16t': 1}`
- `NoWhereToRun9x8.xml` t=50 — Best skill for Base U4 at (1,1) to win this match? → `{'U4: produce Worker at (1,2)': 1, 'Wait 16t': 1}`
- `NoWhereToRun9x8.xml` t=50 — Best skill for Base U4 at (1,1) to win this match? → `{'U4: produce Worker at (1,2)': 1, 'Wait 16t': 1}`
- `NoWhereToRun9x8.xml` t=50 — Best skill for Base U4 at (1,1) to win this match? → `{'U4: produce Worker at (1,2)': 1, 'Wait 16t': 1}`
