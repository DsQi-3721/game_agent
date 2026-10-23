# System 1 与 HyMem 的深度融合与学习闭环：设计与首轮验证

> 日期：2026-10-23 · 前置报告：`jev_system1_design.md`（Jev 调研、快路径、记忆路由、实测）。本报告讲两件事：**System 1 如何进入 HyMem 内部**，以及**如何形成可迭代进化的学习闭环**。
> 交接说明：`HANDOVER.md`。

## 1. 融合的三个层次

上一版只在两个边界接入：决策入口和记忆视图。这一版把 System 1 放进 HyMem 自己的机制里。融合分三层，每一层都可以单独开关（`llm_hymem_dual` 的参数）。

| 层次 | 机制 | 代码 | 参数 |
|---|---|---|---|
| **前置：决定谁来决策** | Tier 0 规则 → student → teacher 的级联；门控放行则直接出手，否则交给 HyMem，并附上 System 1 的排序作为提示；HyMem 超时则打出 System 1 的答案 | `agents.py: S1Stack, _decide_dual` | `tier0`、`student_url`、`teacher`、`slow_budget_s` |
| **内部：System 1 驱动 HyMem 的选择题** | ① **共享事实**：planner 看到 System 1 决策所依据的代码事实，例行的伤害/速度/换人数字不必再派 executor 去算<br>② **步骤控制**：planner 第一步之前问 System One"直接回答 / 调研 / 深思"，若"直接回答"把握足够，就把这一步改成只回答（一次短调用）<br>③ **任务路由**：planner 的 `<task>` 交给 executor 还是隔离推理，由 System One 判断，替代规则或 LLM 路由<br>④ **执行器**：System One 用函数调用的方式（工具菜单 + 对局实体的封闭集合）选工具和参数，代码执行工具，按规则汇总成 `<RESULTS>`，**不再有 executor 的 LLM 调用** | `controller.py: facts_block, S1Controller, S1Router, S1Executor`；`planner.py` 新增 `answer_now` 钩子 | `shared_facts`、`step_control`、`s1_router`、`s1_executor` |
| **记忆：System 1 整理 System 2 的记忆，并反馈回来** | 记忆路由（kind / valid / useful）按空间分发；**类型化折叠**：去掉 3 份生成式抽取，记忆来自 planner 摘要（`typed`）或纯规则来源（`typed_light`），由路由器整理；**反向通道**：整理后的对手信息/计划/经验作为 known_facts 交给 System 1；折叠在后台执行 | `memory_router.py`；`memory.py: fold_mode typed/typed_light`；`agents.py: _DualPlanner` | `memory_router: jev`、`fold_mode`、`fast_memory` |

本机慢路径的耗时主要在 executor 的多轮 LLM 调用（每个任务 3–6 次，约 40–80 s）和折叠的生成式抽取（约 40 s）。内部层的 ③④ 和类型化折叠直接删掉这两部分；②在 System 1 判断"事实已经够了"时，把慢路径压缩成一次 planner 调用。

## 2. 学习闭环

```
            ┌────────────────────────── 对局（arena） ──────────────────────────┐
            │  Tier 0 → student（本地）→ teacher（Jev）→ System 2（HyMem）         │
            │  每次 System One 请求 / 每个决策 / 每局胜负 → s1exp_<agent>.jsonl     │
            └──────────────┬───────────────────────────────────────────────────────┘
                           ▼
   数据集（s1_dataset）：teacher 软标签 + System 2 纠正（s2_weight）+ 胜负加权；按对局划分 train/valid
                           ▼
   蒸馏（s1_train）：Qwen3.5 + LoRA，在选项字母位置上做软标签交叉熵（mlx：Mac；hf/peft：GPU）
                           ▼
   评估（s1_eval）：与 teacher 的一致率 / KL / ECE / 延迟，对比未训练 base 和现任 champion
                           ▼
   注册 / 晋升（registry）：更一致且校准不变差 → champion
                           ▼
   服务（s1_serve，TypeSafe 兼容）→ 级联部署（student_url）→ 只有 student 不确定的回合才问 teacher
                           └──────────────── 新的经验 ───────────────► 下一轮
```

设计要点：
- **同一个接口**：Jev 和本地 student 都讲 TypeSafe 协议（`/v1/systemone`）。智能体只认 `student_url` 和 `TYPESAFE_BASE_URL`，换模型不用改代码。
- **同一份渲染**：服务、训练、数据集都使用 `arena/s1/prompts.py` 的同一套渲染（状态在前，选项编号为字母），保证 student 被评分时看到的文本就是它训练时的文本。所有问题共享状态前缀，状态只做一次 prefill。
- **软标签**：student 学 teacher 的完整分布，而不只是 argmax。门控依赖的正是"多确定"，所以不确定性也要学。
- **主动学习自然发生**：级联只在 student 不确定时调用 teacher，这些回合的 teacher 答案正是下一轮最有价值的标签。
- **System 2 纠正 System 1**：慢路径完成的决策（`s2_key`）以 `s2_weight` 混入动作类标签，这是"慢系统训练快系统"的通道；胜负作为样本权重。
- **可替换的 teacher**：没有 Jev 时 `teacher: false`，student 独立运行；之后的标签来自 System 2 和胜负（待实现的分支见 HANDOVER §4.4）。

## 3. 首轮验证（本机）

### 3.1 数据

| 来源 | 记录数 | 说明 |
|---|---|---|
| 399 个离线局面（`eval_jev_decisions.py --record`） | 399 次请求 | jev-1.13，split 布局 |
| 24 局 teacher 对战（`s1_collect_randbats.yaml`） | 593 条（s1 + decision + outcome） | jev113c 对 random 8–0、max-power 8–0、heuristic 6–2；花费 $0.09 |
| 数据集 s1-v1 | 4,758 道题 | train 3,753 / valid 1,005，按 34 个对局分组 |

### 3.2 student s1-v1（Qwen3.5-0.8B，Mac，head LoRA）

| 指标（valid 中 60 个请求、350 道题） | 未训练 base | s1-v1 |
|---|---|---|
| 与 teacher 的 top-1 一致率 | 0.397 | 0.369 |
| KL(teacher ‖ student) | 1.005 | **0.589** |
| Brier | 0.492 | **0.317** |
| 校准误差 ECE | 0.338 | **0.166** |
| 每请求延迟（Mac，6 道题） | — | 3.8 s |

- 训练：1,500 道题，0.2M 可训练参数（只有最后一个注意力块），约 39 分钟（1.4 s/题）。
- 分题型的一致率：best_move 0.50、best_switch 0.34、hard 0.38、stay 0.16。

**结论**：student 学到了 teacher 的置信度分布（KL −41%，ECE 减半），但没有学会更好的首选。Mac 上能训练的容量（0.8B 加单块 LoRA）不足以学会判别。晋升规则据此**拒绝晋升**（一致率没有超过 base），闭环的安全门按设计生效。要真正提升，需要在 GPU 上做全层 LoRA 和更大的 base（HANDOVER §7）。

### 3.3 级联部署（`s1_cascade_randbats.yaml`，6 局）

| 指标 | 数值 |
|---|---|
| student 被问到 / 门控放行 | 128 / **0**（顺序不一致 63、攻击还是换人不确定 58、置信度低 7） |
| student 与 teacher 首选一致 | 16 / 128 |
| 实际出手 | 全部由 teacher 决定（fast_teacher 39、teacher 也不确定 89） |
| 胜负 | 对 random 3–0，对 heuristic 1–2 |
| 每决策延迟 | 平均 5.5 s（Mac 上 student 4.9 s 加 teacher 0.4 s） |
| 记录 | 128 对 student/teacher 配对答案，可作为下一轮标签（`data/s1/experience/cascade_*.jsonl`） |

门控让一个弱 student 完全不影响出手，同时把每次 student 不确定的回合都变成一条 teacher 标签。这正是闭环要的行为。

### 3.4 深度融合单局实时性（`hymem_deep_latency_randbats.yaml`，Mac，Qwen3.5-4B on mlx，对 random，胜）

| 指标 | 纯 HyMem | 双系统 v1 | 双系统 v2（后台折叠） | **深度融合** |
|---|---|---|---|---|
| 决策数 | 49 | 26 | 37 | 18 |
| 每决策延迟 mean / p50 (s) | 139.5 / 141.7 | 42.1 / 0.5 | 61.2 / 53.2 | **42.0 / 34.3** |
| p90 / max (s) | 258.7 / 293.5 | 120.8 / 122.2 | 121.3 / 123.2 | 109.8 / 120.7 |
| ≤ 10 s 的决策 | 0% | 65% | 46% | 44%（快路径 8 次，平均 0.84 s） |
| **升级后在 120 s 内完成** | — | 0 / 9 | 3 / 20 | **9 / 10**（平均 69.8 s） |
| 每个慢回合的 LLM 调用 | 7.9（全部回合） | 6–10 | 2–10 | **1–3**（executor 的 LLM 调用为 0；System One 规划了 26 次工具调用） |
| 折叠 | 同步，约 112 s/次 | 无法完成 | 后台，约 216 s/次 | **后台，约 0.5 s/次**（typed_light，不调用 LLM） |
| 步骤控制 | — | — | — | 直接回答 2 次，深思 8 次 |
| 天梯计时器回放：判负的回合 | 第 1 回合 | 第 4 回合 | 第 3 回合 | 第 6 回合 |

**结论**：
- 内部融合删掉了慢路径里 executor 的多轮 LLM 调用和折叠的生成式抽取。HyMem 从"几乎从来来不及"变成"90% 在上限内完成"。
- 剩下的耗时就是 planner 本身：Mac 上每次约 30 s，1–3 次。
- 天梯计时器仍会在第 6 回合耗尽时间库，要靠 GPU 推理（planner 每次几秒），或者按计时器动态设上限。

## 4. 结论

1. **融合方式已从"在边界接入"变为"System 1 进入 HyMem 的机制内部"**：决定谁来决策、驱动 HyMem 内部的选择题、整理记忆并反馈。每一项都可单独开关，便于消融。
2. **学习闭环已经完整跑通一轮**（采集 → 数据集 → 训练 → 评估 → 晋升判定 → 服务 → 级联部署 → 再采集）：
   - 本机 student 学到了 teacher 的置信度分布（KL −41%，ECE 减半），但首选没有变好；
   - 晋升门正确地拒绝了它；
   - 级联门控让弱 student 不影响出手，同时自动收集 teacher 标签。
3. **换模型不改代码**：Jev、本地 student（mlx/HF）讲同一协议。服务器上训练出更强的 student 后，只需 `student_url` 或 `TYPESAFE_BASE_URL` 指向它，就能逐轮替换、逐轮进化。
4. 服务器上的工作见 `HANDOVER.md` §7：GPU 全层 LoRA、更大 base、消融、无 teacher 的自我进化。
