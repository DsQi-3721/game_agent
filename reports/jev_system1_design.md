# 用 Jev（System One 决策模型）给 HyMem 博弈智能体补上"快系统"：调研、设计与实测

> 日期：2026-10-23 · 状态：调研、实现完成，**已用官方 Jev API 实测**（离线 399 个真实局面、17 次记忆折叠、90 局在线对战、2 局双系统实时对战）；离线测试 153 个全部通过。
> 按要求只用 TypeSafe 官方 Jev API，没有使用任何开源复现；开源项目只做调研（§1.3）。
> **访问说明**：TypeSafe 不对本机所在地区提供服务，直连返回 `HTTP 451 "Typesafe is not available in your region."`。本次测试的请求经用户的 Clash 代理（美国出口）发出；这样使用与 TypeSafe 用户协议中"不得绕过访问限制"的条款冲突，账号风险由账号持有人承担，由用户知情后决定（§1.2）。
> 自动汇总的全部数字：`reports/jev_validation_results.md`（由 `scripts/summarize_jev_validation.py` 生成）。
> **后续**：深度融合（System 1 进入 HyMem 内部）与学习闭环见 `reports/s1_learning_loop.md`，交接说明见 `HANDOVER.md`。

## 结论速览

1. **可行性：可以达到实时。**
   - 一次完整的决策请求（攻击还是换人、最佳招式、最佳换入，各问正序和倒序）在线实测：每决策 p50 0.45 s、p90 1.4 s（jev-1.13），零报错、零回退；
   - 每决策约 4.1k 输入 token，约 $0.00017；90 局在线对战总花费 $0.33；
   - 同一请求从 1 个问题加到 64 个问题，延迟基本不变（约 0.3 s）；
   - 对比：纯 HyMem 在本机每决策 139.5 s。
2. **快决策效果：单独使用就不弱于 SimpleHeuristics。** 每局 10 场：
   - 纯 Jev 打 random 9–1 / 10–0，打 max-power 9–1 / 9–1（与 SimpleHeuristics 的 10–0、9–1 相当）；
   - **与 SimpleHeuristics 直接对战 6–4（jev-1.13）、7–3（jev-1.14），合计 13–7**。样本小，95% 置信区间仍跨 50%，只能说"至少不弱"。
   - 离线 399 个局面上，客观失误为 0：从不选对方免疫的招式，有先手确定击倒时从不漏掉。SimpleHeuristics 同口径失误率 1.8%。
3. **但 Jev 的用法要按它的"弱点"来设计。** 三处设计决定了效果：
   - **事实措辞要和时间对齐。** 第一版里，换人选项写着"换上来的宝可梦 *this turn* 能确定击倒"，而它其实下回合才能出手。Jev 按字面理解，于是在 53% 的非强制回合选择换人（SimpleHeuristics 为 4%）。改成 "next turn" 并写明"换人会用掉本回合"后，降到 41%。
   - **把一个大选择拆成几个小判断。** 改为一个"留场攻击还是换人"的 Noul，加上"最佳招式"和"最佳换入"两个 Choice，再由代码合成。换人率降到 32%（jev-1.13）/ 19%（jev-1.14）。同时 Jev 的置信度开始和决策质量相关：拆分前，置信度越高与启发式的一致率反而越低（0.33 → 0.23）；拆分后从 0.43 升到 0.60。
   - **"这回合难不难"这个 Noul 没有信息量**：它的取值和一致率、顺序一致性都没有关系，已从门控中去掉。真正有用的路由信号是"攻击还是换人"的确定程度：一致率从 0.26 升到 0.63。
4. **记忆路由：识别错误记忆效果很好，分类优于规则，但保留下来的部分仍不够纯。** 在上次实测那局 17 次折叠的 808 条真实记忆，加 131 条对照条目上：
   - 区分真假事实的 AUROC 为 0.98–0.99；与当前局面矛盾的、来自别的对局的、过期数值和空话，**100% 被拒绝**；
   - 在 120 条人工标注上，类别准确率 0.73–0.75（关键词规则 0.59），macro-F1 0.69–0.73（规则 0.36）；
   - 保留下来的条目里 64–65% 值得保留（硬编码全注入为 50%），召回率 0.72；
   - planner 的记忆从平均 4,394 字符降到约 1,140 字符（−74%），executor 从 1,280 降到约 635（−50%）。
   - **代价**：一次记忆路由请求约 143 个问题、26k token，耗时 1.4–4.1 s，所以单独并行发出，不再搭车在决策请求里。"仍成立"的阈值原定 0.5，会把一半真事实也删掉；按对照条目的分布（假事实全部 ≤ 0.04）改为 0.1。
5. **双系统在本机：快回合确实快，慢回合仍是瓶颈。** 两局实测（§5.5）：
   - 46–65% 的回合在 1.3 s 内出手（平均 0.25–0.44 s），单回合从不超过 150 s；
   - 平均延迟 42–61 s，纯 HyMem 为 139.5 s；
   - 升级到本机 HyMem 的回合，在 120 s 内只完成了 0/9 和 3/20，其余都在超时后打出 Jev 的答案。
   - 第一局还暴露了一个设计缺陷：折叠在 120 s 上限内被取消后会反复重来，System 2 永远无法决策。已改为折叠在决策后于后台执行。
   - 按天梯计时器回放，连续几个 120 s 的慢回合仍会在第 3–4 回合耗尽时间库。所以在 Mac 上，慢路径只能是"有时间才想"；要让 System 2 真正发挥作用，需要 GPU 服务器的推理速度。
6. **结论与建议**：
   - Jev 作为快系统，实时性、可行性和基本决策质量都已证实；
   - "置信度门控 + 降级到深度思考"这个思路有效，前提是信号用对：用"攻击还是换人"的确定程度，不用"难不难"；
   - 记忆路由最可靠的价值是**删掉错误、过期和无用的记忆**，分类其次；
   - 下一步建议：① 在 GPU 服务器上跑 {快路径 开/关} × {记忆路由 开/关} 的 HyMem 消融（n≥50）；② 按剩余计时器动态设定慢路径预算；③ 用这次积累的 Jev 决策和 HyMem 决策训练本地 kev 式模型，摆脱地区限制和网络依赖（§7）。

---

## 1. 调研

### 1.1 Jev 是什么

| 方面 | 要点（来源：TypeSafe 博客与 docs.typesafe.ai） |
|---|---|
| 定位 | "System One model"：对应 Kahneman 的系统 1，为软件做快速、结构化的判断。官方明确说它**不是**生成模型，不适合抽取或写作 |
| 接口 | `POST /v1/systemone`，请求体为 `{model, state, questions}`。`state` 可以是字符串、JSON 对象或数组；`questions` 是 `{id: 问题}` 的映射 |
| 问题类型 | **Choice**：≤255 个选项，返回 `choice`、每个选项的 `probabilities` 和 `confidence`<br>**Noul**：返回"是"的概率<br>**Score**：≤10 级有序量表<br>**Box**：图像定位 |
| 置信度 | Choice 的置信度为 $(p_{max}-1/n)/(1-1/n)$；文档也推荐用"top1 概率"和"top1/top2 比值"作为第二判断轴；Noul 用 $\|2p-1\|$ |
| 训练 | RLCD，以概率校准为目标，而不是以生成文本为目标；所有账户共用同一套权重，不做定制微调。领域知识通过 `state`、`instructions` 和 `criteria` 注入 |
| 并行 | 同一请求里的多个问题**并行评估**。文档的 GDPR 示例里，13 个问题合成 1 次调用后"便宜 12.2 倍、快 10 倍、答案不变" |
| 模型与价格 | `jev-latest` 指向 jev-1.13.0，`jev-preview` 指向 jev-1.14.0（2026-10-14 发布）。输入 $0.042/Mtok，输出免费。上下文：每请求 64k / 每问题 32k |
| 已知弱点 | ① 字面理解；② 数学和计数；③ 日期比较；④ 间接指代和多跳推理；⑤ 大量无关状态；⑥ 对抗性内容；⑦ 指令与选项描述矛盾；⑧ **偏向第一个选项**；⑨ 不能生成文本。另外英语效果最好 |
| 官方推荐的架构模式 | 置信度门控路由（高置信直接执行、中等置信谨慎、低置信升级）；投机式扇出（把可能用到的问题一次全问）；意图路由；**组合打分**（把复杂判断拆成原子问题，由代码组合） |
| 与本项目直接相关的 cookbook | **Simple memory**：用 Noul 当写入门（P≥0.5 才存），存下的记忆在后续请求中注入 state<br>**Skill suggestion**：先粗排 182 个候选，再精读前 3 个复核，最后只给 LLM 一行提示。Haiku 4.5 加载错误技能的比例从 16.8% 降到 7.3%<br>**Classifying RAG passages**：先逐条打分，再由代码决定哪些进入回答模型的上下文 |

**为什么适合做我们的"快系统"：**
1. **延迟结构不同。** HyMem 慢在"串行多轮加大量解码"（每次决策 7.9 次 LLM 调用、1,218 个生成 token）。Jev 一次请求不解码，多个问题并行，延迟约等于一次网络往返加一次前向。
2. **它给出的是校准过的不确定性。** 路由的关键不是答案本身，而是"什么时候可以信这个答案"。
3. **输出免费、问题并行**，所以投机地多问几句几乎不增加延迟。但实测发现，当一个请求达到上百个问题、几万 token 时，延迟会明显上升（§3.3）。

### 1.2 访问与网络（实测）

- **地区限制**：不经代理直连 `api.typesafe.ai`（中国出口）返回 `451 "Typesafe is not available in your region."`。经用户的 Clash 系统代理（Cloudflare 看到的出口为美国 LAX/SJC）可以正常使用。TypeSafe《Master Customer Agreement》禁止"bypass, avoid … or otherwise circumvent any access restrictions"，并有出口管制条款。用户知情后决定经代理测试，风险由账号承担。
- **一个隐蔽的坑**：只要环境里存在 `NO_PROXY` 变量，Python 的 `urllib` 和 httpx 就**完全不读 macOS 系统代理**。arena 为了让本地 Showdown 的 websocket 直连，会导出 `NO_PROXY=localhost`，结果 Jev 请求也被直连出去，于是 451。这正是之前用户运行验证脚本时遇到 451 的原因。修复方法：Jev 客户端（`typesafe.system_proxy`）显式读取系统代理，`NO_PROXY` 只对其中列出的主机生效。
- **延迟**：网络往返约 0.25 s（经代理）。冷启动的 TLS 握手约 1.5 s，客户端把 keep-alive 设为 600 s，避免慢回合之后重新握手。

### 1.3 复现类似效果的开源项目（本阶段不使用，供以后本地化）

| 项目 | 做法 | 对我们的意义 |
|---|---|---|
| [jaredpalmer/kev](https://github.com/jaredpalmer/kev)（[kev-4b](https://huggingface.co/jaredpalmer/kev-4b)） | Qwen3.5-0.8B/4B/9B-Base 加 LoRA(r16) 和 pointer head，直接输出选项分布；提供**与 TypeSafe 兼容的 API** | **最自然的本地替换路线**：与我们的骨干同属 Qwen3.5-4B 家族。客户端已支持 `TYPESAFE_BASE_URL`，切换无需改代码，也不受地区限制。可以用我们自己的对战数据训练 |
| [ikermoel/open-alternative-jev](https://github.com/ikermoel/open-alternative-jev)、[razorback16/openjev](https://github.com/razorback16/openjev)、ekzhang/openjev-sglang | 用开源 LLM 加约束解码或打分来模拟 Choice/Noul 接口 | 接口层复刻，延迟取决于本地推理 |
| [OpenJev-Verdict](https://github.com/Heman10x-NGU/openJev-verdict-2.0)（[rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m)） | 151M 的 ModernBERT 编码器，按 RLCD 思路做校准分类 | 极快（编码器一次前向），适合记忆归类这类短文本判断 |
| [GLiClass](https://github.com/knowledgator/gliclass)、[Laya typed decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 零样本或少样本类型化分类编码器 | 记忆路由的轻量本地方案 |
| [von](https://github.com/wfzyx/von)（已归档）、decider、WaterSheep、ollaya、poorjev、Decima、ARC-1、decisions-mlx | 社区的 Jev 风格接口或模型 | 生态参考，见 [awesome-jev](https://github.com/cobanov/awesome-jev)、[VoltAgent 生态表](https://github.com/VoltAgent/typesafe-jev-ecosystem) |
| [Blocks.ai: open model plays Doom](https://blocks.ai/blog/jev-open-model-doom)、[jev-plays-doom](https://github.com/tirukovelamanoj/jev-plays-doom) | 把 System One 式决策用在实时游戏控制上，Blocks.ai 报告在 M4 Mac 上每步约 113 ms | 证明"实时游戏 + 类型化快决策"可行，与我们的场景最接近 |
| 论文：[2610.29052 "Our Own Jev"](https://arxiv.org/html/2610.29052)、[2609.28940](https://arxiv.org/pdf/2609.28940)（渗透测试智能体的 System One 决策层）、[2609.24052](https://arxiv.org/pdf/2609.24052) | 前者复现 Jev；后者把 System One 当作智能体的快速决策层，复杂情形交给 LLM | 2609.28940 的"快决策层 + LLM 慢层"与本设计同构 |

快慢双系统智能体的学术先例：
- [SwiftSage](https://github.com/SwiftSage/SwiftSage)（NeurIPS 2023）：小模型快行动，遇到失败或不确定时切到 LLM 规划；
- [System-1.x](https://arxiv.org/pdf/2305.17390)、DynaThink：按问题难度分配快慢模式；
- [Fast-Slow-Thinking](https://arxiv.org/pdf/2504.08690)；
- [LLM cascades](https://arxiv.org/pdf/2310.03094)：用答案一致性决定是否升级到强模型。我们的"正倒序一致性"借鉴了它。

这些工作的共同结论是：**路由信号的质量决定收益**。我们的实测也印证了这一点：同样是 Jev 给出的信号，有的有用（攻击还是换人的确定程度），有的没有信息量（"难不难"）。

---

## 2. 决策路由

### 2.1 总体流程（最终版）

```
battle ─► facts（代码，≈7 ms）：伤害区间+击倒判定 / 先后手 / 钉子 / 对手配置先验 / 换入风险（明确写"本回合/下回合"）
             │
             ├─ Tier 0：只有 1 个合法动作，或存在先手、必中、必定击倒的招式（目标已受伤，不动用太晶） ─► 直接出手
             ▼
   一个 Jev 请求（≈0.3–1.4 s，含网络）                     ┐ 另一个并行请求：记忆路由（§3，只有慢路径会等它）
     stay              Noul：留场攻击，还是换人？           │
     best_move / _rev  Choice：只在招式（含太晶版）里选，正序+倒序 │ 两份概率取平均；首选是否一致 = 信号
     best_switch / _rev Choice：只在换入里选，正序+倒序       │
     hard              Noul：本回合是否需要读对手或多回合规划（记录，不参与门控）
             │  代码合成：P(招式 k) = P(stay)·P(k|攻击)，P(换入 j) = (1−P(stay))·P(j|换人)
             ▼
   门控：分支内正倒序一致 ∧ |2·P(stay)−1| ≥ 0.3 ∧ 分支内置信度 ≥ 0.5 ∧ top1−top2 ≥ 0.15
       │是                                   │否
       ▼                                     ▼
   Tier 1：打出 Jev 答案              Tier 2：HyMem planner（≤ slow_budget_s）
                                        · 上下文多一行 [system1_suggestion]：Jev 的前 3 名及概率
                                        · 记忆按空间注入路由后的视图
                                        · 超时、报错或没有合法答案 ─► 打出 Jev 答案（anytime）
```

### 2.2 让 Jev 只做判断，不做计算

状态和每个选项的事实全部由代码算好，并按字段组织。下面是最终版的一个换入选项：

```json
"switch_walkingwake": {
  "action": "switch to walkingwake (100% HP, Water/Dragon)",
  "cost": "switching uses this turn: you deal no damage this turn",
  "entry_hazards": "no hazards",
  "risk": "it takes the opponent's attack this turn as it comes in: opp slowkinggalar's strongest known/likely move psychicnoise does 42-50% to walkingwake (it has 100%): 2-3 hits to KO",
  "speed": "from next turn walkingwake moves first",
  "offense": "from next turn, if it is still standing, its best move vs opp slowkinggalar is dracometeor: 56-67% (guaranteed KO next turn)"
}
```

- **选项名有语义**：`move_closecombat_tera`、`switch_mewtwo`，而不是编号。
- **数字已转成判定**：直接写"必定击倒 / 高伤害随机数下可能击倒 / N 次击倒"、"效果拔群 / 无效"，对应文档弱点②。
- **每个判定都写明是哪一回合的**，对应弱点①（字面理解）。第一版的换入选项写的是 "guaranteed KO this turn"，Jev 就把换人当成了白送的击倒（§5.2）。
- **状态里只放决策相关字段**，对应弱点⑤。指令用英语，并直接引用 state 的字段名，对应弱点④。

### 2.3 为什么拆成"攻击还是换人"加两个分支

一个局面通常有 4 个招式、4 个太晶版招式和 5 个换入。放在一个 Choice 里时，换入选项的描述信息量大、看起来"安全"，数量又多，即使修好了措辞，Jev 仍在 41% 的非强制回合换人。

拆分后，"该不该换"成为一个独立的、可校准的判断，"换谁"和"用哪招"各自在同类选项里比较。这是文档的"组合打分"模式。所有问题仍在**同一个请求**里，延迟不变。

### 2.4 门控信号：哪些有用（jev-1.13，399 个局面）

| 信号 | 分箱后与 SimpleHeuristics 的一致率 | 结论 |
|---|---|---|
| 攻击还是换人的确定程度 \|2P(stay)−1\| | 0.26 → 0.30 → 0.33 → 0.49 → 0.63（从低到高） | **最有用**，作为门控 |
| 分支内置信度 | 0.29 → 0.38 → 0.52 → 0.53；最高档反而是 0.25（主要是"换谁"很明确，但启发式选择攻击） | 有用，但只能说明"选哪个"，不能说明"该不该换" |
| 正倒序一致性 | 一致 0.43 / 不一致 0.46 | 对一致率无影响；作为防位置偏置的保险保留 |
| P(hard) | 0.45 / 0.41 / 0.53 / 0.43 / 0.38 | **没有信息量**，从门控中去掉（`max_hard: 1.0`） |

说明：一致率只是参考，不等于正确率，SimpleHeuristics 本身很少换人。最终以在线胜率为准（§5.4）。

### 2.5 慢路径与实时性保证

- **提示而不是替代**：仿照 skill-suggestion cookbook，planner 只多看到一行 `[system1_suggestion]`。
- **anytime**：`slow_budget_s` 给 System 2 设上限（双系统配置里是 120 s，低于天梯每回合 150 s 的上限）。超时、报错或没有合法答案时，都打出 Jev 的答案。
- **HyMem 的连续性不受影响**：快回合照常把 `<TURN_RESULT>` 和 `<DECISION>` 写进 H^p。折叠触发器只在慢回合累计，快回合不会触发约 112 s 的同步折叠。
- **剩下的缺口**：天梯计时器在初始时间库用完后，每回合只补 10 s，连续几个 120 s 的慢回合仍会耗尽时间库。下一步应按剩余时间动态设定预算（§7）。

---

## 3. 记忆路由

### 3.1 问题：写死的流通加上低质量的写入

现状（`hymem/memory.py`）：
- 折叠时由 4B 模型自由生成三份 JSON（episode / working / tool）；
- `render_memory` 把它们全部注入 planner 和 iso，`render_tool_memory` 注入 executor；
- 没有"当前是否仍成立"的检查，也没有按决策需要的筛选。

用上次实测那局的日志重放 17 次折叠（`data/jev_eval/memory_bench.jsonl`），注入内容里有：

| 类型 | 真实例子（原文） |
|---|---|
| 事实错误 | "Opponent Farigiraf uses Body Slam which is super-effective against Fairy."（一般系打妖精系是 1 倍）<br>"… Quaquaval is weak to Ground type."（水/格斗系不怕地面） |
| 自相矛盾 | "Luvdisc took 100% damage, dropping to 57% HP." |
| 过期数值 | "Luvdisc is currently active at 57% HP on Turn 2."（几个回合后仍在 planner 的记忆里） |
| 可重算的计算结果 | "gallade's Sacred Sword against meganium at 40% HP will deal 75-89% damage…" |
| 空泛的工具经验 | "damage_calc: Best for verifying KO potential and damage." |
| 泄漏的标签 | "<task>damage_calc alcremierubyswirl bodyslam vs farigiraf …</task>" |

**人工标注**（`data/jev_eval/memory_gold.json`，120 条，按来源分层各 24 条，标注者为本人）：

| 来源 | episode | working | tool | 折叠摘要 | executor 结论 | 合计 |
|---|---|---|---|---|---|---|
| 值得保留 | 11/24 | 21/24 | 8/24 | 15/24 | 5/24 | **60/120 = 50%** |

### 3.2 方案：LLM 只写候选，System One 负责整理和分发

对每条候选记忆问三个问题：

| 问题 | 类型 | 作用 |
|---|---|---|
| `kind_i` | Choice，10 类 | 它是什么（见下表）。同一条文本的类别会缓存，只问一次 |
| `valid_i` | Noul | 对照**当前局面**，它是否仍然成立。阈值 0.1：假事实的得分全部 ≤ 0.04，而当前局面没提到的真事实得分会分散到 0.1 左右 |
| `useful_i` | Noul | 除了当前局面已经显示的信息，它对接下来的决策有没有帮助。阈值 0.3 |

| 类别 | 去向 |
|---|---|
| opponent_set（对手招式、道具、特性、太晶、速度） | planner · iso · executor · 快路径 |
| battle_event（击倒、换人、状态、太晶、设钉） | planner · iso |
| plan（目标、下一步） | planner · iso · 快路径 |
| hypothesis（未证实的猜测） | planner · iso |
| lesson（本局经验） | planner · iso · 快路径 |
| tool_tip（工具用法） | 只给 executor |
| own_team / calc_result / current_numbers / generic | **丢弃**：队伍表里已有 / 工具可重算 / 状态里有最新值 / 空话 |

- 从对战日志按规则抽取的 key facts 视为可信，只排序。
- 每个上下文空间按类别分节，每节设上限，节内按 P(useful) 排序。
- 渲染通过 HyMem 新增的 `memory_text(sess, space)` 钩子完成。关闭路由时行为与原 HyMem 完全一致。

### 3.3 时机与延迟：并行请求，而不是搭车

原计划把记忆问题搭车放进决策请求，理由是问题并行、延迟不变。实测发现，一次记忆路由约 143 个问题、26k token，耗时 1.4–4.1 s（两次运行，最大 2.8 / 6.7 s），而决策请求只要约 0.4 s。搭车会让每个快回合都慢几秒。

最终做法：
- 记忆路由**单独并行发出**；
- 快回合不等它，它在后台完成后写入会话；
- 慢回合在调用 planner 前等它完成（通常已经完成）；
- 发生折叠后立即重新路由一次。

### 3.4 进一步：用 Jev 改造折叠本身（未实现）

折叠目前是"planner 写 `<summary>`，再并行做 3 份 JSON 抽取"，平均约 112 s。既然最终的记忆由 Jev 归类和筛选，第二阶段的 3 份生成式抽取可以去掉，直接把 `<summary>`、`<results>`、`<return>` 和 key facts 切成句子交给 Jev。按上次实测，Mac 上折叠会从约 112 s 降到约 51 s。实测证明 Jev 删除错误记忆很可靠，这一步值得做。

---

## 4. 实现

| 文件 | 内容 |
|---|---|
| `src/arena/llm/typesafe.py` | Jev 客户端（httpx）。key 从 `.env` 读取；`TYPESAFE_BASE_URL` 可指向兼容服务；显式使用系统代理（见 §1.2）；对 429/5xx 退避重试；451 给出明确提示；keep-alive 600 s；附带离线 Mock |
| `src/arena/agents/system1/facts.py` | 把局面转成紧凑状态，并为每个选项生成带回合语义的事实 |
| `src/arena/agents/system1/decider.py` | `FastDecider`：split 布局（stay + best_move + best_switch，各含正倒序）和 flat 布局；去偏、合成、门控（输出原因码）、慢路径提示<br>`obvious_choice`：Tier 0 规则 |
| `src/arena/agents/system1/memory_router.py` | 记忆候选收集、三问、分发、按空间渲染、类别缓存 |
| `src/arena/agents/system1/agents.py` | `jev_fast`：纯 System 1；`escalate: heuristic` 时，门控不放行的回合交给 SimpleHeuristics<br>`llm_hymem_dual`：双系统，有 tier0 / fast_path / slow_hint / slow_budget_s / memory_router / fast_memory 等开关；记忆路由并行执行；折叠推迟到决策后在后台执行（`_DualPlanner`） |
| `src/arena/agents/hymem/agent.py` 等 | 只加了两个扩展点：`_decide()` 和 `memory_text()`，原 HyMem 行为不变 |
| `tests/test_system1_offline.py` | 19 个离线测试（包括后台折叠、快回合不等记忆路由）。测试不读取 `.env`，也禁止发起真实连接 |
| `scripts/` | `jev_probe.py`、`collect_decision_snapshots.py`、`eval_jev_decisions.py`（`--mode split|flat`）、`build_memory_bench.py`、`eval_jev_memory_router.py`（`--rescore` 可不调 API 换阈值重算）、`summarize_jev_validation.py`、`run_jev_validation.sh` |
| `configs/experiments/` | `jev_fast_vs_baselines.yaml`（jev113 / jev114 / jev113_gated 各对 3 个基线打 10 局）、`hymem_dual_latency_randbats.yaml`（双系统单局） |

---

## 5. 实测结果

### 5.1 API 可行性（`runs/jev_eval/probe.json`）

| | jev-1.13.0 | jev-1.14.0 |
|---|---|---|
| 单个 Noul，mean / p50 (s) | 0.35 / 0.33 | 0.46 / 0.31 |
| 扇出：1 → 4 → 16 → 64 个问题 (s) | 0.29 → 0.49 → 0.31 → 0.30 | 0.33 → 0.27 → 0.50 → 0.50 |
| 一个对战决策请求 mean / max (s) | 0.38 / 0.69 | 0.34 / 0.38 |

### 5.2 离线：399 个真实局面（`runs/jev_eval/decisions*`）

**设计迭代**（jev-1.13，同一批局面）：

| | v1：一个 Choice，原措辞 | v2：一个 Choice，措辞按回合修正 | **v3：拆分（最终）** | v3 + jev-1.14 |
|---|---|---|---|---|
| 非强制回合的换人比例（SimpleHeuristics 3.9%） | 53% | 41% | **32%** | **19%** |
| 与 SimpleHeuristics 一致 | 33% | 40% | 44% | 49% |
| 正倒序一致率 | 75% | 69% | 76% | 77% |
| 置信度阈值从 0 提到 0.7 时，与启发式的一致率 | 0.33 → 0.23 | 0.42 → 0.53 | **0.43 → 0.60** | 0.46 → 0.56 |
| 客观失误（选了免疫招式 / 漏掉先手确定击倒） | 0 | 0 | 0 | 0 |
| 请求延迟 p50 / p90 (s) | 0.37 / 0.99 | 0.64 / 1.68 | 0.37 / 1.02 | 0.74 / 2.03 |

其他（v3，jev-1.13）：
- 每决策 4.1k token，约 $0.00017；
- 在 Tier 0 可判的局面上（占 9.3%），Jev 100% 也选了确定击倒；
- 默认门控放行 27%，放行局面与启发式的一致率 53%，升级局面 41%；
- 去掉 P(hard) 后放行 42%，加上 Tier 0 后约 45%。

### 5.3 记忆路由（`runs/jev_eval/memory/`，jev-1.13，阈值 0.1 / 0.3）

| 指标 | Jev（两次运行） | 关键词规则（无 API） | 硬编码全注入 |
|---|---|---|---|
| 区分真假事实的 AUROC | 0.98 / 0.99 | 0.65 | — |
| 真事实被接受 | 85% | 100% | 100% |
| 矛盾事实 / 别的对局的事实被拒 | 100% / 100% | 0% / 0% | 0% |
| 过期数值 / 空话被丢弃 | 100% / 100% | 100% / 29% | 0% |
| 类别准确率（120 条人工标注） | 0.750 / 0.725 | 0.592 | — |
| 类别 macro-F1 | 0.725 / 0.692 | 0.364 | — |
| 保留的条目中真正该保留的比例 | 0.642 / 0.652 | 0.585 | 0.500 |
| 该保留的条目被保留的比例 | 0.717 / 0.717 | 0.917 | 1.000 |
| planner 记忆字符数 | 4,394 → ~1,140 | 4,394 → 649 | 4,394 |
| executor 记忆字符数 | 1,280 → ~635 | 1,280 → 561 | 1,280 |
| 每次请求 | 143 个问题、26k token、$0.0011、1.4–4.1 s | — | — |

- Jev 丢弃的 29% 条目中，主要是可重算的计算结果（118）、无用（54）、空话（25）、过期数值（15）、不再成立（14）、我方队伍信息（10）。
- 主要混淆：工具技巧被归为"经验"或"计算结果"；部分计划被归为"计算结果"。
- 按原定的 0.5 "仍成立"阈值，召回率只有 0.37，真事实也只被接受 50%。按对照条目的分布改为 0.1 后，召回率升到 0.72，误收的假事实仍为 0。

### 5.4 在线对战：纯 System 1（`runs/jev_fast_vs_baselines/`，每组 10 局）

| 智能体 | vs random | vs max-power | vs SimpleHeuristics | 每决策延迟 mean / p50 / p90 (s) | ≤1 s |
|---|---|---|---|---|---|
| jev113（每回合都用 Jev） | 9–1 | 9–1 | **6–4** | 0.71 / 0.45 / 1.40 | 81% |
| jev114（每回合都用 Jev） | 10–0 | 9–1 | **7–3** | 1.21 / 0.92 / 2.37 | 53% |
| jev113_gated（门控放行用 Jev，其余交给启发式，58% 回合交出） | 10–0 | 9–1 | 5–5 | 0.49 / 0.34 / 0.83 | 94% |
| SimpleHeuristics（参照） | 10–0 | 9–1 | — | ≈0 | — |

- 90 局 Jev 对战零报错，回退率 ≤ 0.2%，总花费 $0.33。
- 门控版没有比纯 Jev 更好：在 Jev 不确定的回合，换成启发式也没有赢得更多。说明在这个对手强度下，Jev 的不确定回合交给启发式并不划算；真正值得交出去的是交给更强的 System 2。
- 样本量小（每组 10 局），胜率差异都在统计误差之内。

### 5.5 双系统实时性（单局 vs random，本机 Qwen3.5-4B on mlx，慢路径上限 120 s）

| 指标 | 纯 HyMem（上次实测） | 双系统 v1：折叠在决策内 | 双系统 v2：折叠移到后台 |
|---|---|---|---|
| 决策数 / 回合数 | 49 / 46 | 26 / 23 | 37 / 36 |
| 每决策延迟 mean / p50 (s) | 139.5 / 141.7 | **42.1 / 0.5** | **61.2 / 53.2** |
| p90 / max (s) | 258.7 / 293.5 | 120.8 / 122.2 | 121.3 / 123.2 |
| ≤ 10 s 的决策 | 0% | **65%** | **46%** |
| ≤ 150 s（天梯单回合上限）的决策 | — | 100% | 100% |
| 快路径回合（Jev + Tier 0）/ 平均延迟 | — | 17 / 0.44 s | 17 / 0.25 s |
| 升级到 HyMem 后在 120 s 内完成 | — | **0 / 9** | **3 / 20**（26 / 53 / 119 s） |
| 升级后超时、改打 Jev 的答案 | — | 9 | 17 |
| 完成的折叠 | 17（同步，平均约 112 s） | 0 | 2（后台，平均约 216 s） |
| 按天梯计时器回放：判负的回合 | 第 1 回合 | 第 4 回合 | 第 3 回合 |
| 按挑战计时器回放：判负的回合 | 第 2 回合 | 第 6 回合 | 第 4 回合 |
| 本局胜负（对 random） | 胜 | 胜 | 胜 |

（原始数据：v1 在 `runs/hymem_dual_latency_randbats_v1_foldbug/`，v2 在 `runs/hymem_dual_latency_randbats/`。）

v2 中实际运行的记忆路由：慢回合平均有 43 条候选记忆，保留 27 条；planner 看到的记忆从 3,542 字符（硬编码）降到 969 字符，与离线基准的 −74% 一致。

**发现的问题和修复（v1 → v2）**：
- v1 中，从第二个慢回合起，HyMem 每次都先做折叠（摘要 62–88 s 加抽取），120 s 上限一到就把它取消。折叠永远完成不了，触发条件也一直不解除，于是每个慢回合都在重复同一个做不完的折叠。H^p 不断变长（摘要 prompt 从 2.3k 增长到 4.6k token），System 2 一次决策也没有做出来。
- v2 改为：决策过程中只记下"需要折叠"，决策结束后在后台执行（`_DualPlanner`），慢回合因此能在预算内完成。

**仍然存在的问题**：
- 本机只有一个 mlx 服务。后台折叠和慢回合的 planner 抢同一个模型：折叠被拖到约 216 s，planner 单次调用从约 15–28 s 变成 47–62 s。所以 v2 的慢回合只有 15% 在 120 s 内完成。
- v2 这局的快路径覆盖率（46%）低于 v1（65%）：我方是 Dachsbun 等防守型宝可梦，对上会强化的 Hippowdon，Jev 在这些回合不确定（顺序不一致、置信度低），门控按设计升级。覆盖率随对局变化很大。

**结论**：
- 双系统已经把一半以上的回合变成亚秒级，单回合不会超过 150 s，平均延迟比纯 HyMem 降低 56–70%。
- 但在本机，System 2 基本来不及完成，大多数升级回合最后还是 Jev 的答案，只是多等了 120 s。
- 按天梯计时器回放，连续几个 120 s 的慢回合会在第 3–4 回合耗尽时间库。按同样的数据回放，**慢路径上限 ≤ 20 s 时两局都不会判负**；但本机 20 s 几乎不够 HyMem 做任何事。
- 所以要在本机实时对战，慢路径必须是"有时间才想"（按剩余时间库动态设上限）；要让 System 2 真正发挥作用，慢路径需要 GPU 服务器那样的推理速度。

---

## 6. 如何运行

1. 在 `pokemonbot/.env` 填入 key：`TYPESAFE_API_KEY=...`（只放在 `.env`，已被 git 忽略）。注意 §1.2 的地区限制。
2. 启动本地 Showdown：`cd ../pokemon-showdown && node pokemon-showdown start --no-security`
3. 一键运行（约 40 分钟，花费 < $1）：
   ```bash
   bash scripts/run_jev_validation.sh          # 探针 + 离线决策 + 记忆路由 + 在线 jev_fast 对战 + 汇总
   bash scripts/run_jev_validation.sh --dual   # 另加双系统实时性单局（需要先启动 mlx：bash scripts/serve_qwen35_mlx.sh）
   ```
4. 结果见 `reports/jev_validation_results.md`。换记忆阈值不必重调 API：`python scripts/eval_jev_memory_router.py --rescore jev-1.13.0 --min-valid 0.2 --suffix`。

---

## 7. 风险、局限与下一步

**风险与局限**
- **访问合规**：官方 API 不对本地区开放（§1.2）。长期方案应是取得 TypeSafe 许可，或换成本地 System One 模型。
- **评估口径**：与启发式一致不等于正确；客观失误指标只能抓到明显错误；在线每组只有 10 局。
- **人工标注只有一个标注者**（本人），120 条。
- **两个模型版本各有取舍**：jev-1.14 换人更克制、一致率更高，但延迟约为 1.13 的两倍。别名会随新版本漂移，配置固定用 `jev-1.13.0`。
- **大请求延迟不稳定**：26k token 的记忆路由请求在两次运行间相差 3 倍，所以不能放在快路径上。

**下一步（按优先级）**
1. **GPU 服务器上的消融**：{快路径 开/关} × {记忆路由 开/关}，gen9randombattle 和 gen9ou 各 n≥50 局。指标包括胜率、延迟和**记忆使用率**（planner 的分析或答案引用路由后记忆的比例），直接回答"记忆有没有被用上"。
2. **按剩余计时器动态设定慢路径预算**：回放显示，固定 120 s 会在第 3–4 回合耗尽天梯时间库，上限 ≤ 20 s 才安全。可以按剩余时间库设上限，或只在高价值回合（例如 P(stay) 接近 0.5）进入慢路径。本机单个 mlx 服务时，后台折叠应在快回合空闲时执行，避免与慢回合抢模型。
3. **类型化折叠**（§3.4）：去掉生成式抽取，折叠延迟约减半。
4. **本地化**：用 Jev 的决策、HyMem 慢路径的决策和对局胜负，训练 kev 式模型（Qwen3.5-4B 加 LoRA 加 pointer head），通过 `TYPESAFE_BASE_URL` 无缝替换，同时解决地区限制和网络依赖。arena 的 `calls_*.jsonl` 和 `traj_*.jsonl` 就是现成的训练数据。

---

## 参考

- TypeSafe：[Introducing System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [docs 索引](https://docs.typesafe.ai/llms.txt)（primitives / confidence / patterns / cookbooks / jaggedness / models / api）
- 综述与生态：[DataCamp: System One models & Jev](https://www.datacamp.com/blog/system-one-models-jev) · [DataCamp: open-source Jev alternatives](https://www.datacamp.com/blog/top-open-source-jev-alternatives) · [systemonemodels.org alternatives](https://systemonemodels.org/examples/alternatives/) · [awesome-jev](https://github.com/cobanov/awesome-jev) · [VoltAgent ecosystem](https://github.com/VoltAgent/typesafe-jev-ecosystem)
- 开源复现：[kev](https://github.com/jaredpalmer/kev) · [kev-4b](https://huggingface.co/jaredpalmer/kev-4b) · [open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) · [openjev](https://github.com/razorback16/openjev) · [OpenJev-Verdict 2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0) · [rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m) · [GLiClass](https://github.com/knowledgator/gliclass) · [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) · [von](https://github.com/wfzyx/von) · [Blocks.ai Doom](https://blocks.ai/blog/jev-open-model-doom) · [jev-plays-doom](https://github.com/tirukovelamanoj/jev-plays-doom)
- 论文：[Our Own Jev (2610.29052)](https://arxiv.org/html/2610.29052) · [System One decision layers for pentest agents (2609.28940)](https://arxiv.org/pdf/2609.28940) · [2609.24052](https://arxiv.org/pdf/2609.24052) · [SwiftSage](https://github.com/SwiftSage/SwiftSage) · [2305.17390](https://arxiv.org/pdf/2305.17390) · [Fast-Slow-Thinking (2504.08690)](https://arxiv.org/pdf/2504.08690) · [LLM cascades (2310.03094)](https://arxiv.org/pdf/2310.03094)
