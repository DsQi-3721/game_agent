# Pokémon Showdown 使用与对战手册

> 本手册面向"想用 Pokémon Showdown（以下简称 **PS**）作为博弈智能体研究环境"的开发者。
> 每个章节末尾的 **来源** 会标注原始文档路径与官方/社区 URL，方便你跳回原文阅读完整内容。
>
> 本地仓库路径默认为 `/Users/hoi/project/game/pokemon-showdown/`。

---

## 0. 一句话说清楚 PS 是什么

PS 不是"一个软件"，而是**同一份代码同时提供 5 种东西**：

| 角色 | 用途 | 入口 |
|---|---|---|
| Web 站点 | 网页对战、聊天、Teambuilder、天梯 | `http://pokemonshowdown.com/` |
| Node.js 库 | 在你自己的 JS 程序中调用战斗模拟器与 Pokédex | `npm install pokemon-showdown` |
| 命令行工具 | 不写 JS 也能模拟对战、生成/校验队伍 | `./pokemon-showdown <cmd>` |
| Web API | 给官方客户端用的鉴权/聊天接口 | pokemon-showdown-client 仓库 |
| 游戏服务器 | 自建社区/私服 | `node pokemon-showdown start` |

PS 模拟了**第 1～9 世代**的**单打、双打、三打**全部规则。

**来源**：`README.md:14-39` ｜ https://github.com/smogon/pokemon-showdown

---

## 1. 文档地图（PS 没有单一手册，按需要跳读）

| 你想知道 | 读哪个文件 |
|---|---|
| PS 总览 | `README.md` |
| 怎么用 CLI | `COMMANDLINE.md` |
| 怎么把模拟器当 npm 包用 | `sim/README.md`、`sim/SIMULATOR.md` |
| **战斗 I/O 协议（写 bot 必读）** | `sim/SIM-PROTOCOL.md` |
| 队伍三种格式 | `sim/TEAMS.md` |
| Pokédex / 招式 / 道具数据 API | `sim/DEX.md` |
| 自定义/非标准数据如何过滤 | `sim/NONSTANDARD.md` |
| 客户端↔服务器 WebSocket 聊天/挑战协议 | `PROTOCOL.md` |
| 自建服务器 | `server/README.md` |
| 代码组织 | `ARCHITECTURE.md` |
| 写 bot 的社区 FAQ（聊天 + 战斗 bot 通用注意事项） | https://gist.github.com/Kaiepi/becc5d0ecd576f5e7733b57b4e3fa97e |

**来源**：`README.md:42-55`

---

## 2. 本地启动与基础使用

### 2.1 启动服务器（给 bot/客户端连接用）

```bash
cd /Users/hoi/project/game/pokemon-showdown
node pokemon-showdown start --no-security
# 默认监听 ws://localhost:8000/showdown/websocket
```

- `--no-security` 关闭账号验证 / 登录节流 / IP 检查 —— 这是 poke-env 等自动化客户端**必须开**的。等价做法是改 `config/config.js` 里的 `noipchecks=true; nothrottle=true`。
- 浏览器打开 `http://localhost:8000` 就是完整 PS 站点（含 Teambuilder、聊天、Ladder）。
- 端口可改：`node pokemon-showdown 9000`。

**来源**：`COMMANDLINE.md:22-32`、`server/README.md:9-65`

### 2.2 命令行工具（每条命令都是 stdin/stdout）

```bash
./pokemon-showdown start [--skip-build] [PORT]   # 启动服务器
./pokemon-showdown generate-team [FORMAT-ID [SEED]]  # 生成随机队（packed 格式）
./pokemon-showdown validate-team [FORMAT-ID]     # 校验队伍是否合规（exit code）
./pokemon-showdown simulate-battle               # 用 stdin 喂命令模拟一整场
./pokemon-showdown pack-team / json-team / export-team   # 队伍格式互转
./pokemon-showdown help
```

支持 Unix 管道，例：
```bash
./pokemon-showdown generate-team gen9randombattle | ./pokemon-showdown export-team
```

**来源**：`COMMANDLINE.md:17-138`

### 2.3 作为 Node 库（最高效，无网络/无进程开销）

```js
const Sim = require('pokemon-showdown');
const stream = new Sim.BattleStream();
(async () => { for await (const out of stream) console.log(out); })();
stream.write(`>start {"formatid":"gen7randombattle"}`);
stream.write(`>player p1 {"name":"Alice"}`);
stream.write(`>player p2 {"name":"Bob"}`);
```

`BattleStream` 是 `ObjectReadWriteStream`：写 `>` 开头的命令，读模拟器回来的事件字符串。和 `simulate-battle` 完全等价、共用同一个引擎。

**来源**：`sim/SIMULATOR.md`（全文，165 行）、`sim/README.md`

---

## 3. 支持的世代、对战类型、格式（Format / Tier）

### 3.1 世代与对战模式

- **世代**：`|gen|` 协议字段，`1` 到 `9`。Stadium 算对应世代；Let's Go 算 7；mod 格式算其基础世代。
- **对战模式**（`|gametype|`）：`singles`（单打）/ `doubles`（双打）/ `triples`（三打）/ `multi`（2v2 各带一只）/ `freeforall`（四人混战）。

**来源**：`sim/SIM-PROTOCOL.md:60-68`

### 3.2 格式（Format）= 世代 + Tier + 规则集

每场战斗都跑在一个**格式 ID**上，例如 `gen9ou`、`gen9randombattle`、`gen8doublesou`、`gen9vgc2024regg`。本地 `config/formats.ts` 一共定义了 **300+ 个格式**，覆盖：

| 世代前缀 | 例子 |
|---|---|
| `[Gen 1]` … `[Gen 9]` | 各世代主流格式 |
| `[Gen 9 Champions]` | 最新 Champions 子赛季 |
| `[Gen 8 BDSP]` | 闪钻闪珍珠 |
| `[Gen 8 DLC 1]` | 剑盾 DLC1 |
| `[Gen 7 Let's Go]` | 皮卡丘/伊布 |

每个世代下的常见**Tier / 玩法**（以 Gen 9 为例）：

- **Random Battle 系列**：`Random Battle`、`Unrated Random Battle`、`Free-For-All Random Battle`、`Random Battle (Blitz)`、`Multi Random Battle`、`Random Doubles Battle` —— **完全自动发队**，最适合做 bot 训练，因为去掉了 teambuilding 这层。
- **Smogon 竞技 Tier**：`Ubers` > `OU`（OverUsed，主流）> `UU`（UnderUsed）> `RU` > `NU` > `PU` > `ZU` > `LC`（Little Cup, 仅未进化的最低形态）。
- **特色规则**：`Monotype`（队伍同属性）、`Monocolor`、`1v1`、`2v2 Doubles`、`4v4 Doubles UU`、`Anything Goes`（几乎无禁用）、`CAP`（Create-A-Pokémon 同人）。
- **官方赛制 / VGC**：`VGC 2023 Reg C/D`、`VGC 2024 Reg G`、`VGC 2026 Reg F/I`、`BSS Reg I/J/M-A`、各种 `Bo3`。
- **双打**：`Doubles OU`、`Doubles Ubers`、`Doubles UU`、`Doubles LC`、`Doubles Custom Game`。
- **Draft / 自定义**：`Draft`、`6v6 Doubles Draft`、`Custom Game`（无任何限制，写 bot 测试用很方便）。

**来源**：`config/formats.ts`（331 条 `name:` 定义）、`sim/SIM-PROTOCOL.md:70-72`

> 完整格式列表也可在 PS 网页 Teambuilder 的 Format 下拉框里看到，或 Smogon Dex 的 Formats 页：https://www.smogon.com/dex/sh/formats/

### 3.3 规则条款（Clauses）

每个格式开局会用 `|rule|` 报告所有生效条款。常见有：

```
Species Clause: Limit one of each Pokémon
OHKO Clause: OHKO moves are banned
Moody Clause: Moody is banned
Evasion Abilities Clause: Evasion abilities are banned
Evasion Moves Clause: Evasion moves are banned
Endless Battle Clause: Forcing endless battles is banned
HP Percentage Mod: HP is shown in percentages
Sleep Clause: 一方同时只能让一只对方宝可梦陷入睡眠（不在协议示例里但 Smogon Tier 默认开）
```

**重要**：`HP Percentage Mod` 开启时，对手 HP 协议里以 `/100` 表示（百分比），关闭时是 `/48`。**你的智能体看到的就不是确切血量**。

**来源**：`sim/SIM-PROTOCOL.md:22-27, 263-265`

---

## 4. 一场对战的完整规则与流程

### 4.1 战斗初始化（开局协议）

```
|player|p1|Anonycat|60|1200
|player|p2|Anonybird|113|1300
|teamsize|p1|4
|teamsize|p2|5
|gametype|doubles
|gen|7
|tier|[Gen 7] Doubles Ubers
|rule|...（多条）
|clearpoke
|poke|p1|Pikachu, L59, F|item
|poke|p1|Kecleon, M|item
...
|teampreview     ← 看得见双方队伍但不知道道具/招式/努力值
|start
```

**信息不对称**：Team Preview 只展示种族、性别、等级、是否持物（不显示具体道具）；**性格、努力值、个体值、招式、确切 HP 永远看不见**对方的。这是 PS 博弈环境的核心约束。

**来源**：`sim/SIM-PROTOCOL.md:10-114`

### 4.2 回合结构（同时博弈，不是回合制）

1. 服务器发 `|request|REQUEST`（一个 JSON）告诉你**当前你能选什么**。
2. 你回 `>p1 CHOICE` 或 `>p2 CHOICE`。
3. **双方都提交后**，引擎按"优先级 → 速度"结算，发送一连串 `|move|`、`|-damage|`、`|-status|` 等事件。
4. 该回合还有强制换人时，引擎会再发一次 `|request|`（switch request）。
5. `|turn|N` 表示进入下一回合。
6. 终局 `|win|USER` 或 `|tie`。

**来源**：`sim/SIM-PROTOCOL.md:115-160, 619-657`

### 4.3 一回合内可以做的事（CHOICE 语法）

```
team 213456                  # Team Preview 阶段：定先后序
move Focus Blast             # 出招（招式名）
move 4                       # 出招（slot 1-4）
move thunderbolt 1           # 双打：指向对方 slot 1
move helpinghand -1          # 双打：指向己方 slot 1（负号=队友）
move <name> mega             # Mega 进化
move <name> zmove            # Z 招式
move <name> max              # 极巨化
move <name> terastallize     # 太晶化（Gen 9，命令实际写法见 PS console）
switch Pikachu / switch 2    # 换人
pass                         # 双打里某只已倒下，跳过
undo                         # 撤销已提交的选择（对手未提交时才有效）
default                      # 用引擎选的第一个合法决策（VGC 超时即用此）
```

双打用 `,` 分隔两只的指令，例：
```
move Thunderbolt 1 mega, move Helping Hand -1
```

若决策非法（被束缚/陷阱/Disable…），引擎回：
```
|error|[Invalid choice] ...
```
或附带一个新的 `|request|` 让你重选（已暴露的信息会写进 `request` 里）。

**来源**：`sim/SIM-PROTOCOL.md:610-705`

### 4.4 `|request|` JSON ——智能体的 observation

服务器每回合给你的"决策请求"长这样（节选）：

```json
{
  "active": [{
    "moves": [
      {"move":"Light Screen","id":"lightscreen","pp":48,"maxpp":48,
       "target":"allySide","disabled":false},
      {"move":"U-turn","id":"uturn","pp":32,"maxpp":32,
       "target":"normal","disabled":false}
    ]
  }],
  "side": {
    "name":"Zarel","id":"p2",
    "pokemon":[
      {"ident":"p2: Ledian","details":"Ledian, L83, M",
       "condition":"227/227","active":true,
       "stats":{"atk":106,"def":131,"spa":139,"spd":230,"spe":189},
       "moves":["lightscreen","uturn","knockoff","roost"],
       "baseAbility":"swarm","item":"leftovers","ability":"swarm"}
      // … 其余 5 只
    ]
  },
  "rqid": 3
}
```

字段含义：
- `active`：当前场上你的宝可梦能选的招式（PP、能否使用、目标类型）。
- `side.pokemon`：你**自己**6 只全部信息（确切 HP、努力值实际数值、招式列表、道具）。
- `rqid`：请求 ID。回复 `>p2 move 3|3` 时带上能防止"撤销 + 串号"。
- 还可能出现 `forceSwitch:[true]`（必须换人）、`teamPreview:true`、`trapped:true` 等字段。

> **设计要点**：bot 的状态空间 = 这个 `request` JSON + 自己维护的"对方信息推断"（从 `|switch|`、`|move|`、`|-damage|` 反推）。poke-env 帮你封装好了。

**来源**：`sim/SIM-PROTOCOL.md:707-893`

### 4.5 战斗中会发生什么（事件总览）

**主要动作（major actions）**：
- `|move|POKEMON|MOVE|TARGET` 使用招式
- `|switch|` / `|drag|` 换人（drag = 被吼叫/U-turn 强换）
- `|detailschange|` / `|-formechange|` 形态变更（Mega/原始回归/动力极巨）
- `|replace|` Illusion 解除（Zoroark 露馅）
- `|swap|` 位置交换
- `|cant|` 无法出招（麻痹、Disable…）
- `|faint|` 倒下

**次要动作（minor actions, 以 `|-` 开头）** —— 实际伤害、状态变化都在这里：
- 伤害与治疗：`|-damage|`、`|-heal|`、`|-sethp|`
- 异常状态：`|-status|`（slp/par/brn/psn/tox/frz）、`|-curestatus|`、`|-cureteam|`
- 能力变化：`|-boost|`、`|-unboost|`、`|-setboost|`、`|-clearboost|`、`|-invertboost|`、`|-swapboost|`、`|-copyboost|`
- 战斗场地：`|-weather|`（天气）、`|-fieldstart|`/`|-fieldend|`（Trick Room、Grassy Terrain 等场地）、`|-sidestart|`/`|-sideend|`（Tailwind、Stealth Rock、Reflect 等场地一方效果）、`|-swapsideconditions|`（Court Change）
- 易变状态：`|-start|`/`|-end|`（confusion、Taunt、Substitute 等 volatile status）
- 招式效果反馈：`|-crit|`（暴击）、`|-supereffective|`、`|-resisted|`、`|-immune|`、`|-miss|`、`|-fail|`、`|-block|`、`|-hitcount|`、`|-singleturn|`/`|-singlemove|`（Protect/Focus Punch 等）
- 道具与特性：`|-item|`/`|-enditem|`、`|-ability|`/`|-endability|`、`|-activate|`
- 形态/Z/极巨：`|-mega|`、`|-primal|`、`|-burst|`、`|-zpower|`、`|-zbroken|`、`|-transform|`

每条事件后还可能跟 `|[from] EFFECT|[of] SOURCE`、`|[silent]`、`|[still]` 这些 tag 说明来源/抑制动画。

**来源**：`sim/SIM-PROTOCOL.md:230-595`

### 4.6 私密 vs 公开信息（`|split|` 协议）

模拟器为了避免泄露信息，会发：
```
|split|p1
<SECRET — 给 p1 看的，含确切 HP 等>
<PUBLIC — 给所有人看的>
```
你写 bot 时一定要尊重这个分隔，不要拿 SECRET 的内容当对手的 observation。

**来源**：`sim/SIMULATOR.md`（"Reading from the simulator" 节）

---

## 5. 队伍 / 构筑（Team / Set）

每只宝可梦 = **种族 (species) + 道具 (item) + 特性 (ability) + 4 招 + 性格 (nature) + 努力值 (EVs) + 个体值 (IVs) + 等级 + 性别 + 是否闪光 + Tera 属性 + 极巨等级**。

PS 用三种格式：

**1. Export format（人类粘贴用，Smogon 标准）**
```
Articuno @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
IVs: 30 SpA / 30 SpD
- Ice Beam
- Hurricane
- Substitute
- Roost
```

**2. JSON format（程序内部用）**
```json
{"name":"","species":"Articuno","gender":"","item":"Leftovers",
 "ability":"Pressure",
 "evs":{"hp":252,"atk":0,"def":0,"spa":252,"spd":4,"spe":0},
 "nature":"Modest",
 "ivs":{"hp":31,"atk":31,"def":31,"spa":30,"spd":30,"spe":31},
 "moves":["Ice Beam","Hurricane","Substitute","Roost"]}
```

**3. Packed format（一行字符串，传输/日志用）**
```
NICKNAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|HAPPINESS,POKEBALL,HIDDENPOWERTYPE,GIGANTAMAX,DYNAMAXLEVEL,TERATYPE
```
多只用 `]` 分隔。EVs/IVs 用 `,` 分隔六维（HP,Atk,Def,SpA,SpD,Spe），空位代表默认（0 / 31）。

API 互转：`Teams.unpack` / `Teams.pack` / `Teams.import` / `Teams.export`。

**校验**：`new TeamValidator('gen6nu').validateTeam(team)` 返回 `null` 表合法，否则返回错误数组。

**来源**：`sim/TEAMS.md`（全文 293 行）

### 关于构筑的关键概念（PS 不教你，但 bot 必须懂）

| 概念 | 一句话 |
|---|---|
| **种族值 (Base Stats)** | 物种固有六维基础值，HP/Atk/Def/SpA/SpD/Spe |
| **个体值 (IVs)** | 0–31，每只随机；可被 Hyper Training 改 |
| **努力值 (EVs)** | 通过对战累积，最多 510 总点，单项 ≤252；每 4 点 EV = 1 点实际数值 |
| **性格 (Nature)** | 25 种，影响 +10% / −10% 两项数值 |
| **特性 (Ability)** | 每只 1–3 个可选（含 Hidden Ability） |
| **道具 (Item)** | 每只 1 个；Choice 系/Leftovers/Life Orb 等是元游戏核心 |
| **Tera 属性** | Gen 9 新机制，可临时把宝可梦变成另一属性 |
| **极巨 / 超极巨** | Gen 8 限定，3 回合内变大并改变招式池 |
| **Z / Mega** | Gen 7 / Gen 6-7 的强化形态 |

完整数值/伤害公式见 §7。

---

## 6. 数据库与速查工具

### 6.1 PS 自带数据 API（`sim/DEX.md`）

```js
const {Dex} = require('pokemon-showdown');

Dex.species.get('garchomp').baseStats;   // { hp:108, atk:130, def:95, ... }
Dex.moves.get('earthquake').basePower;   // 100
Dex.abilities.get('intimidate');
Dex.items.get('choice band');
Dex.mod('gen1').moves.get('Tackle').basePower;  // 35（按 Gen1 规则）
```

- `Dex.mod('gen1'…'gen9')` 切换世代规则。
- `*.all()` 返回该类全部条目。
- **过滤掉非标准条目**：检查 `obj.isNonstandard`（值为 `'CAP'`/`'Past'`/`'Future'`/`'LGPE'`/`'Unobtainable'`/`'Pokestar'`/`'Gigantamax'`/`'Custom'`），`null` 才是当前世代标准合法。详细规则见 `sim/NONSTANDARD.md`。

**来源**：`sim/DEX.md`、`sim/NONSTANDARD.md`

### 6.2 外部检索网站（强烈建议收藏）

| 网站 | 内容 | URL |
|---|---|---|
| **Smogon Dex** | 每个世代每个 Tier 的宝可梦数据、推荐打法、Strategy、招式 | https://www.smogon.com/dex/ |
| **Smogon Damage Calculator** | 在线伤害计算器，所有世代 | https://calc.pokemonshowdown.com/ |
| **Pikalytics** | 当前元游戏使用率统计、典型配置（VGC 尤其有用） | https://www.pikalytics.com/ |
| **Bulbapedia** | 最详尽的英文 Wiki，所有机制、招式、特性的权威解释 | https://bulbapedia.bulbagarden.net/ |
| **Serebii** | 另一权威数据站，更新快 | https://www.serebii.net/ |
| **Smogon Forums** | 每个 Tier 的 Sample Teams、Viability Rankings、机制讨论 | https://www.smogon.com/forums/ |
| **PS 官方站** | 直接玩、看 Replay、Teambuilder | https://play.pokemonshowdown.com/ |
| **PS Replay 库** | 公开对战录像（可作监督学习数据） | https://replay.pokemonshowdown.com/ |

---

## 7. 必须理解的对战机制（速成清单）

PS 自身不教游戏规则。下面是研究博弈智能体**最少需要掌握的机制**，每项给出权威来源。

### 7.1 战斗结算顺序

回合内大致顺序（Gen 9）：
1. 选阶段：Mega → Dynamax → 战术（switch / move）。
2. 换人先发生（除非被 Pursuit 等截断）。
3. 招式按 **Priority → Speed → 随机** 排序结算；速度可被天气/特性/场地（Tailwind, Trick Room）修改。
4. 末尾：天气伤害、毒、烧伤、Leftovers、场地计时、Wish 触发等。

权威解释：https://bulbapedia.bulbagarden.net/wiki/Priority、https://www.smogon.com/articles（搜 "Gen X mechanics"）

### 7.2 伤害公式

简化版（Gen 5+）：
```
Damage = (((2*Level/5 + 2) * Power * A/D) / 50 + 2) * Modifiers
```
`Modifiers` 包括：STAB、属性相克、暴击、随机扰动 (0.85~1.00)、天气、道具、特性、状态…

完整公式见：https://bulbapedia.bulbagarden.net/wiki/Damage 与 Smogon calc 源码（仓库 https://github.com/smogon/damage-calc）。

### 7.3 属性相克表

18 种属性两两克制关系：`Dex.types.getEffectiveness('Fire','Grass')` 或查表
https://bulbapedia.bulbagarden.net/wiki/Type/Type_chart

### 7.4 异常状态（Status）

5 个主异常（**互斥**，一只同时只能有一个）：`brn` 烧伤、`par` 麻痹、`psn` 中毒、`tox` 剧毒、`slp` 睡眠、`frz` 冰冻。每个对数值/出招都有特定影响。

**易变状态（Volatile）** 可叠加：confusion、infatuation、Taunt、Encore、Disable、Substitute、Leech Seed、Yawn、Perish Song …

**来源**：`sim/SIM-PROTOCOL.md:354-395`、https://bulbapedia.bulbagarden.net/wiki/Status_condition

### 7.5 天气、场地、一方场地

- **Weather**：Sun、Rain、Sandstorm、Hail / Snow、Harsh Sunlight、Heavy Rain、Strong Winds —— 影响伤害、命中、特性触发。
- **Terrain**：Electric / Grassy / Misty / Psychic —— 影响一类招式威力或免疫某状态。
- **Side conditions**：Stealth Rock、Spikes、Toxic Spikes、Sticky Web（钉子类，进场触发）；Light Screen、Reflect、Aurora Veil（护盾）；Tailwind、Safeguard、Mist。
- **Pseudo-weather**：Trick Room（反转速度）、Magic Room（停物品）、Wonder Room（互换 Def/SpD）、Gravity。

**来源**：`sim/SIM-PROTOCOL.md:396-432`

### 7.6 换人与陷阱

进场触发：天气特性（Drought）、Intimidate、入场钉子伤害、Regenerator 恢复…
**陷阱效果**：Shadow Tag、Arena Trap、Magnet Pull、Mean Look、Block、Fairy Lock —— 让对方不能换。
**强制换**：Whirlwind、Roar、Dragon Tail、Circle Throw、被打死、Eject Button、U-turn / Volt Switch / Parting Shot。

### 7.7 优先级招式

`Priority = -7~+5`。常用：Quick Attack/Mach Punch (+1)、Bullet Punch、Sucker Punch (+1)、Extreme Speed (+2)、Fake Out (+3)、Protect/Detect (+4)、Helping Hand (+5)、Pursuit (+7 仅对换人)、Trick Room (-7)。
完整表：https://bulbapedia.bulbagarden.net/wiki/Priority

### 7.8 速度（Speed Tier）

Tier 列表是 PVP 核心知识，决定谁先手：https://www.smogon.com/dex/sv/pokemon/?sort=spe-d
（按 Speed 排序的所有合法宝可梦）

### 7.9 PP / Struggle

每招式 PP 限制；全用尽则只能用 Struggle（自伤 1/4 HP，普通属性）。

---

## 8. 客户端协议 vs 战斗协议（别混淆）

PS 有**两层**协议，写 bot 时经常混乱：

| 协议 | 范围 | 文档 |
|---|---|---|
| **客户端协议** | 登录、加入房间、聊天、发起挑战、加入天梯队列 | `PROTOCOL.md` |
| **战斗协议** | 队伍设定、出招/换人、战斗事件、胜负判定 | `sim/SIM-PROTOCOL.md` |

- 用 **`BattleStream` 或 `simulate-battle`** 时：你**只跟战斗协议打交道**（`>start` / `>p1 move 1`），不用关心房间/登录。
- 用 **WebSocket 连真实 PS 服务器**时：先走客户端协议（`|challstr|` 登录、`|/utm` 上传队伍、`|/search`/`|/challenge`），战斗开始后服务器才把战斗协议事件转发到对应房间。

> **建议**：纯做 AI 研究就用 `BattleStream`（最快、确定性、可设 `seed`）。需要跟真人或外部 bot 对战才走 WebSocket。poke-env 默认走 WebSocket。

**来源**：`PROTOCOL.md:1-100`、`sim/SIMULATOR.md`

---

## 9. 给智能体研究者的实操建议

### 9.1 选择起步格式

| 目标 | 推荐格式 |
|---|---|
| 排除 teambuilding，专注对战决策 | `gen9randombattle`（单打）/ `gen9randomdoublesbattle`（双打） |
| 经典竞技 Tier，元游戏稳定 | `gen8ou`（社区数据最多）、`gen9ou` |
| 双打 + 真实赛制 | `gen9vgc2024regg`（VGC 2024 Reg G） |
| 自由测试 / 单元测试 | `gen9customgame`（无任何限制） |
| 简化决策空间 | `gen91v1`（1v1）或 `gen92v2doubles` |

### 9.2 数据来源

- **监督学习**：`https://replay.pokemonshowdown.com/` 有海量公开 replay（每条 replay 末尾加 `.log` 拿原始战斗协议）。
- **生成数据**：自己跑 `BattleStream` 或 `simulate-battle` + 两个 bot，输出量没上限。
- **元游戏统计**：Smogon 每月发布 `usage stats`（每个 Tier 的招式/道具/特性使用率）：https://www.smogon.com/stats/

### 9.3 评估对手

PS 自带的难度阶梯（你可以让 bot 依次打）：
1. `RandomPlayer`（poke-env 内置，随机出招）
2. `MaxBasePowerPlayer`（poke-env 内置，挑威力最大）
3. Smogon 默认的 **`Default` 选择**（在 PS console 里就是 `default`）
4. 公开 baseline：[Future Sight AI](https://github.com/pmariglia/showdown)、[Pokémon Showdown LLM bot 论文](https://arxiv.org/list/cs.AI/recent) 中的若干实现
5. 真人 Ladder（设置账号在公网或邀请人来本地服）

### 9.4 已知陷阱（写过 bot 都踩过）

- **本地代理会拦截 WebSocket**：`export NO_PROXY=localhost,127.0.0.1,::1`（你仓库里 `env_setting.sh` 已有）。
- **没开 `--no-security`** → 一批 bot 同 IP 登录会被节流踢线。
- **`request` 有 `wait:true`** → 对手还没选完，不要重复发指令。
- **`forceSwitch:[true]`** → 当前你必须换人而不是出招。
- **`trapped:true`** → 不能换人。
- **强制换人请求和普通出招请求会在同一回合内出现两次** —— 别把"一回合一个决策点"当死规则。
- **种族条款**：`Species Clause` 下队伍不能有两只同种族；写 random 队伍生成器要注意。
- **HP 是 `/100` 还是 `/48`**：见 §3.3。

---

## 10. 延伸资源汇总（按"我想做 X"分类）

| 目标 | 资源 |
|---|---|
| **理解 PS 代码结构** | `ARCHITECTURE.md`、Discord 开发频道 https://psim.us/devdiscord |
| **写战斗 bot（Python）** | poke-env 文档 https://poke-env.readthedocs.io ｜ 仓库 https://github.com/hsahovic/poke-env |
| **写聊天/管理 bot** | Kaiepi 的 Bot FAQ https://gist.github.com/Kaiepi/becc5d0ecd576f5e7733b57b4e3fa97e |
| **复现 SOTA AI** | Smogon 论坛"AI"分区、arXiv 搜 "Pokemon Showdown" |
| **学习游戏机制深层细节** | Smogon 各代 "Mechanics Research" 帖、Bulbapedia |
| **数据/统计** | https://www.smogon.com/stats/、https://www.pikalytics.com/ |
| **伤害计算器（带源码）** | https://github.com/smogon/damage-calc ｜ https://calc.pokemonshowdown.com/ |
| **Replay 数据** | https://replay.pokemonshowdown.com/ |
| **官方 PS 源码** | https://github.com/smogon/pokemon-showdown |
| **官方 PS 客户端源码** | https://github.com/smogon/pokemon-showdown-client |
| **PS Web API** | https://github.com/smogon/pokemon-showdown-client/blob/master/WEB-API.md |
| **PS 论坛** | https://www.smogon.com/forums/forums/pokémon-showdown.209/ |

---

## 附录 A：常用命令速查

```bash
# 启动服务器（写 bot 必备）
cd /Users/hoi/project/game/pokemon-showdown
node pokemon-showdown start --no-security

# 生成一支 OU 随机队并导出可读格式
./pokemon-showdown generate-team gen9ou | ./pokemon-showdown export-team

# 校验自定义队伍
cat my_team.txt | ./pokemon-showdown validate-team gen9ou

# 在 CLI 里模拟一场对战
echo '>start {"formatid":"gen9randombattle"}
>player p1 {"name":"Alice"}
>player p2 {"name":"Bob"}' | ./pokemon-showdown simulate-battle

# 让两个简单 bot 对战 100 场（已写好的 bot.py）
cd /Users/hoi/project/game/pokemonbot
source env_setting.sh
python bot.py
```

## 附录 B：决策选择语法速查

```
team 213456                          # Team Preview
move <name|slot> [target] [mega|zmove|max]
switch <name|slot>
pass                                 # 双打无需此栏
default                              # 让引擎选
undo                                 # 撤销（须对手未提交）
```
双打：两栏之间 `,` 分隔。Target：正数=对方栏（左→右 1,2[,3]），负数=己方栏。

## 附录 C：协议事件分类速查

```
# 元数据
|player|  |teamsize|  |gametype|  |gen|  |tier|  |rated|  |rule|
|clearpoke|  |poke|  |teampreview|  |start|

# 流程
|request|  |inactive|  |inactiveoff|  |upkeep|  |turn|  |win|  |tie|  |t:|

# 主要动作
|move|  |switch|  |drag|  |detailschange|  |replace|  |swap|  |cant|  |faint|

# 数值变化
|-damage|  |-heal|  |-sethp|  |-status|  |-curestatus|  |-cureteam|

# 能力 / 易变
|-boost|  |-unboost|  |-setboost|  |-swapboost|  |-invertboost|
|-clearboost|  |-clearallboost|  |-clearpositiveboost|  |-clearnegativeboost|  |-copyboost|
|-start|  |-end|  |-singleturn|  |-singlemove|

# 场地 / 天气
|-weather|  |-fieldstart|  |-fieldend|  |-sidestart|  |-sideend|  |-swapsideconditions|

# 命中 / 抗性
|-crit|  |-supereffective|  |-resisted|  |-immune|  |-miss|  |-fail|  |-block|  |-notarget|  |-hitcount|

# 道具 / 特性 / 形态
|-item|  |-enditem|  |-ability|  |-endability|  |-transform|
|-mega|  |-primal|  |-burst|  |-zpower|  |-zbroken|

# 杂项
|-activate|  |-hint|  |-center|  |-message|  |-combine|  |-waiting|  |-prepare|  |-mustrecharge|
```

完整定义见 `sim/SIM-PROTOCOL.md`（942 行）。

---

*本手册基于本地 `/Users/hoi/project/game/pokemon-showdown/` 仓库内的官方 markdown 文档与 `config/formats.ts` 配置整理，外部链接均为官方/社区权威站点。若仓库更新，建议重新跑一遍 §1 表里的文件做差异。*
