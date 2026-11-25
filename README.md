# Game Agent

[Interactive Demo & Benchmark Explorer](https://dsqi-3721.github.io/game_agent/) · [主实验数据 (`baselines.json`)](results/baselines.json) · [跨局记忆数据 (`player_memory.json`)](results/player_memory.json)

面向不完全信息对抗环境（Pokémon Showdown Gen 9 OU 及 MicroRTS 扩展）的博弈智能体研究项目，重点研究**在线联合对手建模**、**跨局玩家策略记忆迁移**、**低时延快速决策（System 1）**以及**大模型长程规划与规则搜索协同**。

---

## 核心结果一览

### 1. 不完全信息对手建模主实验

主实验在相同的 14 名受控玩家、28 个完整玩家对局（978 个非预览提交动作）上进行，覆盖 17,689 个尚未公开招式、4,300 个道具、4,161 个特性和 5,303 个太晶属性目标（只计入决策时尚未公开的隐藏配置，预测失败计入分母；完整指标见 [`results/baselines.json`](results/baselines.json)）：

| 方法 | 隐藏招式 Recall@3 | 隐藏道具 Top-1 | 隐藏特性 Top-1 | 隐藏太晶 Top-1 | 提交动作 Top-1 |
|---|---:|---:|---:|---:|---:|
| **Joint C/X/M/Z filter（Ours）** | **76.53%** | **78.65%** | **96.20%** | **50.88%** | **41.00%** |
| Current neural prediction heads | 73.50% | 67.84% | 95.53% | 43.20% | 23.82% |
| Foul Play native tracker + set sampler | 70.77% | 70.28% | 94.30% | 36.13% | — |
| Metamon NaiveUsagePredictor | 65.85% | 56.21% | 94.71% | 40.17% | — |
| PokéChamp BayesianTeamPredictor | 66.62% | 52.77% | 94.71% | 43.75% | — |
| PokéChamp opponent prompt + Qwen3-8B | — | — | — | — | 24.13% |
| PokéChamp opponent prompt + Qwen3.8-27B | — | — | — | — | 25.56% |
| Pure Qwen3.8-27B | 26.21% | 15.53% | 40.62% | 22.59% | 24.03% |

### 2. 跨局玩家记忆与换队迁移

在 40 名未参与训练的非随机玩家（另设 10 名随机控制器作为负对照）上固定新队伍对局，使用前 5 局历史预测前 10 次决策（完整结果见 [`results/player_memory.json`](results/player_memory.json)）：

| 跨局记忆条件 | 行动 NLL ↓ | 行动分布 KL ↓ | 画像 MAE ↓ | 行动 Top-1 ↑ |
|---|---:|---:|---:|---:|
| **正确玩家历史（5 局）** | **1.4530** | **0.2191** | **0.0675** | 46.75% |
| 无历史 | 1.5154 | 0.2903 | 0.0963 | 45.25% |
| 其他玩家历史 | 1.5107 | 0.2748 | 0.0847 | 47.25% |

正确玩家历史在换队后将行为画像 MAE 由 **0.0963 降至 0.0675（降低 29.9%）**，行动分布 KL 由 **0.2903 降至 0.2191**。

![玩家记忆与换队迁移](results/figures/player_memory.png)

在共同参考局面下比较 50 名留出玩家的预测行为倾向（受威胁时换人、高血量强化、低血量回复、可用时太晶化），数据见 [`results/player_profiles.json`](results/player_profiles.json)：

![标准化玩家行为画像](results/figures/standardized_player_profiles.png)

---

## 项目结构

| 模块 | 内容 | 入口 |
|---|---|---|
| 对手建模 | 隐藏配置（C）、战斗状态（X）、信息记忆（M）与动态策略（Z）的联合推断 | `src/arena/opponent/joint/` |
| System 1 | 本地低时延决策模型、动作监督微调与 PPO | `src/arena/s1/`、`scripts/s1_action_train.py` |
| HyMem 协同 | 策略规划、分类记忆路由、执行协调与搜索融合 | `src/arena/agents/hymem/`、`src/arena/agents/fusion/` |
| 对战框架 | 统一观测、智能体注册、配队、对局调度和终端界面 | `src/arena/`、`configs/experiments/` |
| 外部方法 | Foul Play、Metamon、PokéChamp 和 PokeLLMon 的桥接与复现脚本 | `external/` |
| MicroRTS 扩展 | 基于小模型快速决策与 SA-RLCD 在线强化学习训练套件 | `microRTS/` |
| 研究结果 | 主实验、玩家记忆实验、行为画像 JSON 与图表 | `results/` |

---

## 联合对手建模设计

模块维护四类变量的联合信念：

* **C（初始配置）**：队伍、招式、道具、特性、性格、努力值、个体值与太晶属性。
* **X（战斗状态）**：血量、PP、锁招、道具消耗与交换、状态和能力变化。
* **M（信息记忆）**：对手合法可见的历史及其编码表示。
* **Z（策略状态）**：行为原型、连续策略表示与跨局策略变化。

神经网络采用 **8 层、256 维带记忆因果 Transformer**，包含配置生成、动态策略、条件行动和推断提议四类计算；序贯蒙特卡洛（SMC）过滤器将网络概率与 Showdown 机制观测结合，逐步更新联合假设（默认 1,024 粒子、128 配置候选）。新一局重置配置与战斗状态，仅继承玩家级策略记忆 `Z`。

`JointOpponentModel` 提供 `update`、`predict`、`query`、`sample_joint`、`condition`、`forecast`、`new_game`、`snapshot` 与 `restore` 接口；`dossier()` 输出结构化 JSON 资料卡，包含配置预测、当前状态、行为倾向和行动分布。

---

## 安装与运行

建议使用 Python 3.11、Node.js 和支持 CUDA 的 PyTorch。训练与完整推断需要另行准备模型权重和数据。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[s1,s1-gpu,opponent,viz]'
cp .env.example .env
```

Showdown 放在与本仓库同级的 `pokemon-showdown/`（对应提交 `1636bf6d0848fb3c9330576963a647d4e179d278`）：

```bash
git clone https://github.com/smogon/pokemon-showdown.git ../pokemon-showdown
git -C ../pokemon-showdown checkout 1636bf6d0848fb3c9330576963a647d4e179d278
(cd ../pokemon-showdown && npm install && npm run build)
```

运行智能体对战时，另开终端启动本地模拟器：

```bash
cd ../pokemon-showdown
node pokemon-showdown start --no-security
```

基础对战与终端界面：

```bash
arena run configs/experiments/quick_baselines.yaml
arena tui
```

模型服务地址在 `configs/models/registry.yaml`，密钥放在 `.env`。外部方法通过各自 `external/<method>/setup.sh` 安装；桥接协议见 [external/README.md](external/README.md)。

---

## 数据构建、训练与评测复现

训练数据由两类来源构成：本地受控对局（双方合法视角、提交动作、完整配置与状态轨迹）与 Metamon / PokéChamp 公开人类回放（缺失隐藏标签保持缺失，不补造）。原始数据与权重存放在 `artifacts/`、`models/` 或 `runs/`（不随代码发布）。

```bash
# 1. 构建配置目录与生成受控对局数据（默认预算 40 万局，可用 --groups 缩减）
python scripts/opp_joint_catalog.py \
  --sets /path/to/showdown_sets/gen9ou.json \
  --teams /path/to/foulplay/gen9ou/teams.json
node scripts/node/opp_joint_catalog.js artifacts/opponent_joint
python scripts/opp_joint_generate.py --workers 12

# 2. 处理人类回放并训练联合模型
python scripts/opp_joint_human_data.py --workers 6
python scripts/opp_joint_prepare_training.py --workers 8
torchrun --standalone --nproc_per_node=2 scripts/opp_joint_train.py \
  --epochs 0 --min-epochs 5 --batch 256

# 3. 独立主实验与跨局玩家记忆迁移评测
python scripts/opp_joint_public.py --split test \
  --start-group 70 --groups 14 --games 2 \
  --output artifacts/opponent_joint/training/joint_validation --workers 2

python scripts/opp_profile_transfer.py prepare
python scripts/opp_profile_transfer.py support --workers 4
python scripts/opp_profile_transfer.py match
python scripts/opp_profile_transfer.py query --workers 4
python scripts/opp_profile_transfer.py report
python scripts/opp_profile_visualization.py
```

单局 JSON 历史推断示例（字段定义见 [`data/schemas/opponent-joint-input.json`](data/schemas/opponent-joint-input.json) 与 [`data/schemas/opponent-joint.json`](data/schemas/opponent-joint.json)）：

```bash
python scripts/opp_joint_predict.py \
  --input /path/to/history.json \
  --output /path/to/opponent.json \
  --model artifacts/opponent_joint/training/adaptation/model.pt \
  --device cuda:0
```

---

## 已知限制

当前主匹配实验与跨局记忆实验基于受控玩家，外部人类回放提供另一类训练与评测监督。28 个玩家对局的主实验规模仍有限；神经网络直接预测与联合过滤采用不同的行动预测头，因此主实验支持整体系统对比，不单独归因于单一组件。跨局记忆对行动 Top-1 的提升尚未稳定，主要增益体现在概率分布校准（NLL / KL）与行为画像误差（MAE）。
