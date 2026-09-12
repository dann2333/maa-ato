# CLAUDE.md — ATO 项目工作约定

> 本文件是给未来会话的自己看的。**动手写代码前先读完。**
> 详细论证见 `docs/INVARIANTS.md`；本文件是不可违反的浓缩版。

---

## 0. 项目是什么

ATO = 一个**真正会玩明日方舟**的 AI。不是自动化脚本，不是抄作业执行器。

四条用户提出的硬需求：
1. **完整游玩**：战斗（主线/活动/突袭）、集成战略、日常自动化、高难内容（保全/危机合约）。
2. **持续学习**：能在已有基础上理解并学会新关卡、新机制、新干员，而不是全量重训。
3. **推理够快**：跟得上关卡节奏；必要时可暂停游戏做深度思考。
4. **自我进化 + 多模型互学**。

技术路线（用户已拍板）：**自建离线战斗模拟器 → 搜索 + 模仿学习 → RL → 蒸馏成快策略**。
运行平台：**安卓模拟器 + ADB**（先只做这个，控制层接口要抽象好）。
算力：初始训练可放开；**增量/持续学习必须能在单张消费级 GPU 上跑完**。

---

## 1. 九条不可违反的约束

| # | 约束 | 执行机制 |
|---|------|----------|
| I-1 | **纯视觉 + 触控**。禁止 hook / 内存读写 / 注入 / 封包解析 / 改客户端 | `DeviceController` 只暴露 `screencap/tap/swipe/long_press` |
| I-2 | **按属性建模干员，不按身份**。必须适应不同练度 | 属性向量输入；身份嵌入可丢弃且训练时随机置零；练度域随机化 |
| I-3 | **特权只进 critic 和监督目标，绝不进 policy 输入** | `ObservationSpec.realizable_from_pixels` 逐特征标注 + 硬断言 + 特权消融测试 |
| I-4 | **奖励不可被刷**。主奖励用游戏自身成败判据 | shaping 必须是 potential-based（Ng 1999），单测断言其形式 |
| I-5 | **模拟器是假设不是真相**。未从数据读出的常数都是标定参数 | 登记在 `docs/SIM_SPEC.md`；训练时对其加噪；未建模机制抛异常 |
| I-6 | **对抗式课程必须有界** | 白名单扰动算子 + 可解性过滤 + 真实关卡占比下限 |
| I-7 | **泛化要测量不要声称** | 门禁分四切片报告：见过的关 / 没见过的关 / 没见过的干员 / 没见过的练度 |
| I-8 | **真实账号不可逆操作需人工确认** | 受限动作清单：理智药、抽卡、消耗合成玉等 |
| I-9 | **真实对局是第一学习信号**，模拟器是被校验的前瞻模型 | 每场实机对局产出学习轨迹 + 发散报告 + novelty 标记；计划须过标定集成检验 |

**训练一律 `strict=True`。** 模拟器算不准的关卡绝不能产生奖励信号。

**定时一律走整数 tick，不要用浮点累加。** 30 Hz 下 `sum(30 × 1/30) < 1.0`，
任何"每 tick 加一点、加满就触发"的写法都会慢约 3%（DP、SP、冷却全中招）。
周期用 `ato.sim.timing.PeriodicGrant`（精确整数比），速率用 **tick 单位**累加
（每 tick 加"每秒速率"这个好数，只在比较时除以 tick 率）。
`state.time` 由 tick 计数一次除法导出，**不许累加**。
这类误差最坏的地方不是它错，而是**标定会吸收它**——
CMA-ES 会用别处的偏差把它补回来，于是每个常数都错了而整体"看起来很准"。

**模拟器的可信度是被测量的，不是被假设的。** `ato.fidelity.FidelityLedger` 按关卡记录
sim-vs-real 对拍证据，给出 `UNKNOWN / UNTRUSTED / TRIAGE / IMITATION / PLANNING / TRAINING`
六级信任。没有证据就是 `UNKNOWN`，`UNKNOWN` 什么都不许干。
**sim-vs-sim 的一致性永远不能晋级到 `TRAINING`**——两次模拟跑得像，跟真实游戏无关。

---

## 2. 坐标约定（最容易出错的地方）

- 格子 = `Tile(row, col)`，**`row` 从地图底部往上数**。
- `level.mapData.map` 是**从顶行开始**存的 → `display_index = height - 1 - row`。
  （已在多张不对称地图上实测验证，不要重新推导，也不要"顺手改成从上往下"。）
- 连续坐标 `Vec2(x=col, y=row)`，格子中心落在整数点，1 单位 = 1 格。
- MAA 作业文件的 `location` 是 `[x, y]` 且 `y` 从**顶部**数 —— 与本项目相反，
  转换必须走 `ato.copilot.convert` 里那个唯一的命名函数，不要就地手算。

---

## 3. 已核实的游戏数据事实（不要重新推导，不要与之矛盾）

- 数据源：`raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/zh_CN/gamedata/...`
  （容器内可达；`api.github.com` 被墙返回 403，用 GitHub MCP 工具或 raw 域名）。
- 关卡文件路径 = `levels/<stage_table.levelId 转小写>.json`；共 3533 个 stage，201 个无 levelId（纯剧情）。
- **突袭（stageId 带 `#f#`）与普通关共用同一个 level 文件**，只靠 rune 的 `difficultyMask` 区分。
  1-7 实测：敌人 atk/def/hp ×1.2、生命 +1、回费 ×2。
- 旧版本 level 文件把枚举存成**整数**（checkpoint / waveAction / motionMode）→ 走 `ato.sim.registry` 归一化。
- **地格枚举同样有整数编码**（抽样 85 关中 2 关，约 2.4%），且这些文件的 `playerSideMask` 为 `null`。
  当字符串读会让整关"零可部署格、零可通行格"——静默不可玩，无任何报错。
  映射经实测反推（对同一 tileKey 在字符串文件中的取值比对，八种地格全部自洽）：
  `heightType` 0=LOWLAND / 1=HIGHLAND；`buildableType` 0=NONE / 1=MELEE / 2=RANGED；
  `passableMask` 2=FLY_ONLY / 3=ALL。归一化在 `ato.sim.grid._enum`，映射外的整数产生 novelty。
- `enemy_database.json` 每个字段包成 `{m_defined, m_value}`，高等级只覆写声明过的字段，其余继承 level 0。
- `character_table.json` 每个精英阶段只存两个关键帧（1 级与满级），中间等级线性插值。
- `enemyDbRefs[].overwrittenData` 会覆写敌人数值（实测 113 关中 19% 的引用带覆写、38.9% 的关卡至少一处，
  最极端一例 BOSS 血量 18,000→80,000）。`useDb:false` 表示该敌人完全由关卡就地定义。走 `resolve_enemy_ref`。
- 敌人伤害类型在 `enemy_handbook_table.damageType`：`PHYSIC / MAGIC / NO_DAMAGE / HEAL`；
  `enemyLevel` 为 `NORMAL / ELITE / BOSS`。
- `options.moveMultiplier` 在抽样的 58 个关卡里**恒为 0.5** → 它是全局速度换算系数，不是关卡旋钮。
- **零是合法取值，不是"字段缺失"**：`initialCost: 0`（66 关抽样中 3 关）、`moveSpeed: 0`（不动的敌人）、
  `spRecoveryPerSec: 0`（不自动回 SP）、波次 `count: 0`（不出怪）都真实存在。
  解析一律走 `ato.gamedata.models.field_or`（`is None` 判定），**永远不要写 `node.get(k) or default`**。
- `costIncreaseTime` 存在合法值 `999999.0`（该关几乎不回费）；`initialCost` 观测域
  `[0, 3, 5, 10, 15, 16, 18, 20, 24, 30, 35, 45, 50]`。

---

## 4. 代码约定

- Python ≥ 3.10，`from __future__ import annotations`，全量类型标注，ruff line-length 100。
- 注释解释**为什么**，不解释**做了什么**。不写文件头横幅。不留 `TODO: implement`。
- 游戏数据**永不入库**：`data/` 全部 gitignore，靠 `ato.gamedata.fetch` 按需下载。
  提交任何游戏素材/数据转储都是错的。
- 新增机制 = 在 `ato.sim.registry` 注册一个 handler。查不到的键产生 `NoveltyEvent`，
  strict 模式抛 `UnknownMechanism`。**不要为了让流程跑通而静默忽略未知键。**
- 分层：`gamedata`（数据） → `sim`（模拟器） → `agent`（决策） → `perception`/`control`（实机）
  → `train`（学习） → `evolve`（自进化与门禁）。依赖方向单向向下，不得反向。

---

## 5. 分支与提交

- 开发分支：`claude/arknights-ai-agent-2e1q0n`。推送用 `git push -u origin <branch>`，
  失败按 2s/4s/8s/16s 退避重试最多 4 次。
- 未经明确要求**不要创建 PR**。
- 提交信息说清楚"为什么这么做"和"验证了什么"，不要罗列文件清单。

---

## 6. 当前状态与下一步

见 `docs/ROADMAP.md`。简述：

- [x] 游戏数据层：快照 + 哈希清单 + 类型化视图 + 版本 diff
- [x] 模拟器静态半边：地图 / 路线 / 波次 / rune / 敌人名册 + 机制注册表
- [x] 模拟器动态半边：tick 循环、DP、部署、阻挡、索敌、技能、结算
- [x] 控制层（纯视觉+触控）、作业语料导入、内容 diff、CLI
- [x] agent 层：特征模式（含特权守卫）、观测编码、动作空间、rollout 规划器
- [x] 保真度系统：对局轨迹、发散度量、信任台账、标定集成
- [ ] 感知栈（实机读屏）
- [ ] 模仿学习 → 搜索 → RL → 蒸馏
- [ ] 自进化循环与门禁

**模拟器保真度是整个项目的地基。** 地基不准，上面所有学习都在学一个不存在的游戏。
任何"先跑通再说"的捷径，如果代价是模拟器静默算错，都不要走。
