# 模拟器规格与标定台账

模拟器是**假设的集合**，不是真相。本文件是这些假设的完整台账。

每一条都必须有：当前取值、来源、**如果它错了会怎样**、以及用什么实验把它定下来。
没有登记的魔数不许出现在 `ato/sim/` 里。

> 依据 INVARIANT I-5：训练时对这些参数做域随机化。
> 只在单一取值下成立的策略优势，判定为 sim exploit，不予晋级。

---

## A. 直接来自游戏数据（不是假设）

这些不需要标定——它们是从 `ArknightsGameData` 里读出来的。列在这里是为了说明
**哪些不用怀疑**，把注意力集中到真正的未知上。

| 项 | 来源 |
|---|---|
| 地图格子、可部署类型、可通行掩码 | `level.mapData.tiles` + `.map` |
| 敌人路线、检查点、朝向 | `level.routes[].checkpoints` |
| 波次调度：数量、间隔、preDelay、routeIndex | `level.waves[].fragments[].actions[]` |
| 初始 DP、回费周期、DP 上限、生命上限、上场人数 | `level.options` |
| 难度 rune（突袭/危机合约修正） | `level.runes[]` + `difficultyMask` |
| 敌人数值（逐等级）、免疫、taunt、漏怪扣血 | `enemy_database.json` |
| 敌人伤害类型、精英/BOSS 分级 | `enemy_handbook_table.json` (`damageType`, `enemyLevel`) |
| 干员数值（分阶段关键帧）、信赖、潜能 | `character_table.json` |
| 攻击范围格子 | `range_table.json` |
| 技能 sp 类型/消耗/初始/持续、效果 blackboard | `skill_table.json` |

---

## B. 标定参数（假设，必须实测）

按"错了会有多糟"排序。

### B-1 `move_tiles_per_second` — 最关键的一个

- **当前取值**：`1.0` 格/秒（对 `moveSpeed == 1.0` 的单位），再乘 `options.moveMultiplier`（抽样 58 关恒为 `0.5`）→ 有效 `0.5 格/秒`。
- **来源**：`moveMultiplier` 在所有采样关卡里都是常数，说明它是全局换算系数而非关卡旋钮；系数本体是猜的。
- **错了会怎样**：**灾难性**。它决定了干员有多少时间输出。偏高 20%，模拟器就认为拦不住的怪其实拦得住，
  学出来的策略在实机上会一路漏怪；偏低同理会过度保守。这是模拟器里唯一一个错了会让**所有**训练白费的参数。
- **怎么定**：从 B 站录像里取单只敌人从出生点走到蓝门、且**全程未被阻挡**的片段，
  用 DP 计数器当时钟（DP 是确定性的：`initialCost + t/costIncreaseTime`），量出格数/秒。
  不同 `moveSpeed` 的敌人各取 ≥10 个样本做线性回归，同时验证比例关系是否真的线性。

### B-2 波次片段的时间基准 `fragment_relative_delays`

- **当前取值**：`True`（action.preDelay 相对**片段**开始计时）。
- **来源**：1-7 里 `fragment[1].preDelay == actions[0].preDelay == 3.0`，两种解释在单片段波次下等价，数据本身无法区分。
- **错了会怎样**：多片段关卡的出怪时间整体错位，波次越多误差累积越大。判断"能不能来得及回费"会系统性偏差。
- **怎么定**：找一个有 ≥3 个片段且片段 preDelay 各不相同的关卡，录像量第一只怪的出生时刻，与两种模型的预测对拍。

### B-3 干员部署锁定 `deploy_lock_seconds`

- **当前取值**：`1.0` 秒（落地动画期间不可攻击、不可被攻击）。
- **错了会怎样**：影响"卡时间点部署"这类高阶操作的可行性判断。对紧张关卡的最优解影响明显。
- **怎么定**：录像逐帧量从松手到第一次挥刀的间隔。

### B-4 重新部署费用与冷却 `redeploy_multiplier` / 费用递增

- **当前取值**：冷却 = `respawnTime × 1.0`；再部署费用 = `cost + min(已部署次数, 2)`。
- **来源**：费用 +1/+2 封顶是社区共识，未在数据中找到出处。
- **错了会怎样**：影响所有"快速反复部署"（如快活跃干员）战术的评估。
- **怎么定**：实机连续部署/撤退同一干员，读卡面费用数字。

### B-5 攻击间隔量化 `attack_quantization`

- **当前取值**：`interval = baseAttackTime × 100 / attackSpeed`，攻速钳位见 B-13；
  默认 `"none"`——间隔换算成 tick 后**进位取整并把余数带到下一次**，长期速率精确。
- **备选**：`"ceil"` / `"round"`——每次攻击各自量化到整 tick，不带余数。
  定步长客户端循环的量化是涌现属性，语义偏 ceil；但没有实测。
- **错了会怎样**：`baseAttackTime = 0.78` → 23.4 tick，round 得 23 / ceil 得 24，约 4% DPS 差；
  在"刚好差一下打不死"的边界关卡会翻转结论。
- **怎么定**：录像里数固定时间窗内的攻击次数。
- **在此之前**：三种语义都进域随机化（`CALIBRATION_CHOICES`），策略不许只在一种读法下成立。

### B-6 增益叠加顺序

- **当前取值**：`(base + Σadd) × (1 + Σmul)`——先加法后**求和**的百分比。
- **错了会怎样**：如果客户端实际是**连乘**，模拟器会低估多重攻击力增益的收益，
  策略会系统性回避叠 buff 的阵容——这正是用户担心的"学到错误偏好"。
- **怎么定**：实机拿一个能同时吃到 2-3 个百分比增益的干员，读伤害数字反解。

### B-7 技能 SP 充能速率

- **当前取值**：`INCREASE_WITH_TIME` 每秒 1 SP × `spRecoveryPerSec`；攻击/受击各 1 SP。
- **错了会怎样**：技能循环节奏错位，影响所有依赖技能开关时机的战术。
- **怎么定**：录像量技能条从 0 到满的秒数，对照 `spCost`。

### B-8 索敌规则

- **当前取值**：干员优先打**已阻挡**的敌人，否则打**最早进入攻击范围**的；平局按 uid 稳定排序。
  敌人：被阻挡则打阻挡者；远程敌人打半径内最近的干员。
- **错了会怎样**：AOE 与溅射干员的价值评估失真；"换血位"这类操作的模拟不准。
- **怎么定**：可控实验——摆多只敌人进同一个干员范围，观察打谁。注意部分干员特性会改写索敌
  （优先低血量 / 优先空中 / 优先未阻挡），这些需要按 subProfession 逐个登记。

- **扫描频率 `target_scan_period_ticks`**：当前取 3（10 Hz）。
  调研称客户端每 3 帧扫描一次，未经实测。每 tick 扫描会让干员的反应比真实快最多 2 帧，
  系统性高估拦截能力。开火仍然按冷却到期的那一 tick 进行（对上一次扫描到的目标）——
  让开火也等扫描会给每次冷却额外加最多 2 帧，反而压低持续 DPS。
  随机化取 {1, 2, 3}。

### B-9 阻挡判定的几何

- **当前取值**：敌人**连续坐标四舍五入到的格子** == 干员所在格 时被阻挡。
- **错了会怎样**：阻挡发生的时机偏移半格，影响"卡格子"打法。
- **怎么定**：逐帧看敌人图元与干员的相对位置在何时定住。

### B-10 rune 语义

- **已建模**：`gbuff_lifepoint` / `global_lifepoint`（+生命）、`global_initial_cost_add`（+初始 DP）、
  `cbuff_cost_recovery`（回费倍率）、`ebuff_attribute` / `enemy_attribute_mul`（敌人属性倍率）。
- **未建模**（会产生 novelty 事件）：`env_system_new`、`enemy_dynamic_ability_new`、`level_hidden_group_enable`、
  `enemy_skill_blackb_mul`、`enemy_talent_blackb_add`、`map_tile_blackb_assign`、`char_*_mul` 等。
- **错了会怎样**：高难内容（危机合约、保全）的难度修正正是靠 rune 表达的。未建模就直接把关卡
  判为不可训练，**这是正确行为**——绝不能用一个更容易的假关卡产生奖励。

### B-11 难度掩码的整数编码

- **当前取值**：`NORMAL=1, FOUR_STAR=2, EASY=4, SIX_STAR=8` 的位标志。
- **来源**：部分旧关卡文件把 `difficultyMask` 存成整数；1 与 2 出现的 rune 键与字符串版本一致，据此推断。
- **错了会怎样**：突袭修正被错误地应用到普通关（或反之），整关难度算错。
- **怎么定**：找同时存在整数版与字符串版掩码的关卡对照。

### B-12 枚举整数序

- **当前取值**：checkpoint / waveAction / motionMode 的整数按 `ato/sim/registry.py` 里的声明序解释。
- **来源**：按整数出现频次与字符串出现频次的分布匹配推断，**未经权威确认**。
- **错了会怎样**：旧关卡的检查点被误读（例如把"等待"读成"移动"），出怪路径整体错误。
- **怎么定**：找同一关卡的新旧两版数据对照；或解包客户端读枚举定义。

### B-13 攻速钳位 `aspd_min` / `aspd_max`

- **当前取值**：`[20, 600]`（百分比）。
- **来源冲突（未解决）**：一份调研称引擎常量为 `[10, 600]`（社区报告值，非从二进制读出），
  并提到 PRTS 另称实际下限 20；另一份直接称 `[20, 600]`。**两者都不是权威来源**，
  上界更是完全没有出处。
- **错了会怎样**：下界决定重减速下干员还能不能打（`attackSpeed=5` 时 10 与 20 差一倍输出）；
  上界决定多重攻速增益的收益上限——在修复前根本没有上界，`attackSpeed=2000` 被照单全收，
  算出 0.05 秒的攻击间隔。
- **怎么定**：实机用重减速 / 重叠加速单位实测攻击次数。
- **在此之前**：下界的随机化区间覆盖 `[10, 20]`，跨过两份调研的分歧。

---

## C. 明确不建模的部分（当前）

以下会产生 novelty 事件并在 strict 模式下拒绝该关卡，**而不是**假装支持：

- 干员天赋（`talents[].blackboard`）——只解析不生效
- 大部分技能效果（只实现了 `atk_scale` / `def_scale` / `attack_speed` / `max_hp` 等通用键）
- 模组（`battle_equip_table`）
- 特殊地格效果（传送门 `tile_telin/telout`、毒雾、流沙、减防地板等）
- 预设单位的激活/撤离（`ACTIVATE_PREDEFINED` 等波次动作）
- 敌人技能与天赋、召唤物、分裂
- 元素损伤累积（`epDamageResistance` / `epResistance`）
- 隐藏组（`hiddenGroup`）与分支波次（`branches`）

**这份清单就是路线图。** 每实现一项，就从这里移到 A 或 B。

---

## D. 校准协议

1. **对拍实验**：同一关卡 + 同一队伍 + 同一动作脚本，模拟器与实机各跑一次。
2. **对齐信号**：用 DP 计数器当主时钟（确定性、易 OCR），击杀计数当校验。
3. **发散度量**：首次漏怪时刻误差（秒）、击杀时刻序列 L1 误差、通关/不通关一致率。
4. **通过阈值**：
   - 模仿学习可用：通关判定一致率 ≥ 90%
   - 搜索/规划可用：击杀时刻 L1 误差 ≤ 1.0 秒
   - RL 训练可用：击杀时刻 L1 误差 ≤ 0.5 秒 且 首次漏怪时刻误差 ≤ 0.5 秒
5. **参数拟合**：以上述误差为目标，用 CMA-ES 拟合 B 节参数。
6. **回归套件**：固定一组关卡 × 脚本，每次改模拟器都必须复跑，误差不得退化。

---

## E. 缺陷台账

已复现并量化的模拟器缺陷。**改好一条就在这里留下复现数字和守住它的测试**，
不要只在提交信息里说——提交信息会沉下去，这份表不会。

### E-1 DP 累加器有浮点漂移 ✅ 已修（`tests/test_timing.py`）

**症状**：`_cost_accum += dt * scale / cost_increase_time`，`sum(30 × (1/30)) =
0.99999999999999988898 < 1.0` → 第 1 点 DP 落在 tick 31（应为 30），300 tick 只累出 9 点 DP。

**为什么这是最高优先级**：DP 决定每一次部署的时机，约 3% 的系统性亏欠会推迟每一个计划。
更要命的是**它不会被标定暴露，只会被标定吸收**——CMA-ES 会通过放慢敌人把缺的 DP 买回来，
于是模拟器在拟合过的轨迹上看起来很准，而里面每一个常数都是错的。

**修法**：`ato/sim/timing.PeriodicGrant`。周期一次性化成精确整数比，累加器是 `int`。
`state.time` 改由 tick 计数**一次除法**导出，不再累加。

**回归测试**：`test_periodic_grant_lands_on_the_exact_tick`、
`test_first_dp_arrives_on_tick_30_not_31`、`test_ten_seconds_of_battle_yields_ten_dp`、
`test_the_clock_is_derived_from_the_tick_count`。

### E-1b 同一个 bug 也在 SP 上 ✅ 已修（同上）

`charge_sp(dt * rate)` 是同一种累加，所以每个 30 SP 技能都在 31 秒才好。
SP 改为存**tick 单位**：每 tick 加的是「每秒速率」这个好数，只在比较阈值时才除以 tick 率。
`sum(30 × 1.0)` 精确，`sum(30 × (1/30))` 不精确——这就是全部诀窍。
回归测试：`test_sp_is_stored_in_tick_units_so_a_30_sp_skill_is_ready_at_30_seconds`。

### E-2 攻击速度钳位缺上界 ✅ 已修（`tests/test_gamedata_models.py`）

**症状**：`aspd = max(self.attack_speed, 10.0)`，**没有上界**；`attack_speed = 2000`
被照单全收，算出 0.05 秒攻击间隔。

**修法**：`Stats.attack_interval_within(low, high)`，钳位两端都由调用方给；
默认 `ATTACK_SPEED_BOUNDS = (20.0, 600.0)`，模拟器从 `EngineCalibration.aspd_min/aspd_max`
覆盖。下界的来源冲突未解决（见 B-13），所以它是标定参数而不是常数，随机化区间跨过
`[10, 20]` 这段分歧。

**回归测试**：`test_attack_interval_scales_with_attack_speed`（上下界各一例）、
`test_attack_interval_bounds_are_caller_supplied`。

### E-3 `or` 兜底吞掉合法的零值 ✅ 已修（`tests/test_gamedata_models.py`）

**症状**：`node.get(k) or default` 分不出「零」和「缺字段」，而这份数据里零是真值：

| 字段 | 零的含义 | `or` 干了什么 |
|---|---|---|
| `initialCost` | 开局没有 DP（66 关抽样中 3 关） | 白送 10 点 |
| `moveSpeed` | 这个敌人不动 | 让它走向蓝门 |
| `spRecoveryPerSec` | 技能不自动回 SP | 照回不误 |
| `count`（波次动作） | 不出怪 | 出 1 只关卡里没有的怪 |
| `lifePointReduce` | 漏了不扣血 | 扣 1 点 |

另：`moveMultiplier` 的解析兜底（1.0）与 dataclass 默认（0.5）**互相矛盾**，
字段缺失时敌人速度翻倍。统一为 0.5（唯一观测到的取值）。

**修法**：`ato.gamedata.models.field_or`，一律 `is None` 判定。
**回归测试**：`test_zero_valued_fields_survive_parsing`、`test_zero_initial_cost_is_not_ten_free_dp`、
`test_missing_move_multiplier_falls_back_to_the_observed_value`。

### E-4 攻击间隔帧量化 ✅ 已做成开关（`tests/test_timing.py`）

语义未定，所以**不定**：`attack_quantization` 取 `none` / `ceil` / `round`，
三种都进域随机化。详见 B-5。
回归测试：`test_attack_quantisation_policies`、`test_carrying_the_remainder_keeps_the_long_run_rate_exact`。

### E-5 索敌频率过高 ✅ 已修（默认 3 tick）

`target_scan_period_ticks` 默认 3。开火仍每 tick 判定，只有**索敌**降频——
见 B-8 里为什么不能让开火也等扫描。顺带把 1-7 的规划时间从 51.7 秒降到 34.0 秒。

### E-6 命中时机未建模（抬手 + 弹道飞行）⬜ 未修

**证据**：忽略前摇与弹道飞行时间约 0.8 秒，按 0.5 格/秒计约 **0.64 格**敌人位移——
超过漏怪判定的边界宽度。公开的 `bakemuzzledata` 含 7,064 条 `OnAttack` 命中帧数据。

**为什么先不修**：它依赖解包客户端（RECON §7 第 2 项决定），没有那批数据只能靠猜，
而猜一个 0.8 秒的偏移比现在的零偏移更难发现错在哪。

**影响**：所有"刚好差一下打不死"的边界关卡会给出错误结论。

---

## F. 修复后的基线数字

改了模拟器就要重跑，**并且更新这里**——旧数字比没有数字更危险。
下列全部在 `main_01-07`、同一六人队（phase=1, level=40, mastery=0）上测得：

| 量 | 修复前 | 修复后 |
|---|---|---|
| 规划耗时（120 rollout） | 51.7 s | 34.0 s |
| 规划结果 | CLEARED 41/41，漏 0 | CLEARED 41/41，漏 0 |
| 首次部署 | t=27.6 s（DP≥14） | t=0.0 s（DP≥10） |
| 集成检验耗时（10 次扰动） | 5 s | 3 s |
| 击杀时刻 L1 中位 / 最差 | 1.95 s / 23.37 s | 2.66 s / 9.97 s |
| 扰动下掉血 均值 / 最差 | 0.3 / 1 | 1.7 / 7 |
| 信任等级 | IMITATION | IMITATION |
| `allows_training` | False | False |

首次部署时刻的变化正是修复的意义：原来的计划是对着一个走慢了的钟排的。

**信任等级没有变，这是对的。** 上面全部是 sim-vs-sim 证据，
再漂亮也不能晋级到 `TRAINING`——两次模拟跑得像，跟真实游戏无关。
