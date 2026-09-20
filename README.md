# ATO

一个**真正会玩明日方舟**的 AI —— 不是自动化脚本，也不是作业执行器。

设计约束见 [`CLAUDE.md`](CLAUDE.md)（浓缩版）与 [`docs/INVARIANTS.md`](docs/INVARIANTS.md)（论证）。
路线与现状见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

---

## 现在能跑什么，不能跑什么

**先说清楚：它还不能玩游戏。** 一行都不能。

| | 状态 |
|---|---|
| 离线战斗模拟器（地图/路线/波次/DP/阻挡/索敌/技能/结算） | ✅ 能跑 |
| 搜索规划器（给定关卡和队伍，搜出一套部署方案） | ✅ 能跑 |
| 保真度检验（扰动标定常数，给出信任等级） | ✅ 能跑 |
| **实机读屏（感知栈）** | ❌ **`ato/perception/` 是空的，0 行** |
| **驱动真机打关** | ❌ 不存在 |

具体到"从主界面自动操作还是手动进关"这个问题：**两者都还做不到。**

- `ato/agent/execute.py` 的 `PlanExecutor` 驱动的是 `BattleEngine`（模拟器），不是 `DeviceController`。
- `ato/control/gestures.py` 只有**战斗内**的手势：部署、撤退、放技能、暂停、变速。
  **没有任何菜单导航**——没有进关卡、没有选队、没有开始行动、没有结算确认。
- 没有感知，程序不知道当前在哪个界面，也读不出 DP、血量、击杀数。

做出来之后，**第一版应该是"你手动进关，程序只接管战斗内"**，理由有两条：
战斗内才是价值和风险所在（标定需要的就是战斗内轨迹）；
菜单导航是纯 UI 自动化，MAA 已经解决得很好，而且理智药 / 抽卡这类不可逆操作都在那一侧（见 I-8）。

---

## 安装

```bash
pip install -e ".[dev]"          # 核心只依赖 numpy
ato gamedata fetch               # 拉取公开游戏数据快照（约 350 MB，几分钟）
```

游戏数据**永不入库**，`data/` 全部 gitignore。

## 跑起来

```bash
ato stage find 1-7                       # 找关卡
ato stage show 1-7                       # 编译并打印地图 / 波次 / 敌人 / rune / novelty
ato op show 能天使 --phase 1 --level 40   # 干员在指定练度下的属性

# 搜一套方案（干员可以用 char id 或中文名）
ato plan 1-7 --squad 芙蓉,能天使,米格鲁,黑角,夜烟,炎熔 \
    --phase 1 --level 40 --mastery 0 --rollouts 120
```

输出：

```
stage      main_01-07 (NORMAL)
result     CLEARED  kills=41/41  leaked=0  life=10  (120 rollouts)

  0. [COST>=10] deploy char_285_medic2 at (1,3) facing UP   # t=0.0s kills=0
  1. [COST>=11] deploy char_124_kroos at (4,1) facing RIGHT   # t=4.0s kills=0
  ...
  ! unmodelled skill_key:skcom_charge_cost[2] (no modelled effect among ['cost'])
```

方案是**条件触发**的（`COST>=10`），不是时间脚本——这样才能在帧率抖动、二倍速、暂停下照样重放。
末尾的 `!` 是 novelty：模拟器遇到了它不会算的机制，已记录而不是假装算对了。

加 `--check` 会额外跑标定集成检验并报告信任等级：

```
fidelity   robust (reference cleared, 92% of 40 perturbed runs cleared, ...)
trust      UNTRUSTED   allows_training=False
           (sim-vs-sim evidence only; grounding needs a real-device run)
```

**`allows_training=False` 是对的，不是 bug。** 以上全是 sim-vs-sim 证据，
两次模拟跑得像跟真实游戏无关。要让它往上走，必须有一次实机对拍。

---

## 接手开发（交给本地 Claude Code 时看这里）

动手前读 `CLAUDE.md`，它是不可违反的约束清单，会话开始会自动加载。

下一步按依赖顺序，**不要跳**：

1. **给 `ato/control/` 补测试。** 目前零覆盖。上一轮是靠一个临时的假 adb 脚本测出四个 bug 的
   （异步触摸管道吞掉主机侧等待、`input` 回退把一次部署拆成四段、`connect()` 对在线设备返回 False、
   每个动作多跑两次 `adb devices`），那个脚本没入库。真去点设备之前先把这层焊死。
2. **`ato device probe` / `ato device record`。** CLI 目前没有 device 命令组。
   `AdbDevice` 已经支持 `address="127.0.0.1:5555"`，`BenchmarkResult` 也有了，缺的是接起来。
3. **感知栈。** 冷启动可以借 MAA（见 `docs/RECON.md` §8）：它的 `Arknights-Tile-Pos` 覆盖 4,203 关，
   每关带一个 `view` 块（两组三维相机位置），正是 `ato/control/projection.py` 要的输入——
   屏幕↔格子可以直接算，不必从截图拟合单应性。另有 2,509 张模板图和现成 OCR 模型。
   **但 MAA 不含任何 sim-vs-real 配对轨迹**，标定那一次实机跑省不掉。
4. **一次计划驱动的实机对拍**，按 `docs/SIM_SPEC.md` §D 的协议。这是 1-7 脱离 `UNTRUSTED` 的唯一途径。
5. 之后才是模仿学习 → 搜索 → RL → 蒸馏。

`docs/SIM_SPEC.md` §E 是缺陷台账（E-6 命中时机、E-8 robust 判据只看通关率不看余量，两条未修），
§F 是当前基线数字。`docs/RECON.md` §7 有三件等你拍板的事。

**不要走的捷径**：为了跑通而静默忽略未知机制键；用 sim-vs-sim 一致性给关卡晋级信任；
在浮点里累加定时（见 `CLAUDE.md` §1）。

## 许可

AGPL-3.0-or-later。不含任何游戏素材或数据转储。
