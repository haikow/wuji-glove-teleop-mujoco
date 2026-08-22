# 实测结论与复现方式

这一页把仓库里出现过的每个数字，连同**测量方法**和**复现命令**列出来，便于核对。
所有数字都来自真机（手套 WG1K 系列 + 二代手 WH2K 系列，SN 见各自 episode 的 `meta.json`），
采集数据本身不入库（`data/` 已 gitignore），所以复现需要自己先采一批。

判据实现都在 `tools/qc_episode.py` 和 `tools/check_model_fit.py` 里，
并有离线单测覆盖（合成夹具，不需要硬件）：`tests/test_mcap_join.py`、`tests/test_qc_episode.py`。

---

## 1. 每条 episode 开头有 0.24~0.33s 的假空档

**现象**：不做预热时，每条 episode 的第一帧到第二帧之间，**设备时间戳**凭空拉开
0.24~0.33s，QC 判 `gap` + `dropout`。之后 900+ 帧 dt 恒定 8.3ms、seq 零缺口。

**根因是两个，叠在一起**：

1. 订阅后数据流本身有一段不稳定期；
2. **首次 `RetargetSession.step()` / 首次 render 是冷路径**，处理第一帧就吃掉 ~0.24s，
   这段时间设备又产了约 29 帧，于是头两帧的设备时间戳被拉开。

只治第一个不够：改成固定时长预热后仍残留 0.242s（紧贴 0.25s 阈值）。

**修法**：按**收到的帧数**预热（首帧本身可能晚于任何固定时长窗口），并在预热阶段
把 `sess.step()` / render 跑热；`recorder.start()` 之后再丢一帧（那帧是 start 之前
就躺在订阅队列里的，obs 没进 MCAP，为它写的 action 会成为 join 不上的孤儿）。

**效果**（三条 episode 的首帧 dt）：

| | 无预热 | 按时长预热 | 按帧数预热 + 预跑计算路径 |
|---|---|---|---|
| first_dt | 0.267 / 0.333 | 0.242 | 0.050 / 0.008 / 0.008 |
| dt_max | 0.333 | 0.242 | 0.050 / 0.025 / 0.017 |
| dropout | 3.23% | 1.04% | 0.00% |

**复现**：`record_episode.py --warmup-frames 0` 与默认（60）各录三条，对比
`quality.json` 里的 `dt_max_s` / `dropout_ratio`。

---

## 2. 二代手用一代 MJCF：过半帧越界

**现象**：`--hand-model wuji_hand_2` 时 retarget 输出落在**二代**关节限位内，
拿本仓库 vendor 的一代 MJCF 去 clip，大量帧越界 —— 也就是说，指令曾被**错误的限位
裁剪之后**才发给真机。

**判据**：逐帧逐关节算 `max(lo - q, 0) + max(q - hi, 0)`，统计越界最大值与越界帧占比。

**实测**（两批数据口径不同，都列出来）：

| 数据集 | 一代 MJCF 越界最大 | 越界帧占比 | 二代 MJCF |
|---|---|---|---|
| 3 条自由挥手（`hand_teleop_calib`） | 0.526 rad (30°) | 63.7% | 0.0000 / 0.0% |
| 23 条（含 20 条逐指对指任务） | 0.974 rad (56°) | 56.5% | 0.0000 / 0.0% |

占比随动作内容变化（幅度越大越容易顶到一代的窄限位），但**二代 MJCF 恒为零越界**这一点
在两批数据上都成立。hand2 的 beta1 与 beta2 的 20 个关节限位完全一致，只差几何。

**复现**：

```bash
tools/fetch_hand2_description.sh
./venv312/bin/python tools/check_model_fit.py data/episodes
./venv312/bin/python tools/check_model_fit.py data/episodes --field hand_state
```

---

## 3. `hand_model_path()` 不能用来判断标定是否生效

**现象**：默认用户下 `g.hand_model_path().get()` 返回一个路径，看起来像"当前加载的模型"；
切到具名标定用户后**反而抛 `Path not found: calibration.hand_model_path`**。

**根因**：它是**覆盖槽**（stub 原文 "Custom hand URDF path accessor for online IK"），
不是"当前用的是哪套模型"。默认用户下有值只因 legacy 路径被设成了 override。

**正确判据**：`WujiGlove.offline_pipeline(sn, side).urdf_source`，取值
`builtin_default` / `calibration_file` / `override`。实测：

```
默认用户        urdf_source=builtin_default   path=None
具名标定用户    urdf_source=calibration_file  path=.../users/<uid>/models/...urdf
```

这同时正面印证了"默认用户忽略标定、回落内置 URDF"。录制器把 `urdf_source` /
`urdf_source_path` 写进 `meta.json`，导出时按它分组。

**复现**：`tools/calibrate_glove.py` 末尾会打印这两个值；或直接
`WujiGlove.offline_pipeline(sn, side).urdf_source`，切换用户前后各读一次。

---

## 4. 端到端遥操延迟与跟踪误差（把两者解耦）

**方法**：真机 `joint_states`（~999Hz）按时间戳并到手套帧（120Hz）上之后，
对"指令 vs 实测"扫描 0~200ms 的滞后，取误差最小值。零滞后的 MAE 里混着通信+伺服延迟，
**扫描后的最小值才是真正的跟踪偏差，而取到最小值的那个滞后就是端到端延迟估计**。

实现在 `tools/qc_episode.py`（`track_mae_rad` / `track_mae_best_rad` /
`track_best_lag_ms`），单测 `test_tracking_lag_is_recovered` 注入 5 帧滞后后能准确还原。

**实测**：

| 配置 | 零滞后 MAE | 扣滞后 MAE | 最优滞后 | clip_max |
|---|---|---|---|---|
| 默认用户 + 一代 profile + 一代 MJCF | 0.032~0.058 | 0.0241~0.0409 rad (1.38~2.35°) | 5 帧 / 42ms | 0.47~0.53 |
| 标定用户 + 二代 profile + 二代 MJCF | 0.028~0.040 | **0.0155~0.0178 rad (0.89~1.02°)** | **4 帧 / 33ms** | **0.0** |

对照组：手**不使能**时同样的指标是 MAE 0.597 rad、滞后 17 帧（噪声值）—— 说明这个指标
确实在区分"手跟没跟"，不是恒定输出。

**复现**：`record_episode.py --hand-sn <SN>` 录制后看 `quality.json`。

---

## 5. obs/action join 覆盖率不是 100%

retarget 回路用"取最新帧"主动跳过积压以免累积延迟，而 MCAP 录全部帧，
两边帧集不完全相同。实测覆盖 **98.1%~98.4%**。低于 `min_action_join_ratio`（默认 0.9）
判 fail —— 不允许悄悄导出半份数据。

`recorder.start()` 之后不丢那一帧的话，每条 episode 恰好多出 1 条 join 不上的孤儿 action。

---

## 6. 数据量对 BC baseline 的影响

同一套链路，只换数据集：

| 数据集 | episode | train/val | train_mse | val_mse | val_MAE |
|---|---|---|---|---|---|
| 自由挥手 | 3 | 2/1 | 0.0016 | 0.0253 | 1.64° |
| 逐指对指任务 | 19 | 15/4 | 0.00083 | 0.0070 | **0.726°** |

过拟合缺口 15.6× → 8.4×。`val_MAE` 在 24~30 epoch 触底后回升（早停点）。
留出 episode 上策略回放与录制动作逐帧吻合（MAE 0.88°）。

**这只是格式验证，不是建模成果** —— 19 条 episode 的规模下 8.4× 的过拟合缺口说明
模型基本在记训练集。它的作用是证明导出的数据能被训练栈消费。

**复现**：`tools/flywheel_once.sh` 后 `tools/train_bc.py --val-episodes 4`。

---

## 7. SDK 录制里的数据是全的

官方 `examples/python/wuji_glove/2.recording.py` 三个 topic 的实测结构：

| topic | 频率 | frame_id | 内容 |
|---|---|---|---|
| `hand_skeleton` | ~120Hz | `r_wrist` | 21 关节 position + orientation + confidence |
| `emf_poses` | ~120Hz | `r_hand_emf_tx` | 5 指尖接收线圈 6-DoF + confidence |
| `tactile` | ~120Hz | — | 744 值（24×31，-1 = 无效/屏蔽） |

四元数是真数据：`|q|` 恒为 1，21 个关节里只有 `wrist` 和 `thumb_cmc` 恒为单位四元数，
其余 19 个带真实旋转。

⚠️ **两个 topic 坐标系不同**（腕系 vs EMF 发射器系）。要接相机做同步渲染，
还缺发射器→相机的外参，那不在录制数据里。

**复现**：`tools/viz_mcap.py <file>.mcap`，或直接用 `mcap` 库读 JSON payload。

---

## 容量参考

- `joint_states` ~999Hz 是大头：obs.mcap 约 **0.7 MB/s**，即 **2.5 GB/小时/手**（LZ4 后）
- 不录真机反馈时（只有手套 120Hz 两路）：15s 约 1.3MB
- Rerun `.rrd`：15s episode 含 26 块网格约 9MB（网格静态 log 一次，不随帧数增长）
