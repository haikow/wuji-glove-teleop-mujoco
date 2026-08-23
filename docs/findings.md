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

## 8. 规模压测：链路能撑多少、瓶颈在哪一段

真机采集受人力限制（一条 10~15s 还要复位），测不出管线的规模边界。用
`tools/bench_pipeline.py` 合成 **1000 条 × 300 帧 = 30 万帧**压测（16 核 CPU，单进程管线）：

| 阶段 | 耗时 | 吞吐 | 备注 |
|---|---|---|---|
| QC | 11.2s | **89 条/s · 26731 帧/s** | 纯标准库单进程，1000/1000 通过 |
| 导出 LeRobot（优化前） | 125.5s | 2391 帧/s | — |
| 导出 LeRobot（优化后） | **43.8s** | **6846 帧/s** | 见 §11，产物逐位一致 |
| 峰值内存 | — | **815 MB** | 全程单进程，不随 episode 数线性增长 |
| 磁盘 | — | — | 730 MB JSONL → 179 MB parquet（**0.25×**） |

⚠️ **小规模测出来的导出速率会严重低估**：5 条 × 200 帧时只有 340 帧/s，因为被
`save_episode()` 的 per-episode 固定开销主导；到 1000 条时是 2391 帧/s，差 7 倍。
报吞吐必须说清样本规模。

**复现**：

```bash
./venv312/bin/python tools/bench_pipeline.py --episodes 1000 --frames 300 \
    --max-batches 60 --dataloader-workers 0 2 4 8
```

## 9. dataloader 取样吞吐：调参 6.73×

训练侧真正的瓶颈是取样速度，不是导出速度。在上面那份 30 万帧数据集上扫参：

| num_workers | batch_size | pin_memory | prefetch | samples/s |
|---|---|---|---|---|
| 0 | 256 | – | – | 13 059 ← **朴素默认** |
| 2 | 256 | False | 2 | 23 223 |
| 4 | 256 | False | 2 | 40 120 |
| 8 | 256 | False | 2 | 54 027 |
| 12 | 256 | False | 2 | 70 532 |
| 12 | 512 | False | 2 | 77 813 |
| 12 | 1024 | False | 2 | **87 844** ← 最快 |
| 12 | 1024 | True | 2 | 85 387 |
| 12 | 1024 | False | 4 | 79 102 |

**13 059 → 87 844 samples/s，6.73×。** 两个反直觉的点：

- `prefetch_factor` 从 2 调到 4 **反而掉 10~25%** —— 预取带来的内存压力盖过了收益。
- `pin_memory=True` 在纯 CPU 取样下是**负收益**（多一次锁页拷贝，却没有 GPU 传输来摊销）。
  只有真往 GPU 灌数据时才该开。

也就是说：这套数据规模下 dataloader 不是瓶颈（8.8 万 samples/s 远高于任何策略网络的
消费速度），**瓶颈在导出段**（2391 帧/s，单进程）。要继续优化应该先并行化导出。

**复现**：

```bash
./venv312/bin/python tools/bench_pipeline.py --dataset <数据集目录> \
    --dataloader-workers 0 2 4 8 12 --sweep
```

## 11. 导出提速 2.9×：给没有图像的数据集"嵌入图像"

**怎么发现的**：§8 显示导出是链路里最慢的一段，但**先别急着并行化** —— 2391 帧/s 意味着
1 小时真机采集（43 万帧）只要 3 分钟，到 100 小时才开始疼。先 profile 再动手。

`cProfile` 结果（60 条 × 300 帧）：

```
save_episode              17.9s
└─ _save_episode_data     15.8s
   └─ embed_images        14.4s   ← 占 77%
      └─ datasets.map(embed_table_storage, batched=False)
         apply_function        被调 18000 次（逐帧）
         from_arrow_schema     被调 18180 次（每帧重建一次 Features）
```

**根因**：`lerobot/datasets/io_utils.py: embed_images()` **无条件**跑
`dataset.map(embed_table_storage, batched=False)`。而我们的数据集
`use_videos=False`、features 只有 `observation.state` / `action` / `timestamp_dev`，
**一个 Image/Audio 列都没有** —— 这份逐行 + 逐行重建 schema 的工作是纯空转。

**改法**：在调用点按**实际列类型**判断，没有媒体列才跳过；有相机就原样走官方实现，
所以以后加了图像也不会静默丢数据（`tools/export_dataset.py:
skip_embed_images_when_no_media`，`--no-skip-embed` 可关掉做对比）。

**效果**：

| 数据 | 优化前 | 优化后 | 倍数 |
|---|---|---|---|
| 真机 19 条 / 22350 帧 | 2887 帧/s | **12782 帧/s** | 4.4× |
| 合成 1000 条 / 30 万帧 | 2391 帧/s | **6846 帧/s** | 2.9× |

**正确性**：同一批数据两条路径各导一次，抽检 231 帧，`observation.state` /
`action` / `timestamp_dev` 三个张量最大差异 **0.000e+00**，`meta/stats` 一致，
产物大小同为 14.81 MB。安全边界有单测守着（有 Image 列时必须不跳）：
`tests/test_export_label.py: SkipEmbedImagesTest`。

**这是上游可以改的**：`embed_images` 加一个"无媒体列直接返回"的短路，对所有
纯本体感（state/action）数据集都有效；顺带 `batched=False` 改成 `batched=True`
对有图像的场景也该有明显收益。

## 12. 怎么知道导出慢了

不用专门跑压测：`export_dataset.py` 每次导出都会打印耗时与帧率，并写进
`wuji_provenance.json` 的 `export_seconds` / `export_frames_per_s` /
`skipped_embed_images`；低于 500 帧/s 会直接告警。拿它和 §8 的基准比即可。

## 13. 训练侧：IO→compute 的临界点在哪

`tools/bench_train.py`。只测 BC baseline 那个 0.1M 的 MLP 是没意义的——它在任何现代
GPU 上都空转，结论永远是"GPU 利用率个位数"。扫三档规模才能定出**瓶颈翻转的临界点**，
那个点决定了要不要上 AMP、batch 开多大、worker 给几个。

环境：RTX 2060（6GB, **compute 7.5 / Turing**），22350 帧数据集，obs=108 → action=20。

| 模型 | 参数 | 朴素基线¹ | 调优后 | 倍数 | 最优配置瓶颈 | GPU 利用率 | 峰值显存 |
|---|---|---|---|---|---|---|---|
| mlp | 0.10M | 10 502 | 73 985 | 7.04× | **IO**（89% dataload） | 8.4% → 6.0% | 24 MB |
| big | 12.85M | 8 104 | 39 835 | 4.92× | **IO**（63% dataload） | 15.6% → 26.3% | 276 MB |
| xl | 118M | 4 005 | **14 694** | 3.67× | **compute** | 59.8% → 74.5% | 1955 MB |

¹ 朴素基线 = `num_workers=0` + 无 AMP + batch 256，很多人默认就这么写。单位 samples/s。

**临界点在 13M~118M 参数之间**。低于它，GPU 是空的，砸显卡没用，钱该花在 CPU 核数和
数据格式上；高于它才轮到 AMP / fused optimizer / compile。

**AMP 的收益完全跟着瓶颈走**（同 batch=1024 下 fp16 vs fp32）：

| 模型 | fp32 | fp16 | 收益 |
|---|---|---|---|
| mlp（IO 绑死） | 71 627 | 73 985 | **+3%**（白开） |
| big | 29 622 | 39 835 | +34% |
| xl（compute 绑死） | 6 449 | 13 090 | **+103%** |

⚠️ **Turing 必须用 fp16 不能用 bf16**：`torch.cuda.is_bf16_supported()` 在 compute 7.5
上返回 True，但没有原生 bf16 张量核，走模拟路径反而更慢。脚本按 capability 自动选
（`>= 8.0` 用 bf16，否则 fp16），`--amp-dtype` 可强制。

**fused Adam**：xl 上 optimizer 一度占 44.3% 的步时间——1.18 亿参数的逐张量 elementwise
kernel 太多。换 `Adam(fused=True)` 后：

```
12 887 → 14 694 samples/s (+14%)    optimizer 占比 44.3% → 27.1%    峰值显存 2379 → 1955 MB (-18%)
```

**复现**：

```bash
./venv312/bin/python tools/bench_train.py --dataset data/datasets/finger_tap \
    --repo-id local/wuji_finger_tap --models mlp big xl --batch-sizes 256 1024 --fused-adam
```

## 14. 推理侧：策略能不能塞进遥操回路

遥操的时间预算是硬的：手套 **120Hz → 每帧 8.33ms**；端到端（出帧→retarget→zenoh→
伺服→反馈）实测 **33ms**（§4）。所以只测 **batch=1 的尾延迟**——逐帧同步调用，
决定掉不掉帧的是 p99 不是均值。

`tools/bench_infer.py`，BC baseline（0.099M 参数），2000 次采样：

| 部署 | p50 | p95 | p99 | max | 占帧预算(p99) |
|---|---|---|---|---|---|
| **CPU** | 0.026 | 0.036 | **0.037 ms** | 0.057 | **0.4%** |
| CUDA | 0.060 | 0.067 | 0.094 ms | **1.402** | 1.1% |

（端到端口径：numpy → tensor → H2D → forward → D2H → numpy，不是只测 forward。）

**结论：CPU 完胜 GPU。** 小模型上一次 H2D + kernel launch 的固定开销盖过计算，
而且 GPU 的**最坏延迟差 20 倍**（1.40ms vs 0.071ms）——实时回路里尾延迟才是要命的。
同理 fp16 在 batch=1 上也是负收益（p50 0.084 vs 0.050，autocast 开销大于收益）。

余量 **224×**，叠加到 33ms 端到端上只 +0.1%。**这个策略可以直接塞进回路。**
反推可用预算：只要单帧 p99 < 4ms（半个帧预算）都算安全，对应模型规模远大于当前 baseline。

**复现**：`./venv312/bin/python tools/bench_infer.py --model data/models/bc_tap.pt`

## 10. CI 抓到的可移植性问题

无头 runner 上缺 EGL 时，`import mujoco` 会在 **import 期**抛
`AttributeError: 'NoneType' object has no attribute 'eglQueryString'`，
**不是 `ImportError`** —— 只接 `ImportError` 的 skip 守卫会漏掉，测试直接变 error。
修法：守卫接 `Exception`；CI 装 `libegl1/libgl1/libglx-mesa0` 让相关用例真的跑起来
（skip 数从 4 降到 2）。

## 容量参考

- `joint_states` ~999Hz 是大头：obs.mcap 约 **0.7 MB/s**，即 **2.5 GB/小时/手**（LZ4 后）
- 不录真机反馈时（只有手套 120Hz 两路）：15s 约 1.3MB
- Rerun `.rrd`：15s episode 含 26 块网格约 9MB（网格静态 log 一次，不随帧数增长）
