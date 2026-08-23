# wuji-glove-teleop-mujoco

用 **Wuji Glove → SDK 内置 retargeting → MuJoCo** 实时预览灵巧手遥操效果。
无手套时可用录制数据回放；也提供一套 ROS2 两节点管线（retarget 节点发 topic → MuJoCo 订阅）。

> 📊 **实测结论与复现方式**见 [`docs/findings.md`](docs/findings.md) —— 每个数字都附测量方法、
> 数据口径和复现命令（假空档成因、手型-模型不匹配越界率、标定生效判据、
> 端到端延迟与跟踪误差解耦、数据量对 BC 的影响、容量参考）。

> 📖 **客户使用指导**（标定 URDF 加载 + 数据录制的完整排查步骤）见
> [`docs/customer-guide-calibration-and-recording.md`](docs/customer-guide-calibration-and-recording.md)。
> 关键结论：**默认用户永远回落内置 URDF、忽略标定；要加载标定必须用具名用户。**

> **重要说明**：本仓库**不包含任何 retargeting 算法实现**，只调用 `wuji-sdk` 内置重定向，
> 与官方示例 `wuji-sdk/examples/python/retargeting/1.teleop_real.py` 同一套算法。本仓库是一套
> **ROS2 / MuJoCo 集成方案** + 真机遥操脚本。
>
> **retargeting 打包随版本变化（脚本已用 `retarget_compat.py` 兼容两者，无需手动区分）**：
> - **2026.7.21 / 8.3 起**：retargeting 已**原生化**，`HandModel` / `RetargetSession` **顶层导入**
>   （`from wuji_sdk import HandModel, RetargetSession`），**不再需要 `[retarget]` extra**，
>   `pip install "wuji-sdk"` 即可。
> - **≤ 2026.7.15**：在 `wuji_sdk.retargeting` 子模块，需 `pip install "wuji-sdk[retarget]"`。

## 环境（重要）

**Python 版本随 SDK 版本**：`wuji-sdk` **2026.7.21 / 8.3 起** retargeting 原生化，提供 **cp312 / cp313 / cp314**
wheel（Python **3.12 / 3.13 / 3.14 均可**，实测 3.12 / 3.14 下整条真机遥操链路——连接 → retarget → 驱动一代/二代手——均正常）；
**≤ 7.15** 只有 cp312、需 Python 3.12。3.12 同时是 ROS2 Humble/Kilted 的 ABI，一套环境通吃，故本仓仍以 3.12 为默认。

```bash
python3.12 -m venv --system-site-packages venv312
./venv312/bin/pip install -U pip
./venv312/bin/pip install -r requirements.txt
```

仅 Linux x86_64/aarch64（retargeting 编译扩展不支持 macOS/Windows）。

### Docker（ROS2 部署推荐，含二代手以太网驱动栈）

**核心规则：`rclpy` 必须与 Python 版本严格匹配**，所以看 ROS2 发行版而不是随便配 Python：
Humble=Python 3.10、Kilted/Jazzy=Python 3.12。`wuji-sdk` 提供 **cp310–cp314** wheel（`manylinux_2_34`，
glibc≥2.34，22.04/24.04 都满足），所以**两条路都成立，二选一**：

| 选项 | 栈 | Dockerfile | 何时选 |
|---|---|---|---|
| **A（最省事）** | Ubuntu 22.04 / Humble / **Python 3.10** | [`docker/Dockerfile.humble`](docker/Dockerfile.humble) | 已在用 `wuji-hand-teleop`（22.04/Humble）→ **不用升级**，加 `pip install wuji-sdk`(cp310) 即可 |
| **B（新栈）** | Ubuntu 24.04 / Kilted(或 Jazzy) / **Python 3.12** | [`docker/Dockerfile`](docker/Dockerfile) | 新部署 / 想用较新 ROS2 |

> ⚠️ **不要做「Ubuntu 22.04 + Python 3.12」**：Humble 的 rclpy 是给 3.10 编的，22.04 上没有 3.12 的预编译
> ROS2（Kilted/Jazzy 才是 3.12，但基于 24.04），硬上得在 22.04 源码重编整个 ROS2，不值当。要 3.12 就整套上 24.04（选项 B）。

```bash
# 选项 A（Humble/22.04/3.10，留在现有 base，最省事）
docker build -t wuji-teleop:humble -f docker/Dockerfile.humble .
docker run --rm -it --network host wuji-teleop:humble
#   source /opt/ros/humble/setup.bash

# 选项 B（Kilted/24.04/3.12）
docker build -t wuji-teleop:kilted -f docker/Dockerfile .
docker run --rm -it --network host wuji-teleop:kilted
#   source /opt/ros/kilted/setup.bash

# 两者容器内跑法相同（--network host：二代手以太网 zenoh/UDP 发现 + DDS；USB 手套/一代手再加 --device /dev/bus/usb）：
python3 ros2_realhand_node.py --side right --hand-sn <二代手SN> --prefix "" --diagnostics
```

> 选项 B 想要 LTS 支持周期更长，可把基础镜像换成 `ros:jazzy-ros-base`（同为 24.04 / Python 3.12），节点代码不变。
>
> ⚠️ 两个镜像都**只装 wuji-sdk**（ROS2 驱动栈）。**不含 MuJoCo 仿真**：mujoco/opencv 需要 numpy≥2，会顶掉
> ROS2 依赖的 numpy（Humble 1.21 / Kilted 1.26）并破坏 rclpy。MuJoCo 仿真预览（`glove_teleop_live.py` 等）
> 请用上面「环境」小节的 **独立 non-ROS venv** 跑，与本 ROS2 驱动镜像分开。
>
> **两镜像均已实测（build → 容器内 `--network host` 真机驱动全程）**，连以太网二代手
> （WH2JA…@192.168.1.110:7447）→ 使能 → 订阅 `joint_commands` 驱动真手 → 回发 `joint_states` **~100Hz**
> → 首帧平滑过渡：
> - **A · Humble**：镜像 1.21GB，Python 3.10.12 / wuji_sdk 2026.8.3 / numpy 1.21.5
> - **B · Kilted**：镜像 1.38GB，Python 3.12.3 / wuji_sdk 2026.8.3 / numpy 1.26.4

## 用法

### 实时手套遥操（看效果）

**左手 / 右手启动**：手型由 `--side` 决定（留空则自动读手套 `hand_side()`），
手套 `--glove-sn` 换成你现场那只的 SN（脚本默认值仅示例，通常要覆盖）。

```bash
# 左手（加载 mjcf/left.xml，SDK 内部自动做左手 Y 镜像）
MUJOCO_GL=egl ./venv312/bin/python glove_teleop_live.py \
    --glove-sn <左手手套SN> --side left --record left.mp4 --seconds 20

# 右手（加载 mjcf/right.xml）
MUJOCO_GL=egl ./venv312/bin/python glove_teleop_live.py \
    --glove-sn <右手手套SN> --side right --record right.mp4 --seconds 20
```

其它开关（左右手通用）：

```bash
# EGL 离屏 + 录像（无显示器/CI/桌面合成器不稳时，最稳的看效果方式）
MUJOCO_GL=egl ./venv312/bin/python glove_teleop_live.py --side left --record out.mp4 --seconds 20

# 叠加"人手输入骨架"对齐对比（橙=手套输入，白=机器人FK，对齐到 palm_link）
# 现已支持 EGL 离屏路径，不再必须 GLFW，可直接录像/截图：
MUJOCO_GL=egl ./venv312/bin/python glove_teleop_live.py --side left --show-input --record align.mp4 --seconds 20

# 官方 MuJoCo 交互窗口（GLFW，可轨道/接触可视化；需要可用 GLX 的桌面）
./venv312/bin/python glove_teleop_live.py --side left --viewer --show-input

# 二代手模型
... --hand-model wuji_hand_2
```

> 注：GLFW 交互窗口和 cv2 on-screen 窗口都依赖桌面 GLX/合成器；在坏 GLX 或合成器抖动的
> `DISPLAY` 上会崩。要稳定看效果就用 `MUJOCO_GL=egl ... --record`（EGL 纯离屏，不碰桌面）。

### 真机遥操（驱动真手：一代 / 二代，自动识别）

把 retarget 结果直接下发到**真手**（不是仿真）。一代 Wuji Hand 走 `realtime_controller`，
二代 Wuji Hand 2 走 MIT `joint_command`，脚本按连接到的设备类型自动选择。

> ⚠️ **安全**：会**使能真手并让它实时跟随你的手运动**。跑前把手固定、周围无障碍、急停/断电在手边；
> Ctrl+C 或到时自动 disable。

```bash
# 自动扫描；内置 URDF；跑到 Ctrl+C
./venv312/bin/python glove_teleop_realhand.py --side right

# 用你自己标定的 per-user 手模型（连接前不切默认用户，加载 users/<uid>/models/right_hand.urdf）
./venv312/bin/python glove_teleop_realhand.py --side right --keep-user --seconds 20

# 总线上同时有一代+二代手时，用 --hand-sn 指定目标手（一代手 SN 形如 347A…）
./venv312/bin/python glove_teleop_realhand.py --hand-sn <目标手SN> --glove-sn <手套SN> --side right
```

- **默认**切到内置默认 URDF（跟随更稳，照官方 `1.teleop_real.py`）；**`--keep-user`** 才用你的标定。
- 是否加载了标定，看连接日志：`online hand_model: loading user hand model from …/right_hand.urdf`
  = 已加载；`using built-in default URDF` = 内置。

### 采集手套 + 触觉原始数据（CSV/JSONL）

```bash
# hand_skeleton(21×3) + hand_joint_angles + tactile，带主机时间戳，落盘 CSV 宽表 + JSONL
./venv312/bin/python record_glove_data.py --glove-sn <手套SN> --seconds 20 --out rec
```

> 这条路径**只有 obs、没有 action**，适合看原始信号，不适合做模仿学习数据集。
> 要训练数据请走下面的数据飞轮（`record_episode.py`）。

计算节点发 `/{side}_hand/joint_commands`（`sensor_msgs/JointState`，position[20]，固件关节序）；
换真机时把 MuJoCo 订阅端（`ros2_mujoco_hand_node.py`）替换成真机驱动节点即可，计算节点不动。

### 真机·二代手 ROS2 驱动节点（`ros2_realhand_node.py`）

二代手（Wuji Hand 2）**目前没有官方 ROS2 包**（`wuji-ros2` 仍在设计阶段；`wujihandros2` 只支持一代手），
所以用一个很薄的 rclpy 节点把“非 ROS2 的手”封装成标准 ROS2 接口——它与仿真 sink 可互换：

> ⚠️ **Python 版本必须与 ROS2 的 rclpy 对齐（易踩坑）**：`rclpy` 是给某个具体 Python 编的
> （如 Kilted 是 `python3.12`，`/opt/ros/<distro>/lib/python3.12/site-packages/rclpy`）。若系统默认
> `python3` 已升级到别的版本（比如 3.14），直接 `python ...` 会 `import rclpy` 失败。**用与 rclpy 匹配的
> `python3.12` 跑**，并把装了 `wuji_sdk` 的环境的 `site-packages` 挂到 `PYTHONPATH`，让同一进程同时拿到
> `rclpy` 和 `wuji_sdk`。（cp312 = Humble/Kilted 的 ABI，一套 env 通吃。）

```bash
source /opt/ros/<distro>/setup.bash

# 让 python3.12 同时看到 ROS 的 rclpy + 你装了 wuji_sdk 的环境
#   若 wuji_sdk 装在某个 venv 里，把它的 site-packages 加进来：
export PYTHONPATH="/path/to/venv/lib/python3.12/site-packages:$PYTHONPATH"
python3.12 -c "import rclpy, wuji_sdk, sensor_msgs.msg; print('env OK')"   # 自检

# ① 计算节点照旧发 /{side}_hand/joint_commands（二代手用 --hand-model wuji_hand_2）
python3.12 ros2_retarget_node.py --side right --source glove --glove-sn <SN> --hand-model wuji_hand_2
# ② 真机驱动节点：订阅命令 → SDK 驱动真手；同时回发 /{side}_hand/joint_states
python3.12 ros2_realhand_node.py --side right --hand-sn <二代手SN>
```

- 订阅 `/{side}_hand/joint_commands`（`position[20]` 固件序，带 `velocity[20]` 则作 MIT 速度前馈）→
  `hand.joint_command().publish().send([JointCommand(pos, vel, 0), ...])`；回发实测角到 `/{side}_hand/joint_states`。
- **SDK 只在这个节点里用**，对 ROS2 侧是纯标准接口；首帧从当前实测位置平滑过渡到第一条命令，避免上电猛弹。
- 已在 **ROS2 Kilted + 真二代手** 上端到端验证：使能 → 订阅命令驱动真手 → 回发 `joint_states` ~100Hz → Ctrl+C 干净 disable。

#### 作为 `wujihandros2` 的二代手（以太网）即插替换

官方 `wujihandros2` 驱动是 `wujihandcpp::device::Hand(serial)` = **USB / 一代手**，**没有 UDP/以太网**，
连不了走以太网（zenoh/UDP）的二代手。本节点用 `wuji_sdk`（scan/connect 走以太网）补上这块，且**沿用
wujihandros2 的话题契约**（`joint_commands`/`joint_states` 均 `sensor_msgs/JointState`，可选
`hand_diagnostics` = `wujihand_msgs/HandDiagnostics`），所以能直接顶替：

```bash
# wujihandros2 单手默认（相对名）——用 --prefix "" 对齐
python3.12 ros2_realhand_node.py --side right --hand-sn <SN> --prefix "" --diagnostics
# wujihandros2 多手（按手性绝对名 /hand_<side>/）
python3.12 ros2_realhand_node.py --side right --hand-sn <SN> --prefix "/hand_right/" --diagnostics
```

- `--prefix` 对齐话题命名（默认 `/{side}_hand/`）；也可用 ROS2 remap `--ros-args -r`。
- `--diagnostics` 发 `hand_diagnostics`（`wujihand_msgs/HandDiagnostics`：温度/电压/error_code/enabled/
  effort_limit），供 Monitor GUI 消费；该消息包在 `wuji-hand-teleop`/`wujihandros2` 环境里已有，缺失则自动跳过。
- 二代手诊断字段映射自 `joint_diagnostics`（`mcu_temp_c_fb`→温度、`vbus_v_fb`→电压、`error_code_current`）。
- 已真机验证：`hand_diagnostics` 输出 `system_temperature≈65℃ / input_voltage≈12.4V` 等真实值。

### 接 ROS2 机械臂一起遥操

手（SDK-only）+ 臂（ROS2）联动的推荐结构：计算节点除了发手命令，再把手套腕部 6DoF 位姿
（`emf_poses`/`tf`）发成臂的目标位姿（`geometry_msgs/PoseStamped` 或 TF）→ 喂 **MoveIt Servo** /
笛卡尔控制器让臂跟随；手仍走上面的 `ros2_realhand_node.py`。连续遥操时手套流本身平滑、MIT 直接跟即可，
无需额外插补；只有下发**离散预设动作**才需要 Ruckig / 梯形加减速做轨迹。

## 数据飞轮（data flywheel）

```
采集 → episode 目录 → 自动质检 → 分流 clean/rejected → 导出 LeRobot → 训 BC 验证 → 再采集
```

一键跑完质检到导出（采集要人戴手套，不在脚本里）：

```bash
tools/flywheel_once.sh                 # 质检 → 分流 → 导出 → 读回自检
tools/flywheel_once.sh --task pick_cube
SKIP_EXPORT=1 tools/flywheel_once.sh   # 只质检
```

### 0. 依赖分层

采集和质检只要 `requirements.txt`（轻量）；导出 LeRobot 数据集和训 BC 才需要
`requirements-training.txt`（会拉 torch + CUDA，约 4GB）：

```bash
./venv312/bin/pip install -r requirements.txt              # 采集 + 质检
./venv312/bin/pip install -r requirements-training.txt     # 再加导出 + 训练
```

### 1. 先标定，否则 retarget 一定对不准

**默认 SDK 用户永远回落内置 URDF、忽略标定** —— 实测判据：

```
默认用户        urdf_source=builtin_default   path=None
jues_20260822  urdf_source=calibration_file  path=.../users/u_xxx/models/...urdf
```

所以标定必须落到**具名用户**下，遥操时再用 `--user` 切回去：

```bash
# 交互式，6 个姿势：4 个捏合 + 四指弯 90° + 摊平张开
./venv312/bin/python tools/calibrate_glove.py --glove-sn <手套SN> \
    --user-name <你的用户名> --hand-profile wujihand2
```

`--hand-profile` 要和真机手对齐：`wujihand`=一代，`wujihand2`=二代（SN 以 `WH2` 开头、
SDK 类型 `WujiHand2`）。`both` 则两套都生成。

> ⚠️ **`hand_model_path()` 不能当"当前用了哪套模型"的判据** —— 它是*覆盖槽*
> （"Custom hand URDF path accessor"），没设 override 时直接抛
> `Path not found: calibration.hand_model_path`。硬证据是
> `WujiGlove.offline_pipeline(sn, side).urdf_source`，取值
> `builtin_default` / `calibration_file` / `override`。录制器会把它写进 `meta.json`。

### 2. 二代手要换 MJCF

本仓库只 vendor 了一代 `wuji_hand_description/`。用 `--hand-model wuji_hand_2` 时，
retarget 输出落在**二代**关节限位内，拿一代 MJCF 去 clip 会错得很离谱 —— 实测
**63.7% 的帧越界、最大 0.53 rad（30°）**：

```
一代(本仓库)   越界最大=0.5262 rad  越界帧占比=63.7%
hand2_beta1   越界最大=0.0000 rad  越界帧占比=0.0%
hand2_beta2   越界最大=0.0000 rad  越界帧占比=0.0%
```

（beta1/beta2 的 20 个关节限位完全一致，只差几何，脚本取较新的 beta2。）

```bash
tools/fetch_hand2_description.sh    # → wuji_hand_description2/（约 18MB，不入库）

# 判定一批数据到底属于哪套手模型（上面那组数字就是这么来的）
./venv312/bin/python tools/check_model_fit.py data/episodes
```

拉下来后 `--hand-model wuji_hand_2` 会自动用它；没拉则回落一代并打印告警。

### 3. 采集：episode 化录制

```bash
# 录 5 条 10 秒 demo，每条录完提示打 success 标签
./venv312/bin/python record_episode.py --glove-sn <手套SN> --task pick_cube \
    --user <标定过的用户名> --hand-model wuji_hand_2 \
    --seconds 10 --episodes 5 --label

# 带真机手：使能 + 下发 joint_command + 把 joint_states 一起录进 MCAP
./venv312/bin/python record_episode.py --glove-sn <手套SN> --hand-sn <手SN> \
    --user <标定过的用户名> --hand-model wuji_hand_2 --task pick_cube
#   --no-enable 只录反馈不驱动；录制中输入 i + 回车标记一段人工介入；
#   退出时自动 disable。

# 顺带出 preview.mp4，方便事后人工过一遍
MUJOCO_GL=egl ./venv312/bin/python record_episode.py --glove-sn <手套SN> \
    --task pick_cube --video
```

产出（每条 episode 一个自包含目录）：

```text
data/episodes/ep_<YYYYMMDD_HHMMSS>_<side>/
    meta.json       # 身份/环境/标签
    obs.mcap        # SDK TopicRecorder 录的 obs（LZ4，Foxglove 可开）
    action.jsonl    # retarget action 旁路，按 header.seq 与 MCAP 对齐
    preview.mp4     # --video 时
    quality.json    # 质检后写入
```

### 4. 容器为什么是 MCAP + 旁路

**obs 走 `wuji_sdk.TopicRecorder`**：这是这套栈的 house format —— LZ4 压缩、自描述
jsonschema、Foxglove 直接打开，还自带 `QualityMetrics`（丢帧率/抖动/跨通道同步率）和
阈值告警，没必要自己再造一个 JSONL。

**action 只能走旁路**：`TopicRecorder.record()` 的入参是 `Subscription`（设备话题），
而 retarget action 是 host 侧算出来的、没有对应的设备资源路径（`publish()` 是往设备发、
路径必须已存在），**存不进那个 MCAP**。所以单独落 `action.jsonl`。

**join 键是 `hand_skeleton.header.seq`**，不是时间戳。实测依据：

- 同一资源开两个订阅（一个给录制器、一个给 retarget 回路）看到的 seq 完全一致（361/361）
- `hand_skeleton` 与 `hand_joint_angles` 共用同一 seq 空间、时间戳 diff = 0µs

所以 join 是精确的。读回时 `tools/episode_format.iter_frames()` 把两个容器合成统一帧结构，
下游看到的东西和自包含 JSONL 版本完全一样（老的 `frames.jsonl` episode 仍然能读）。

其余三个设计要点：

- **开录前预热（`--warmup-frames`，默认 60）**：真机实测，不预热每条 episode 开头都有一个
  0.24~0.33s 的假 gap，被 QC 判 `gap`+`dropout`。成因有两个，都要治：订阅后数据流本身有
  一段不稳定期；**首次 `sess.step()` / 首次 render 是冷路径**，处理第一帧吃掉 ~0.24s，
  期间设备又产了约 29 帧，于是头两帧的设备时间戳凭空拉开。预热同时空转 drain 和跑
  `sess.step()`，正式录制就是纯 8.3ms（120Hz）稳态。另外 `recorder.start()` 之后要再丢一帧
  —— 那帧是 start 之前就躺在队列里的，obs 没进 MCAP，会变成 join 不上的孤儿 action。
- **join 覆盖率不是 100%**：retarget 回路用"取最新帧"主动跳过积压以免累积延迟，实测覆盖
  98.1%~98.4%。低于 `min_action_join_ratio`（默认 0.9）判 fail，不允许悄悄导出半份数据。
- **action 不做本地 clip**：retarget 已经在目标手型自己的 URDF 限位内解算，输出直接下发；
  MJCF clip 只用于 MuJoCo 显示。`action_raw_max_ovr` 记录超出本地 MJCF 多少 ——
  它非零就说明 MJCF 和手型不匹配（QC 打 `action_clipped` 警告）。
- **meta 记标定身份**：retarget 输出强依赖加载了哪套手 URDF。`meta.json` 存 `user_id` /
  `calibrated` / `urdf_source` + `urdf_source_path`（**硬证据，导出按它分组**）/ `mjcf` /
  `sdk_version` / `sdk_quality` / `hand`（真机 SN、在线关节数、是否使能）。

### 真机手跟踪误差

给了 `--hand-sn` 后，`joint_states`（~999Hz）也进同一个 MCAP，读回时按时间戳并到帧上。
QC 会把**滞后**和**跟不动**分开：扫描 0~200ms 的滞后取误差最小值，最优滞后本身就是
端到端延迟估计。实测（标定 + 二代模型 + 二代 MJCF）：

```
track_mae(零滞后)=0.036~0.040   track_mae(扣滞后)=0.0161~0.0178 rad (0.92~1.02°)
最优滞后=4 帧/33ms              clip_max=0.0
```

对照组（默认用户 + 一代模型 + 一代 MJCF）：扣滞后 0.0241~0.0409 rad（1.38~2.35°），
滞后 42ms，clip_max 0.47~0.53 rad。**标定 + 正确手型让跟踪误差差不多减半。**
BC baseline 的验证误差也从 2.62° 降到 1.64°。

**数据量的影响**（同一套链路，只换数据集）：

| 数据集 | episode | train/val | train_mse | val_mse | val_MAE |
|---|---|---|---|---|---|
| 自由挥手 | 3 | 2/1 | 0.0016 | 0.0253 | 1.64° |
| 逐指对指任务 | 19 | 15/4 | 0.00083 | 0.0070 | **0.726°** |

过拟合缺口 15.6× → 8.4×，验证误差减半。`val_MAE` 在 24~30 epoch 触底（0.726°）后回升，
再训就是过拟合 —— 这个规模下 30 epoch 早停即可。留出集上回放与录制动作逐帧吻合（MAE 0.88°）。

### 5. 质检：自动打分 + 分流

```bash
python tools/qc_episode.py data/episodes                # 只写 quality.json
python tools/qc_episode.py data/episodes --route link   # 顺便分流 clean/rejected
python tools/qc_episode.py data/episodes --set min_rate_hz=30 --strict
python tools/qc_episode.py data/episodes --print-thresholds
```

判 **fail** 的 flag：`too_short` / `too_long` / `low_rate` / `gap` / `dropout`（按 seq 跳号）/
`duplicate_seq` / `action_jump` / `near_static` / `nonfinite` / `no_action` /
`low_action_join` / `empty`。仅记录不拦截的 **warning**：`low_confidence` /
`action_clipped`（顶到限位）/ `tactile_dead` / `tactile_saturated` / `clock_backwards` /
`host_clock_only` / `no_device_seq`。SDK 自己算的 `drop_rate`/`sync_rate` 会原样带进
`quality.json` 的 `metrics.sdk_quality`，可以和我们算的交叉验证。

### 6. 标注

```bash
python tools/label_episode.py data/episodes --list      # 看标注状态
python tools/label_episode.py data/episodes --review    # 逐条过（带 QC 摘要 + preview 路径）
python tools/label_episode.py data/episodes/ep_xxx --success y
```

### 7. 导出 LeRobot 数据集

LeRobot **本身就是 parquet** —— v3 布局 = `data/chunk-000/*.parquet` +
`meta/info.json`（schema/fps/path 模板）+ `meta/stats.json`（归一化）+
`meta/tasks.parquet` + `meta/episodes/`（边界与分片索引）。这套分片索引手写容易错，
所以直接用官方 `LeRobotDataset.create()/add_frame()/save_episode()` 写，格式由库负责。

```bash
./venv312/bin/python tools/export_dataset.py --input data/clean \
    --repo-id local/wuji_pick_cube --out data/datasets/pick_cube
./venv312/bin/python tools/export_dataset.py --verify data/datasets/pick_cube \
    --repo-id local/wuji_pick_cube
```

- `observation.state` = skeleton(63) + joint_angles，`--obs skeleton|joints|both`
- `action` = 20 维 retarget 目标关节角
- 额外存一列 `observation.timestamp_dev`：LeRobot 按 `frame_index/fps` 生成均匀
  timestamp，而我们的帧列有约 2% 空洞，把设备时钟原样留一列，事后能查真实间隔
- 另写 `wuji_provenance.json`：哪些 episode、什么手模型、什么 SDK 版本进了这个数据集
- **溯源分组守卫**：默认按 `(task, side, hand_model_path, action_dim, obs_dim)` 分组，
  发现多于一组直接报错退出（返回码 2），要混必须显式 `--allow-mixed`

### 8. 消费端：BC baseline

这不是要做 SOTA 策略，而是**唯一能证明导出的数据真能被训练栈吃进去**的手段 ——
QC 全绿只说明数据自洽，不说明格式对。

```bash
./venv312/bin/python tools/train_bc.py --dataset data/datasets/pick_cube \
    --repo-id local/wuji_pick_cube --epochs 20 --out data/models/bc.pt

# 把预测的 action 放回 MuJoCo，肉眼看手动得对不对
MUJOCO_GL=egl ./venv312/bin/python tools/viz_episode.py data/episodes/ep_xxx \
    --policy data/models/bc.pt
```

验证集按 **episode** 切，不按帧切 —— 同一条 demo 的相邻帧几乎一样，帧级切分会让验证集
泄漏训练集内容，误差看起来好得离谱。

### 9. 回看

```bash
MUJOCO_GL=egl ./venv312/bin/python tools/viz_episode.py data/episodes/ep_xxx
MUJOCO_GL=egl ./venv312/bin/python tools/viz_episode.py data/episodes/ep_xxx \
    --policy data/models/bc.pt --out policy.mp4     # 策略回放 + 逐关节误差
./venv312/bin/python tools/viz_episode.py data/episodes/ep_xxx --stats-only

# Rerun：3D 手骨架 + 逐关节 指令/实测/跟踪误差/策略预测 时间序列
./venv312/bin/python tools/viz_episode.py data/episodes/ep_xxx --rerun
./venv312/bin/python tools/viz_episode.py data/episodes/ep_xxx --rerun \
    --policy data/models/bc.pt          # 再多一层 policy/<joint> 曲线
rerun data/episodes/ep_xxx/episode.rrd  # 本地打开
```

`--rerun` 落 `.rrd` 文件而不是开窗 —— 这台机器常跑无头 EGL，落盘后在有显示器的机器上打开。

3D 视图里三层叠在同一坐标系，可直接比对：

| 实体 | 内容 |
|---|---|
| `/world/robot_cmd` | 指令位姿下的机器人手，**MJCF 真网格**（二代 26 块 link，不是点云） |
| `/world/robot_real` | 真机 `joint_states` 实测位姿（半透明蓝，有 `--hand-sn` 数据时才有） |
| `/world/glove` | 手套 21 点人手骨架，置信度 <0.3 的点染红 |

网格只静态 log 一次，每帧只更新 `Transform3D`，所以 rrd 不随帧数爆（15s 约 9MB）。
自带 blueprint：3D 占主视图，`action`/`hand_state`/`track_err`/`policy` 四组曲线收进右侧标签页
—— 否则 80 条标量会把 3D 挤没。

### 9b. 直接可视化 SDK 录的 MCAP

官方 `examples/python/wuji_glove/2.recording.py` 只产 MCAP、**不带可视化**。
`tools/viz_mcap.py` 吃任意 `TopicRecorder` 录的 MCAP，不需要 episode 目录、不需要连设备：

```bash
./venv312/bin/python tools/viz_mcap.py rec.mcap
# 离线复算 retarget 并渲染机器人手（同样不连设备）
./venv312/bin/python tools/viz_mcap.py rec.mcap --retarget --hand-model wuji_hand_2
./venv312/bin/rerun rec.rrd
```

渲染 `/world/skeleton`（21 点 + 骨架 + **逐关节朝向轴**）、`/world/emf`（5 个线圈 6-DoF）、
`/tactile`（744 taxel → 24×31 热力图，独立常驻视图）、`/tactile_stat`（峰值 + 接触 taxel 数曲线）、
`--retarget` 时再加 `/world/robot`（MJCF 真网格）。

触觉两个坑（都已处理）：单通道 float 直接 `rr.Image` 会被按灰度画、0~0.9 的值几乎全黑，
所以自己上蓝→青→绿→黄→红色标，`-1` 的无效 taxel 单独染深灰（正好勾出手型）；
色标上限默认按本次录制的 **p99 自动定标**（实测单帧峰值常只有 0.3~0.5，
固定用 1.0 会让颜色停在蓝青段），跨录制比较时用 `--tactile-max 1.0` 固定。

**录制里的数据是全的**（实测官方三 topic 录制）：

| topic | 频率 | frame_id | 内容 |
|---|---|---|---|
| `hand_skeleton` | ~120Hz | `r_wrist` | 21 关节 position + orientation + confidence |
| `emf_poses` | ~120Hz | `r_hand_emf_tx` | 5 指尖接收线圈 6-DoF + confidence |
| `tactile` | ~120Hz | — | 744 值（-1 = 无效/屏蔽） |

- 四元数是**真数据**不是占位符：`|q|` 恒为 1，21 个关节里只有 `wrist` 和 `thumb_cmc`
  恒为单位四元数，其余 19 个带真实旋转。
- ⚠️ **两个 topic 不同坐标系**：骨架在 `r_wrist`、EMF 在 `r_hand_emf_tx`。本工具各自
  独立成组渲染，**没有对齐到同一世界系**。要接相机做同步渲染，还缺发射器→相机的外参，
  那不在录制数据里，得自己标。

### 9c. 规模压测

真机采集测不出管线边界，用合成数据压：

```bash
./venv312/bin/python tools/bench_pipeline.py --episodes 1000 --frames 300
./venv312/bin/python tools/bench_pipeline.py --dataset <数据集> --sweep   # 只调 dataloader
```

30 万帧合成数据实测：QC **26731 帧/s**（JSONL 容器；**真机 MCAP 路径只有 2961 帧/s**，
差 9 倍来自 MCAP 逐帧 JSON 解码和真机反馈的滞后扫描；
优化后 **6148 帧/s（2.06×）**，见 findings §8）、导出 **6846 帧/s**（730MB → 179MB parquet）、
峰值内存 **815MB**；dataloader 调参后 **12 140 → 77 807 samples/s（6.41×，5 次中位数）**。

导出那 6846 是优化后的数字：profile 发现 77% 的时间花在给**没有图像的数据集**跑
`embed_images`（逐帧 + 每帧重建 Features schema），按实际列类型短路后 2.9~4.4×，
产物逐位一致。详见 [`docs/findings.md`](docs/findings.md) §8/§9/§11。

每次导出都会打印耗时与帧率并写进 `wuji_provenance.json`，低于 500 帧/s 直接告警 ——
不用专门跑压测也能发现变慢（§12）。

### 9d. 训练与推理侧压测

```bash
# 训练吞吐 / 显存 / GPU 利用率，扫模型规模找 IO→compute 临界点（MLP 探针）
./venv312/bin/python tools/bench_train.py --dataset data/datasets/finger_tap \
    --repo-id local/wuji_finger_tap --models mlp big xl --fused-adam

# 拿真实策略（LeRobot ACT）复核，而不是用探针外推
./venv312/bin/python tools/bench_policy.py --dataset data/datasets/finger_tap \
    --repo-id local/wuji_finger_tap --batch-sizes 8 16

# 推理尾延迟，与 120Hz 帧预算 / 33ms 端到端延迟对账
./venv312/bin/python tools/bench_infer.py --model data/models/bc_tap.pt
```

**训练**（RTX 2060，重复取中位数）：MLP 探针测出翻转点在 13M~118M 参数之间，但拿
**真实 ACT 策略**（40.24M，动作块 100）复核发现它**已经是 compute 绑死**（dataload 仅 7.6%）——
**决定瓶颈的不是参数量而是每样本计算量**，动作块长度会把翻转点往左推。
ACT 实测 **110 → 444 samples/s（4.04×）**，AMP +46%、fused Adam 再 +35%。
调优提速 mlp **5.82×** / big **4.82×** / xl **4.14×**；AMP 收益跟着瓶颈走：
IO 绑死时 +3%，compute 绑死时 **+103%**；fused Adam 再 +42% 且省 18% 显存。
⚠️ Turing 的 `is_bf16_supported()` 返回 True 但无原生张量核，必须用 fp16。
⚠️ IO 绑死的配置单次波动可达 40%，必须重复取中位数（工具默认 `--repeats 3`）。

**推理**：BC baseline 单帧端到端 p99（3 次）**CPU 0.043~0.061ms / GPU 0.257~0.777ms**
—— 小模型上 CPU 完胜，且 GPU 尾延迟又高又不稳（最坏 2.5ms）。按最差一次算余量仍有
137×，可直接塞进遥操回路。

详见 [`docs/findings.md`](docs/findings.md) §13/§14。

### 10. 无硬件自测

```bash
python3 tools/synth_episode.py --out /tmp/fw/episodes --defect all   # 12 条带缺陷样本
python3 tools/qc_episode.py /tmp/fw/episodes

python3 tests/test_qc_episode.py                       # QC 规则（纯标准库）
./venv312/bin/python tests/test_mcap_join.py           # MCAP join + 真机跟踪误差/滞后
./venv312/bin/python tests/test_record_dedup.py        # 录制回路去重/对齐/预热
./venv312/bin/python tests/test_export_label.py        # 导出筛选/分组守卫/标注/手型解析
#   test_mcap_join.py 里还覆盖 rerun 导出（含真网格）与裸 MCAP 读取/四元数转换
```


## 实现要点 / 踩过的坑

- **仿真用运动学显示**：MJCF 的 position 执行器 `kp` 很软（2/1/0.8），`ctrl+mj_step` 会严重欠到位
  （握拳只弯一点）。遥操可视化改用 `d.qpos[:] = retarget输出; mj_forward`，精确呈现姿态。
- **连手套前 `switch_to_default_user()`**：默认让手套跑内置默认手 URDF（跟随更可靠），退出还原；
  连接用 `ConnectOptions(enable_bridge=False)`。照官方 `1.teleop_real.py`。要用**自己的标定手模型**
  就加 `--keep-user`（保留当前具名用户，加载其 `right_hand.urdf`）——标定/加载细节见
  [`docs/customer-guide-calibration-and-recording.md`](docs/customer-guide-calibration-and-recording.md)。
- **`step()` 输出已是固件关节序**，与 MJCF 的 `finger1_joint1..finger5_joint4` 一一对应，直接下发。
- **输入骨架对齐**：`apply_mediapipe_transformations` 后经机器人 `palm_link` 位姿搬到世界（同官方
  `tuning_tool` 的做法）。样式照 `wuji_retargeting/viz/skeleton_drawer.py` 的 `DEFAULT_LAYER_CONFIG`。
- **手套单客户端**：另一个窗口占用会报 `Session already exists`，脚本已加自动重试。

## 致谢

- 手模型 `wuji_hand_description/`（URDF/MJCF/mesh）来自
  [wuji-technology/wuji_hand_description](https://github.com/wuji-technology/wuji_hand_description)。
- retargeting 算法：`wuji-sdk` 内置（8.3 起顶层 `HandModel` / `RetargetSession`；≤7.15 在 `wuji_sdk.retargeting` 子模块）。
- 可视化参考：[wuji-technology/wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) 的 `tuning_tool`。
