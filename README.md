# wuji-glove-teleop-mujoco

用 **Wuji Glove → SDK 内置 retargeting → MuJoCo** 实时预览灵巧手遥操效果。
无手套时可用录制数据回放；也提供一套 ROS2 两节点管线（retarget 节点发 topic → MuJoCo 订阅）。

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

### 无手套：录制数据回放

```bash
MUJOCO_GL=egl ./venv312/bin/python mujoco_teleop_replay.py --mode video --out teleop.mp4 --frames 900
```

### ROS2 两节点管线（retarget 发 topic → MuJoCo 订阅）

```bash
source /opt/ros/<distro>/setup.bash
./run_ros2_teleop.sh view      # 或 video
```
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
