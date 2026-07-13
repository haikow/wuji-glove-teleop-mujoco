# wuji-glove-teleop-mujoco

用 **Wuji Glove → `wuji_sdk.retargeting`(SDK 内置 retargeting)→ MuJoCo** 实时预览灵巧手遥操效果。
无手套时可用录制数据回放；也提供一套 ROS2 两节点管线（retarget 节点发 topic → MuJoCo 订阅）。

> **重要说明**：本仓库**不包含任何 retargeting 算法实现**，仅通过 `wuji-sdk` 的**公开 API**
> （`wuji_sdk.retargeting.RetargetSession`）调用其内置重定向。算法本体在 `wuji-sdk` wheel 内的
> 编译扩展里，不在本仓库、也不随本仓库分发。本仓库是一套 **ROS2 / MuJoCo 集成方案**，
> 由使用者自行 `pip install "wuji-sdk[retarget]"` 获取算法。

> 手部重定向用的是 `wuji-sdk` wheel 里内置的 retargeting（`pip install "wuji-sdk[retarget]"`），
> 与官方示例 `wuji-sdk/examples/python/retargeting/1.teleop_real.py` 同一套算法；这里把输出喂给
> MuJoCo 仿真手而不是真机，方便直接看效果。

## 环境（重要）

**必须 Python 3.12**：`wuji_sdk.retargeting` 的编译扩展只提供 cp312 ABI（Python 3.13/3.14 会
`import` 到但拿不到 `Retargeter`）。3.12 同时也是 ROS2 Humble/Kilted 的 ABI，一套环境通吃。

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
换真机时把 MuJoCo 订阅端替换成 `wujihandros2`/自研驱动即可，计算节点不动。

## 实现要点 / 踩过的坑

- **仿真用运动学显示**：MJCF 的 position 执行器 `kp` 很软（2/1/0.8），`ctrl+mj_step` 会严重欠到位
  （握拳只弯一点）。遥操可视化改用 `d.qpos[:] = retarget输出; mj_forward`，精确呈现姿态。
- **连手套前 `switch_to_default_user()`**：让手套跑内置默认手 URDF（跟随更可靠），退出还原；
  连接用 `ConnectOptions(enable_bridge=False)`。照官方 `1.teleop_real.py`。
- **`step()` 输出已是固件关节序**，与 MJCF 的 `finger1_joint1..finger5_joint4` 一一对应，直接下发。
- **输入骨架对齐**：`apply_mediapipe_transformations` 后经机器人 `palm_link` 位姿搬到世界（同官方
  `tuning_tool` 的做法）。样式照 `wuji_retargeting/viz/skeleton_drawer.py` 的 `DEFAULT_LAYER_CONFIG`。
- **手套单客户端**：另一个窗口占用会报 `Session already exists`，脚本已加自动重试。

## 致谢

- 手模型 `wuji_hand_description/`（URDF/MJCF/mesh）来自
  [wuji-technology/wuji_hand_description](https://github.com/wuji-technology/wuji_hand_description)。
- retargeting 算法：`wuji-sdk`（`wuji_sdk.retargeting`）。
- 可视化参考：[wuji-technology/wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) 的 `tuning_tool`。
