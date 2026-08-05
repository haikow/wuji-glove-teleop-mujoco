# Wuji Glove 客户使用指导：标定 URDF 加载 + 数据录制

面向使用 Wuji Glove + wuji-sdk 做遥操 / 数据采集的客户。解决两个高频问题：
① 标定后"手伸直、模型却弯 / URDF 没加载"；② 手套 + 触觉原始数据怎么完整录制。

> 本指南以 **2026.8.3**（CLI / SDK / Studio 同版本）、手套固件 **0.11.2** 为例，
> 在 Linux x86_64 上 **CLI 与 Studio 两种标定方式均实测通过**。

---

## 〇、先记住三条铁律（后面全是它们的展开）

1. **版本要配套**：`wuji` CLI、Python `wuji_sdk`、Studio 用**同一个版本**。换了新 SDK 就要用配套新工具**重新标定**——旧版本标出来的产物，新 SDK 不认，会静默回退内置手模型。
2. **必须用具名用户**：默认用户（Default）永远回退内置，忽略标定。标定和加载都要在一个具名用户下。
3. **加载只认这行日志**：连接时 SDK 日志出现
   `online hand_model: loading user hand model from …/right_hand.urdf` = 成功；
   若是 `using built-in default URDF` = 没加载。

---

## 一、标定：CLI 或 Studio，二选一（均 2026.8.3 实测通过）

标定是"每用户 + 每手性"的：产物落在
`~/.wuji/sdk/users/<user_id>/models/{left,right}_hand.urdf`（新格式，7.14 起）。

> 旧版本（7.2 时代）产物名形如 `<sn>_hand_<profile>_<hash>.urdf` 并带 `params/<sn>.toml`——
> **那种旧格式当前 SDK 不加载**，必须用新工具重标。

### 方式 A：CLI 标定（推荐；平台无关，最省事）

```bash
# 0) 装配套 CLI —— 从公开仓下载：https://github.com/wuji-technology/wuji-cli/releases
#    x86_64:
curl -L -o wuji https://github.com/wuji-technology/wuji-cli/releases/download/v2026.8.3/wuji_2026.8.3_amd64
#    ARM64（树莓派 / RK3588 等）改用：
#    curl -L -o wuji https://github.com/wuji-technology/wuji-cli/releases/download/v2026.8.3/wuji_2026.8.3_arm64
chmod +x wuji && mv wuji ~/bin/wuji          # 确保 ~/bin 在 PATH 里
wuji --version                               # 2026.8.3

# 1) 建具名用户并切换（默认用户不能标定）
wuji user create cust_calib --switch
wuji user list                              # 当前用户前带 *，且不是 Default

# 2) IK 标定（引导式，戴手套按提示摆 6 个姿势，每个摆到位保持约 1 秒自动采集）
wuji calib ik --sn <你的手套SN>
#   姿势：拇指依次捏 食指→中指→无名指→小指指尖；五指张开(flat_open)；四指弯 90°

# 3) 确认成功
wuji user show cust_calib
#   Right: calibrated  right_hand.urdf   ← 出现即成功

# 4)（可选）触觉接触标定 —— 需要 tactile_binary / tactile_residual 真实数据时做
wuji calib tactile --sn <你的手套SN>
```

### 方式 B：Studio 标定

1. 用 **CLI 先建并切到具名用户**（同上 `wuji user create cust_calib --switch`），
   或在 Studio 界面里把活动用户/profile 选成该具名用户 —— **务必不是 Default**。
2. 启动 `wuji-studio`（**版本与 SDK 一致，本文 2026.8.3**），连接手套，走标定流程。
3. 标完同样用 `wuji user show <用户>` 确认 `Right/Left: calibrated  right_hand.urdf`。

> ⚠️ **ARM 平台**：Studio 在 ARM 上曾有标定 / 3D 渲染的已知问题。ARM 用户如遇 Studio 标定异常，
> 直接改用**方式 A（CLI）** 兜底（CLI 平台无关）。

---

## 二、在代码里加载标定 URDF（关键顺序）

**先切到标定所在的具名用户 → 再 `connect`**。SDK 在连接那一刻按当前用户自动加载 `right_hand.urdf`。

> `switch_user()` **只接受 `user_id`，不接受显示名**（传显示名会报 not found）。

```python
from wuji_sdk import SdkManager

mgr = SdkManager.instance()

# 1) 由显示名查出 user_id
uid = next(u["user_id"] for u in mgr.list_users() if u["display_name"] == "cust_calib")

# 2) ★连接之前★切到该用户，并确认不是默认用户
mgr.switch_user(uid)
assert not mgr.current_user()["is_default"]

# 3) 再连接 —— 此时自动加载该用户的 right_hand.urdf
glove = mgr.connect(address="192.168.1.101:50001", device_name="glove_0")
# 之后订阅 hand_skeleton / tactile_point_cloud，用的就是标定后的手模型
```

**更省事的 CLI 法（脚本不用改）**：先全局切一次当前用户，再跑原脚本（脚本里不要再动用户）：

```bash
wuji user switch cust_calib
python 2.recording.py
```

### 验证是否真加载了

把日志级别设到 info，连接时看这行：

```
INFO wuji_sdk: online hand_model: loading user hand model from …/<user_id>/models/right_hand.urdf   ← 成功
```

> ⚠️ **不要**用 `glove.get("calibration.hand_model_path")` 判断——它**即使已加载也返回 `None`**，会误导。只认上面这行日志。

（进一步验证可录一份 mcap，用 `mcap info` 看 `hand_skeleton` / `tactile_point_cloud` 有帧数即可。）

### 常见坑（对照排查）

| 现象 | 原因 | 处理 |
|---|---|---|
| 日志仍是 `using built-in default URDF` | 连接前没切用户 / 切成了 Default | `switch_user(uid)` 必须在 `connect` **之前**，且非默认用户 |
| 同上 | 脚本里调了 `switch_to_default_user()` | 去掉该调用 |
| 同上 | 用的是**旧版本标定产物**（旧格式 `<sn>_hand_<profile>_<hash>.urdf` + toml） | 用配套新工具**重新标定**，产出 `right_hand.urdf` |
| 同上 | 该用户对**这台设备的手性**没有标定 | `wuji user show` 确认 `Right/Left: calibrated`；缺就补标 |
| 加载了但行为怪 | SDK 与标定所用 CLI/Studio **版本不配套** | 统一到同一版本 |
| `switch_user("名字")` 报 not found | 传了显示名 | 传 `user_id`（`list_users()` 查） |
| `tactile_binary` / `tactile_residual` 为空 | 当前用户挂着**不兼容的旧接触模型**触发 fail-closed | 换到干净的新用户；或 `wuji calib tactile` 重做接触标定 |

---

## 三、数据录制：两种方式

| | 官方 `reference/wuji_sdk_official_2_recording.py` | 本仓库 `record_glove_data.py` |
|---|---|---|
| 落盘格式 | **MCAP**（二进制，LZ4 压缩） | **CSV + JSONL**（纯文本） |
| 打开方式 | Foxglove Studio / `mcap info` | Excel / pandas / 任意文本工具 |
| 录制机制 | SDK 原生 `TopicRecorder`，**全帧**、设备端时间戳、不丢帧 | 手写循环**只取最新帧**（丢积压防延迟）=同步快照 |
| 通道 | 可自定义（`tactile` / `emf_poses` / `hand_skeleton` / `tactile_point_cloud` …） | `hand_skeleton` + `hand_joint_angles` + `tactile` |
| 适用 | 归档 / Foxglove 回放 / 提交研发复现 | 快速看数、喂自研 pipeline、算延迟/量程 |

> 📌 **录制也受"当前用户"影响**：要让 `hand_skeleton` / `tactile_point_cloud` 用的是**标定后手模型**，
> 录制脚本连接前同样要 `switch_user` 到标定用户（见第二节），否则录到的是内置模型解算的结果。

**选哪个：**
- 要**原始高保真、全帧、给 Wuji 研发复现** → 用官方 MCAP（录 `emf_poses` 原始电磁位姿）。
- 要**直接 Excel/pandas 分析、含解算后的关节角** → 用 `record_glove_data.py`。
- 两者可同时跑。

**用法**
```bash
# 官方 MCAP（需 pip install "wuji-sdk"）
python reference/wuji_sdk_official_2_recording.py         # 产物 ./data/<时间戳>.mcap

# 本仓库 CSV + JSONL
python record_glove_data.py --glove-sn <你的手套SN> --seconds 20 --out rec
#   rec.csv   宽表: t_host + 63列骨架 + N列关节角
#   rec.jsonl 每帧完整 JSON: skeleton(21×3) + joint_angles + tactile + 主机时间戳
```

### 触觉通道说明（tactile 家族）

| 通道 | 含义 | 出数据前提 |
|---|---|---|
| `tactile` | 原始 744 点（24×31）压力矩阵 | 直接可用 |
| `tactile_zones` | 按分区聚合 | 直接可用 |
| `tactile_point_cloud` | 触点 3D 位置 + 压力（LBS 蒙皮算） | 需 `hand_skeleton` + `tactile_zones` 都在出帧（即标定手模型正常） |
| `tactile_binary` / `tactile_residual` | 接触 / 残差 | 需**有效的 744 维接触模型**；缺失或旧格式会 fail-closed（空）→ 做 `wuji calib tactile` |

---

## 四、连接的手型（左/右）

`--side` 决定加载 `mjcf/left.xml` 还是 `right.xml`（留空自动读手套 `hand_side()`）；SDK 内部对左手自动做 Y 镜像。详见仓库 README「实时手套遥操」。

---

## 五、最小自检清单

```bash
wuji --version                                             # CLI 版本
python -c "import wuji_sdk; print(wuji_sdk.__version__)"   # SDK 版本，二者一致
wuji ping                                                 # 手套在线、固件版本
wuji user show <你的用户>                                  # Right/Left: calibrated  right_hand.urdf
```

代码里连接后，日志出现 `loading user hand model from …right_hand.urdf` = 全部就绪。
