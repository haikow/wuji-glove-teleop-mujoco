# Wuji Glove 客户使用指导：标定 URDF 加载 + 数据录制

面向使用 Wuji Glove + wuji-sdk 做遥操 / 数据采集的客户。解决两个高频问题：
① 标定后"手伸直、模型却弯 / URDF 没加载"；② 手套 + 触觉原始数据怎么完整录制。

---

## 一、标定 URDF 为什么"没加载"——因为跑在【默认用户】上

### 结论（已在真机复现）

**SDK 的「默认用户(Default)」永远使用内置 built-in URDF，忽略任何标定 —— 即使标定文件确实存在。**
标定 URDF 的自动加载**只对具名用户(named user)生效**。

实测对照（连接并订阅 `hand_skeleton` 时，SDK 日志的 `online hand_model:` 那一行）：

| 运行时用户 | `online hand_model:` 日志 | 结果 |
|---|---|---|
| 默认用户 Default | `default SDK user uses built-in default URDF` | ❌ 用内置，标定无效 |
| 具名用户（已标定） | `loading calibration.hand_model_paths.<profile> from ~/.wuji/sdk/users/<user_id>/models/<sn>_hand_<profile>_<id>.urdf` | ✅ 加载你的标定 |

> 验证方法：把一份**有效的**标定（toml + 真实存在的 urdf）放进默认用户目录后，以默认用户连接，
> 日志仍然是 `default SDK user uses built-in default URDF`。可见默认用户回落 built-in 是**设计行为**，
> 与标定文件在不在无关。

### 标定结果落盘路径

标定 URDF 命名：`<sn>_hand_<profile>_<calibration_id>.urdf`。落盘根目录按 SDK 用户区分：

| SDK 用户 | URDF 目录 | 参数表 |
|---|---|---|
| 默认用户 | `~/.wuji/sdk/models/` | `~/.wuji/sdk/params/<sn>.toml` |
| 具名用户 | `~/.wuji/sdk/users/<user_id>/models/` | `~/.wuji/sdk/users/<user_id>/params/<sn>.toml` |

唯一性由 `(user_id, device_sn, hand_profile)` 三元组保证。切换具名用户时，已连接设备会自动重新加载该用户名下对应 URDF，无需重连。
**⚠️ 但默认用户是例外：文件会落 `~/.wuji/sdk/models/`，运行时却强制 built-in、不加载。**

### 正确做法：三步（每步带验证）

**① 建一个具名用户（不要用默认）**
```python
from wuji_sdk import SdkManager
mgr = SdkManager.instance()
u = mgr.create_user("cust_calib", description="customer calibration")
print(u["user_id"])                     # 记下 user_id
```
C SDK 对应 `wuji_create_user(display_name, description, external_id, &created)`。

**② 切到该用户**
```python
mgr.switch_user(u["user_id"])
print(mgr.current_user())               # 确认 display_name=cust_calib, is_default=False
```

**③ 在该用户下重新标定**（Studio 里先把活动 profile 选成 cust_calib，再标定）。
> 注意：标定会存到"标定那一刻 SDK 的当前用户"名下。务必先确认当前用户，避免"以为标在 A、实际落在 B"。

**验证标定已加载**——连接并订阅 `hand_skeleton`，看日志：
```
online hand_model: loading calibration.hand_model_paths.<profile> from ~/.wuji/sdk/users/<user_id>/models/....urdf
```
看到它=成功；仍是 `default SDK user uses built-in default URDF`=还在默认用户上。

一句话自检（磁盘层面，确认标定确实落到某用户）：
```bash
cat ~/.wuji/sdk/users/<user_id>/params/<sn>.toml
# 应有 [calibration.hand_model_paths]，且 wujihand/wujihand2 指向真实存在的 .urdf 文件
```

### 常见坑

- **默认用户标定后不加载** → 换具名用户（本节核心）。
- **`g.get("calibration.hand_model_path")` 恒返回 None** → 那是设备侧单数参数，**不是**判断依据；真正的选择走 SDK 侧 toml 里的 `hand_model_paths`（复数表）。只认 `online hand_model:` 日志。
- **旧版 Studio / 旧格式** → 参数表若是 `hand_model_content = """..."""`（内联 URDF）而非 `[calibration.hand_model_paths]`，可能不被当前 SDK 加载。用能正常标定的 Studio 版本在具名用户下重标一次，产出新格式最稳。
- **标定成功但 profile 不对** → toml 里 `hand_profile` 要与遥操侧 `HandModel`（wujihand / wujihand2，即一代/二代手）一致。

---

## 二、数据录制：两种方式

| | 官方 `reference/wuji_sdk_official_2_recording.py` | 本仓库 `record_glove_data.py` |
|---|---|---|
| 落盘格式 | **MCAP**（二进制，LZ4 压缩） | **CSV + JSONL**（纯文本） |
| 打开方式 | Foxglove Studio / `mcap info` | Excel / pandas / 任意文本工具 |
| 录制机制 | SDK 原生 `TopicRecorder`，**全帧**、设备端时间戳、不丢帧 | 手写循环**只取最新帧**（丢积压防延迟）=同步快照 |
| 通道 | `tactile` + `emf_poses` + `hand_skeleton` | `hand_skeleton` + `hand_joint_angles` + `tactile` |
| 适用 | 归档 / Foxglove 回放 / 提交研发复现 | 快速看数、喂自研 pipeline、算延迟/量程 |

**选哪个：**
- 要**原始高保真、全帧、给 Wuji 研发复现** → 用官方 MCAP（`reference/wuji_sdk_official_2_recording.py`，录 `emf_poses` 原始电磁位姿）。
- 要**直接 Excel/pandas 分析、含解算后的关节角** → 用 `record_glove_data.py`。
- 两者可同时跑；也可按需给 `record_glove_data.py` 增补 `emf_poses` 通道或 `--mcap` 导出。

**用法**
```bash
# 官方 MCAP（需 pip install "wuji-sdk"）
python reference/wuji_sdk_official_2_recording.py         # 产物 ./data/<时间戳>.mcap

# 本仓库 CSV + JSONL
python record_glove_data.py --glove-sn <你的手套SN> --seconds 20 --out rec
#   rec.csv   宽表: t_host + 63列骨架 + N列关节角
#   rec.jsonl 每帧完整 JSON: skeleton(21×3) + joint_angles + tactile + 主机时间戳
```

### 关于"运动慢半拍 / 延迟"

- 消费端**只取最新帧、丢弃积压**（`record_glove_data.py` 已这么做）：`while (x := sub.recv()) is not None: last = x`。积压旧帧是延迟主因。
- 固件时间同步：确保 `time_sync=true`（较新固件已修 emf 时间漂移）。
- 仿真显示用 `d.qpos = qpos; mj_forward`（运动学置位），**不要** `ctrl + mj_step`（软执行器会欠到位，看着像跟不上，其实是仿真欠驱动，不是数据延迟）。

---

## 三、连接的手型（左/右）

`--side` 决定加载 `mjcf/left.xml` 还是 `right.xml`（留空自动读手套 `hand_side()`）；SDK 内部对左手自动做 Y 镜像。详见仓库 README「实时手套遥操」。
