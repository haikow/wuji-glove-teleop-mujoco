#!/usr/bin/env python3
"""Episode 目录格式：读写 `meta.json` + `frames.jsonl`。

一次录制 = 一个自包含目录：

    data/episodes/ep_<YYYYMMDD_HHMMSS>_<side>/
        meta.json      # 身份/环境/标签，见 EpisodeWriter.finalize
        frames.jsonl   # 每帧一行，obs + action 同帧对齐
        quality.json   # 由 tools/qc_episode.py 写入

只依赖标准库：QC / 合成 / 测试 都不需要 wuji-sdk、numpy、mujoco。
"""
import json
import os
import time

SCHEMA_VERSION = 1

# frames.jsonl 每帧的键（None 值不落盘，保持行紧凑）
#   i                 帧序号（本 episode 内从 0 递增，连续无洞）
#   seq               hand_skeleton 的 header.seq（设备侧帧号，用于去重/丢帧统计）
#   t_dev_us          hand_skeleton 的 header.timestamp_us（设备时钟，已 sync_time 对齐）
#   t_host            主机 time.time()，秒，浮点（设备时钟不可用时的兜底）
#   skeleton          21×3 手部关键点（MediaPipe 序，米）
#   confidence        21 个关键点置信度
#   joint_angles      手套解算关节角（扁平，固件关节序）
#   ja_seq            hand_joint_angles 的 header.seq
#   action            retarget 输出的 20 维目标关节角（弧度，已 clip 到 MJCF 限位）
#   action_raw_max_ovr  clip 前超出限位的最大幅度（弧度，0 表示没顶到限位）
#   tactile           触觉扁平数组


def make_episode_id(side, ts=None):
    """ep_<YYYYMMDD_HHMMSS>_<side>"""
    lt = time.localtime(ts if ts is not None else time.time())
    return "ep_%s_%s" % (time.strftime("%Y%m%d_%H%M%S", lt), side)


def _tolist(v):
    """numpy array / 嵌套序列 → 纯 Python list（json 可序列化）。"""
    if v is None:
        return None
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_tolist(x) for x in v]
    return v


def _build_meta(episode_id, task, side, n_frames, duration_s, created_at,
                success, intervention, notes, env):
    meta = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "task": task,
        "side": side,
        "num_frames": n_frames,
        "duration_s": round(duration_s, 3),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(created_at)),
        "success": success,
        "intervention": bool(intervention),
        "notes": notes,
    }
    # env 里放身份/环境溯源字段（glove_sn / hand_model / user_id / calibrated /
    # sdk_version / action_space / time_sync / sdk_quality ...），见 record_episode.py
    meta.update(env)
    return meta


class EpisodeWriter:
    """逐帧写 frames.jsonl，收尾时写 meta.json。

    用法：
        with EpisodeWriter("data/episodes", side="left", task="pick", env=env) as ep:
            ep.write_frame(seq=..., skeleton=..., action=...)
        ep.finalize(success=True)      # 或在 with 退出时自动 finalize
    """

    def __init__(self, out_dir, side, task, env=None, episode_id=None):
        self.side = side
        self.task = task
        self.env = dict(env or {})
        self.episode_id = episode_id or make_episode_id(side)
        self.path = os.path.join(out_dir, self.episode_id)
        os.makedirs(self.path, exist_ok=True)
        self._f = open(os.path.join(self.path, "frames.jsonl"), "w")
        self.n_frames = 0
        self.created_at = time.time()
        self._t_first = None
        self._t_last = None
        self._finalized = False

    # ---- 写帧 ----
    def write_frame(self, seq=None, t_dev_us=None, t_host=None, skeleton=None,
                    confidence=None, joint_angles=None, ja_seq=None, action=None,
                    action_raw_max_ovr=None, tactile=None):
        if t_host is None:
            t_host = time.time()
        rec = {"i": self.n_frames, "t_host": round(t_host, 6)}
        for k, v in (("seq", seq), ("t_dev_us", t_dev_us), ("ja_seq", ja_seq)):
            if v is not None:
                rec[k] = int(v)
        for k, v in (("skeleton", skeleton), ("confidence", confidence),
                     ("joint_angles", joint_angles), ("action", action),
                     ("tactile", tactile)):
            v = _tolist(v)
            if v is not None:
                rec[k] = v
        if action_raw_max_ovr is not None:
            rec["action_raw_max_ovr"] = round(float(action_raw_max_ovr), 6)
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.n_frames += 1
        if self._t_first is None:
            self._t_first = t_host
        self._t_last = t_host

    # ---- 收尾 ----
    def finalize(self, success=None, intervention=False, notes=""):
        """写 meta.json 并关闭 frames.jsonl。重复调用只生效一次。"""
        if self._finalized:
            return self.path
        self._f.close()
        self._finalized = True
        dur = 0.0
        if self._t_first is not None and self._t_last is not None:
            dur = self._t_last - self._t_first
        write_meta(self.path, _build_meta(
            self.episode_id, self.task, self.side, self.n_frames, dur,
            self.created_at, success, intervention, notes, self.env))
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finalize()
        return False


class McapEpisodeWriter:
    """obs 走 SDK 的 TopicRecorder → `obs.mcap`，本类只负责 action 旁路 + meta。

    SDK 的 `TopicRecorder` 只能录 `Subscription`（设备话题），而 retarget action 是
    host 侧算出来的、不存在对应的设备资源路径（`publish()` 是往设备发，路径必须已存在），
    所以 action 存不进那个 MCAP。这里把 action 单独写成 `action.jsonl`，用
    `hand_skeleton` 的 `header.seq` 作 join 键 —— seq 是设备侧帧号，两个订阅看到的
    完全一致（实测 361/361 重合），所以 join 是精确的、不靠时间戳近似。
    """

    def __init__(self, out_dir, side, task, env=None, episode_id=None):
        self.side = side
        self.task = task
        self.env = dict(env or {})
        self.episode_id = episode_id or make_episode_id(side)
        self.path = os.path.join(out_dir, self.episode_id)
        os.makedirs(self.path, exist_ok=True)
        self._f = open(action_path(self.path), "w")
        self.n_actions = 0
        self.created_at = time.time()
        self._t_first = None
        self._t_last = None
        self._finalized = False

    def write_action(self, seq, action, t_dev_us=None, t_host=None,
                     action_raw_max_ovr=None):
        if t_host is None:
            t_host = time.time()
        rec = {"seq": int(seq), "t_host": round(t_host, 6),
               "action": _tolist(action)}
        if t_dev_us is not None:
            rec["t_dev_us"] = int(t_dev_us)
        if action_raw_max_ovr is not None:
            rec["action_raw_max_ovr"] = round(float(action_raw_max_ovr), 6)
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.n_actions += 1
        if self._t_first is None:
            self._t_first = t_host
        self._t_last = t_host

    def finalize(self, success=None, intervention=False, notes=""):
        if self._finalized:
            return self.path
        self._f.close()
        self._finalized = True
        dur = 0.0
        if self._t_first is not None and self._t_last is not None:
            dur = self._t_last - self._t_first
        env = dict(self.env)
        env["obs_container"] = "mcap"
        env["num_actions"] = self.n_actions
        write_meta(self.path, _build_meta(
            self.episode_id, self.task, self.side, self.n_actions, dur,
            self.created_at, success, intervention, notes, env))
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finalize()
        return False


# ---- 读取 ----

def meta_path(ep_dir):
    return os.path.join(ep_dir, "meta.json")


def frames_path(ep_dir):
    return os.path.join(ep_dir, "frames.jsonl")


def quality_path(ep_dir):
    return os.path.join(ep_dir, "quality.json")


def obs_mcap_path(ep_dir):
    return os.path.join(ep_dir, "obs.mcap")


def action_path(ep_dir):
    return os.path.join(ep_dir, "action.jsonl")


def load_meta(ep_dir):
    with open(meta_path(ep_dir)) as f:
        return json.load(f)


def write_meta(ep_dir, meta):
    """原子写 meta.json（先写 .tmp 再 rename，避免中断留下半个文件）。"""
    tmp = meta_path(ep_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, meta_path(ep_dir))


def _iter_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_frames(ep_dir):
    """逐帧 yield 归一化的帧 dict。

    两种容器都走这一个入口，下游（QC / 导出 / 训练）看到的结构完全一样：
      - `frames.jsonl`：自包含 JSONL（合成数据、老 episode）
      - `obs.mcap` + `action.jsonl`：SDK TopicRecorder 录的 obs + action 旁路
    """
    if os.path.isfile(frames_path(ep_dir)):
        yield from _iter_jsonl(frames_path(ep_dir))
    elif os.path.isfile(obs_mcap_path(ep_dir)):
        yield from _iter_frames_mcap(ep_dir)


def nid_to_flat(nid):
    """真机手关节 nid（1-4/6-9/11-14/16-19/21-24）→ 扁平 0..19（每指跳过第 5 个号）。"""
    return (nid - 1) - (nid - 1) // 5


def _nearest_join(times, values, targets):
    """把高频序列 (times, values) 按最近时间戳对齐到 targets，返回 [value|None]。

    真机手 joint_states 跑 ~999Hz，手套帧 120Hz，两边序列号空间不同，只能按时间戳并。
    """
    out, j, n = [], 0, len(times)
    for t in targets:
        if n == 0 or t is None:
            out.append(None)
            continue
        while j + 1 < n and abs(times[j + 1] - t) <= abs(times[j] - t):
            j += 1
        out.append(values[j])
    return out


def _iter_frames_mcap(ep_dir):
    """解 obs.mcap（jsonschema/JSON 编码）并按 header.seq join 上 action.jsonl。"""
    from mcap.reader import make_reader

    by_seq = {}
    order = []
    js_t, js_v = [], []          # 真机手 joint_states（高频，按时间戳并）
    with open(obs_mcap_path(ep_dir), "rb") as f:
        for _sch, chan, msg in make_reader(f).iter_messages():
            topic = chan.topic.rsplit("/", 1)[-1]
            if topic == "joint_states":
                try:
                    d = json.loads(msg.data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                ts = (d.get("header") or {}).get("timestamp_us")
                if ts is None:
                    continue
                pos = [None] * 20
                for j in (d.get("joints") or []):
                    k = nid_to_flat(j.get("nid", 0))
                    if 0 <= k < 20:
                        pos[k] = j.get("position")
                js_t.append(ts)
                js_v.append(pos)
                continue
            if topic not in ("hand_skeleton", "hand_joint_angles", "tactile"):
                continue
            try:
                d = json.loads(msg.data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            hdr = d.get("header") or {}
            seq = hdr.get("seq")
            if seq is None:
                continue
            rec = by_seq.get(seq)
            if rec is None:
                rec = by_seq[seq] = {"seq": seq}
                order.append(seq)
            if topic == "hand_skeleton":
                joints = d.get("joints") or []
                rec["skeleton"] = [[j["pose"]["position"][a] for a in ("x", "y", "z")]
                                   if isinstance(j["pose"]["position"], dict)
                                   else list(j["pose"]["position"])
                                   for j in joints]
                rec["confidence"] = [j.get("confidence") for j in joints]
                rec["t_dev_us"] = hdr.get("timestamp_us")
                # MCAP log time 是主机写盘时刻，缺 t_host 时拿它兜底
                rec.setdefault("t_host", msg.log_time / 1e9)
            elif topic == "hand_joint_angles":
                ja = []
                for fg in (d.get("fingers") or []):
                    ja.extend(fg.get("angles") or [])
                rec["joint_angles"] = ja
                rec["ja_seq"] = seq
            elif topic == "tactile":
                rec["tactile"] = d.get("data")

    if os.path.isfile(action_path(ep_dir)):
        for a in _iter_jsonl(action_path(ep_dir)):
            rec = by_seq.get(a.get("seq"))
            if rec is None:
                continue          # action 对应的 obs 帧不在 MCAP 里，丢弃
            rec["action"] = a.get("action")
            if a.get("action_raw_max_ovr") is not None:
                rec["action_raw_max_ovr"] = a["action_raw_max_ovr"]
            if a.get("t_host") is not None:
                rec["t_host"] = a["t_host"]

    ordered = sorted(order)
    if js_t:
        pairs = sorted(zip(js_t, js_v))
        ts = [p[0] for p in pairs]
        vs = [p[1] for p in pairs]
        joined = _nearest_join(ts, vs, [by_seq[s].get("t_dev_us") for s in ordered])
        for s, hs in zip(ordered, joined):
            if hs is not None:
                by_seq[s]["hand_state"] = hs

    for i, seq in enumerate(ordered):
        rec = by_seq[seq]
        rec["i"] = i
        yield rec


def load_frames(ep_dir):
    return list(iter_frames(ep_dir))


def load_actions(ep_dir):
    """只读 action 旁路（不解 MCAP），用于快速核对录制端写了什么。"""
    p = action_path(ep_dir)
    return list(_iter_jsonl(p)) if os.path.isfile(p) else []


def is_episode_dir(path):
    return os.path.isdir(path) and (os.path.isfile(frames_path(path))
                                    or os.path.isfile(obs_mcap_path(path)))


def find_episodes(root):
    """root 本身是 episode 就返回它；否则返回 root 下所有 episode 子目录（排序）。"""
    if is_episode_dir(root):
        return [root]
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if is_episode_dir(p):
            out.append(p)
    return out
