#!/usr/bin/env python3
"""Episode 自动质检：算指标 → 打 flag → 写 quality.json → 分流 clean/rejected。

纯离线、只用标准库，不需要手套/SDK/MuJoCo，可在 CI 或任意机器上跑。

用法：
    python tools/qc_episode.py data/episodes/ep_20260822_170000_left   # 单个
    python tools/qc_episode.py data/episodes                           # 整目录
    python tools/qc_episode.py data/episodes --route link              # 顺便分流
    python tools/qc_episode.py data/episodes --set min_rate_hz=30      # 改阈值
    python tools/qc_episode.py data/episodes --strict                  # 有 fail 就退出码 1
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.episode_format import (  # noqa: E402
    SCHEMA_VERSION, find_episodes, iter_frames, load_meta, meta_path, quality_path,
)

DEFAULT_THRESHOLDS = {
    "min_frames": 30,            # 太短的 demo 没有训练价值
    "min_duration_s": 1.0,
    "max_duration_s": 300.0,
    "min_rate_hz": 20.0,         # 中位帧率下限
    "max_gap_s": 0.25,           # 单次最大空档（卡顿/掉线）
    "max_dropout_ratio": 0.05,   # 按 header.seq 跳号算的真实丢帧率
    "max_dup_ratio": 0.01,       # 重复 seq 占比（录制端没去重的老数据）
    "max_action_jump_rad": 0.5,  # 相邻帧单关节最大跳变
    "max_jump_ratio": 0.002,     # 允许的跳变帧占比
    "min_action_range_rad": 0.15,  # 全程 max-min 的最大值，低于此=近乎静止
    "min_confidence": 0.3,
    "max_low_conf_ratio": 0.2,
    "min_action_join_ratio": 0.9,  # obs.mcap 里有多少帧 join 上了 action 旁路
    "max_track_mae_rad": 0.20,   # 真机手跟踪误差（已扣掉最优滞后）上限
    "min_hand_state_ratio": 0.9,  # 带真机反馈的帧占比
    "max_clip_ratio": 0.3,       # 顶到 MJCF 限位的帧占比
    "max_tactile_zero_ratio": 0.98,
    "max_tactile_sat_ratio": 0.5,
}

# 命中即判 fail 的 flag；其余进 warnings（不拦截，但记录在 quality.json）
FAIL_FLAGS = {
    "too_short", "too_long", "low_rate", "gap", "dropout", "duplicate_seq",
    "action_jump", "near_static", "nonfinite", "no_action", "empty_signal", "empty",
    "low_action_join", "poor_tracking",
}


# ---- 无 numpy 的小统计工具 ----

def _median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _quantile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _finite(xs):
    return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]


def _flatten(v):
    """任意嵌套序列 → 扁平数字列表。"""
    out = []
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, (int, float)):
            out.append(x)
    return out


def _timeline(frames):
    """优先用设备时钟 t_dev_us（秒），缺失则回落主机 t_host。返回 (times, source)。"""
    dev = [f["t_dev_us"] / 1e6 for f in frames if isinstance(f.get("t_dev_us"), (int, float))]
    if len(dev) == len(frames) and len(dev) > 0:
        return dev, "device"
    host = [f.get("t_host") for f in frames]
    if all(isinstance(t, (int, float)) for t in host) and host:
        return list(host), "host"
    return [], "none"


def compute_metrics(frames):
    """从帧列表算出所有 QC 指标（不做判定）。"""
    m = {"n_frames": len(frames)}
    if not frames:
        return m

    times, tsrc = _timeline(frames)
    m["time_source"] = tsrc
    dts = [b - a for a, b in zip(times, times[1:])] if len(times) > 1 else []
    dts = [d for d in dts if math.isfinite(d)]
    m["duration_s"] = round(times[-1] - times[0], 3) if len(times) > 1 else 0.0
    if dts:
        med = _median(dts)
        m["dt_median_s"] = round(med, 6)
        m["dt_p99_s"] = round(_quantile(dts, 0.99), 6)
        m["dt_max_s"] = round(max(dts), 6)
        m["rate_hz_median"] = round(1.0 / med, 2) if med > 0 else float("inf")
        m["backwards_dt"] = sum(1 for d in dts if d < 0)   # 时钟回跳
    else:
        m["dt_median_s"] = m["dt_p99_s"] = m["dt_max_s"] = None
        m["rate_hz_median"] = None
        m["backwards_dt"] = 0

    # 丢帧 / 重复：按设备侧 header.seq 判，比时间差可靠
    seqs = [f["seq"] for f in frames if isinstance(f.get("seq"), int)]
    if len(seqs) > 1:
        span = seqs[-1] - seqs[0] + 1
        uniq = len(set(seqs))
        missing = max(span - uniq, 0)
        m["seq_span"] = span
        m["seq_missing"] = missing
        m["dropout_ratio"] = round(missing / span, 5) if span > 0 else 0.0
        m["dup_frames"] = len(seqs) - uniq
        m["dup_ratio"] = round((len(seqs) - uniq) / len(seqs), 5)
        m["seq_backwards"] = sum(1 for a, b in zip(seqs, seqs[1:]) if b < a)
    else:
        m["seq_span"] = None
        m["seq_missing"] = None
        m["dropout_ratio"] = None
        m["dup_frames"] = 0
        m["dup_ratio"] = 0.0
        m["seq_backwards"] = 0

    # action 旁路 join 覆盖率：obs.mcap 录全部帧，而 retarget 回路走"取最新帧"会主动
    # 跳过积压，两边帧集不完全相同，join 不上的帧导不进训练集，必须量出来。
    n_act = sum(1 for f in frames if isinstance(f.get("action"), list))
    m["action_frames"] = n_act
    m["action_join_ratio"] = round(n_act / len(frames), 5) if frames else 0.0

    # 动作信号：优先 action（retarget 输出），没有就退回手套 joint_angles
    key = "action" if any("action" in f for f in frames) else "joint_angles"
    m["signal_key"] = key
    sig = [f[key] for f in frames if isinstance(f.get(key), list)]
    m["n_signal_frames"] = len(sig)
    if sig:
        dim = min(len(v) for v in sig)
        m["signal_dim"] = dim
        cols = [[v[j] for v in sig] for j in range(dim)]
        nonfinite = sum(1 for v in sig for x in v[:dim]
                        if not (isinstance(x, (int, float)) and math.isfinite(x)))
        m["nonfinite_count"] = nonfinite
        ranges = []
        for c in cols:
            cf = _finite(c)
            ranges.append(max(cf) - min(cf) if cf else 0.0)
        m["signal_range_max_rad"] = round(max(ranges), 5) if ranges else 0.0
        m["signal_range_mean_rad"] = round(sum(ranges) / len(ranges), 5) if ranges else 0.0
        jump_max, jump_frames = 0.0, 0
        for a, b in zip(sig, sig[1:]):
            dmax = 0.0
            for j in range(dim):
                x, y = a[j], b[j]
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                        and math.isfinite(x) and math.isfinite(y):
                    dmax = max(dmax, abs(y - x))
            jump_max = max(jump_max, dmax)
            if dmax > DEFAULT_THRESHOLDS["max_action_jump_rad"]:
                jump_frames += 1
        m["signal_jump_max_rad"] = round(jump_max, 5)
        m["signal_jump_frames"] = jump_frames
        m["jump_ratio"] = round(jump_frames / max(len(sig) - 1, 1), 5)
    else:
        m["signal_dim"] = 0
        m["nonfinite_count"] = 0
        m["signal_range_max_rad"] = 0.0
        m["signal_range_mean_rad"] = 0.0
        m["signal_jump_max_rad"] = 0.0
        m["signal_jump_frames"] = 0
        m["jump_ratio"] = 0.0

    # 真机手跟踪误差：把"滞后"和"跟不动"分开。零滞后的 MAE 里混着通信+伺服延迟，
    # 扫一遍滞后取最小值，才是真正的跟踪偏差；最优滞后本身就是端到端延迟估计。
    hs = [(i, f["hand_state"]) for i, f in enumerate(frames)
          if isinstance(f.get("hand_state"), list)]
    m["hand_state_frames"] = len(hs)
    m["hand_state_ratio"] = round(len(hs) / len(frames), 5) if frames else 0.0
    if len(hs) > 10 and key == "action":
        idx = [i for i, _ in hs]
        cmd = [frames[i].get("action") for i in idx]
        fb = [v for _, v in hs]
        dt_ms = (m.get("dt_median_s") or 0.00833) * 1000.0

        def _mae(lag):
            tot = cnt = 0
            for a in range(len(cmd) - lag):
                c, f2 = cmd[a], fb[a + lag]
                if not isinstance(c, list):
                    continue
                for j in range(min(len(c), len(f2))):
                    x, y = c[j], f2[j]
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                            and math.isfinite(x) and math.isfinite(y):
                        tot += abs(y - x)
                        cnt += 1
            return (tot / cnt) if cnt else None

        base = _mae(0)
        best_lag, best = 0, base
        for lag in range(1, min(25, len(cmd) // 4)):     # 最多扫 ~200ms
            v = _mae(lag)
            if v is not None and (best is None or v < best):
                best_lag, best = lag, v
        m["track_mae_rad"] = round(base, 5) if base is not None else None
        m["track_mae_best_rad"] = round(best, 5) if best is not None else None
        m["track_best_lag_frames"] = best_lag
        m["track_best_lag_ms"] = round(best_lag * dt_ms, 2)
    else:
        m["track_mae_rad"] = m["track_mae_best_rad"] = None
        m["track_best_lag_frames"] = m["track_best_lag_ms"] = None

    # retarget 顶限位：说明手型/标定不匹配，动作被截断
    ovr = [f["action_raw_max_ovr"] for f in frames
           if isinstance(f.get("action_raw_max_ovr"), (int, float))]
    if ovr:
        m["clip_ratio"] = round(sum(1 for x in ovr if x > 1e-6) / len(ovr), 5)
        m["clip_max_rad"] = round(max(ovr), 5)
    else:
        m["clip_ratio"] = None
        m["clip_max_rad"] = None

    # 关键点置信度
    confs = [f["confidence"] for f in frames if isinstance(f.get("confidence"), list)]
    if confs:
        mins = [min(_finite(c), default=0.0) for c in confs]
        m["conf_min"] = round(min(mins), 4)
        m["conf_mean"] = round(sum(sum(_finite(c)) / max(len(_finite(c)), 1)
                                   for c in confs) / len(confs), 4)
        m["low_conf_ratio"] = round(
            sum(1 for x in mins if x < DEFAULT_THRESHOLDS["min_confidence"]) / len(mins), 5)
    else:
        m["conf_min"] = m["conf_mean"] = m["low_conf_ratio"] = None

    # 触觉：全零=没接/没触发，饱和=压爆或标定漂了
    tac = [_flatten(f["tactile"]) for f in frames if f.get("tactile") is not None]
    if tac:
        tot = sum(len(t) for t in tac)
        zeros = sum(1 for t in tac for x in t if x == 0)
        peak = max((max(t) for t in tac if t), default=0.0)
        sat_thr = peak * 0.99 if peak > 0 else float("inf")
        sat = sum(1 for t in tac for x in t if x >= sat_thr) if peak > 0 else 0
        m["tactile_frames"] = len(tac)
        m["tactile_zero_ratio"] = round(zeros / tot, 5) if tot else None
        m["tactile_sat_ratio"] = round(sat / tot, 5) if tot else None
        m["tactile_peak"] = round(peak, 5)
    else:
        m["tactile_frames"] = 0
        m["tactile_zero_ratio"] = None
        m["tactile_sat_ratio"] = None
        m["tactile_peak"] = None
    return m


def evaluate(metrics, thr):
    """指标 + 阈值 → (flags, warnings)。flags 里只要有 FAIL_FLAGS 就判 fail。"""
    f, w = [], []

    def hit(name, cond):
        if cond:
            (f if name in FAIL_FLAGS else w).append(name)

    n = metrics.get("n_frames", 0)
    if n == 0:
        return ["empty"], []

    hit("too_short", n < thr["min_frames"]
        or (metrics.get("duration_s") or 0) < thr["min_duration_s"])
    hit("too_long", (metrics.get("duration_s") or 0) > thr["max_duration_s"])

    rate = metrics.get("rate_hz_median")
    hit("low_rate", rate is not None and rate < thr["min_rate_hz"])
    gap = metrics.get("dt_max_s")
    hit("gap", gap is not None and gap > thr["max_gap_s"])
    hit("clock_backwards", metrics.get("backwards_dt", 0) > 0
        or metrics.get("seq_backwards", 0) > 0)

    dro = metrics.get("dropout_ratio")
    hit("dropout", dro is not None and dro > thr["max_dropout_ratio"])
    dup = metrics.get("dup_ratio")
    hit("duplicate_seq", dup is not None and dup > thr["max_dup_ratio"])
    if metrics.get("seq_span") is None:
        w.append("no_device_seq")
    if metrics.get("time_source") == "host":
        w.append("host_clock_only")

    if metrics.get("n_signal_frames", 0) == 0:
        f.append("empty_signal")
    else:
        # 只有 obs 没有 action 的 episode 训不了策略（如 record_glove_data.py 的老数据），
        # 指标仍按 joint_angles 算出来，但直接判 fail。
        if metrics.get("signal_key") != "action":
            f.append("no_action")
        hit("nonfinite", metrics.get("nonfinite_count", 0) > 0)
        hit("action_jump",
            metrics.get("signal_jump_max_rad", 0) > thr["max_action_jump_rad"]
            and metrics.get("jump_ratio", 0) > thr["max_jump_ratio"])
        hit("near_static",
            metrics.get("signal_range_max_rad", 0) < thr["min_action_range_rad"])

    if metrics.get("action_frames", 0) > 0:
        hit("low_action_join",
            metrics.get("action_join_ratio", 1.0) < thr["min_action_join_ratio"])

    if metrics.get("hand_state_frames", 0) > 0:
        hit("low_hand_state", metrics.get("hand_state_ratio", 1.0)
            < thr["min_hand_state_ratio"])
        tm = metrics.get("track_mae_best_rad")
        hit("poor_tracking", tm is not None and tm > thr["max_track_mae_rad"])

    clip = metrics.get("clip_ratio")
    hit("action_clipped", clip is not None and clip > thr["max_clip_ratio"])
    lcr = metrics.get("low_conf_ratio")
    hit("low_confidence", lcr is not None and lcr > thr["max_low_conf_ratio"])

    tzr = metrics.get("tactile_zero_ratio")
    hit("tactile_dead", tzr is not None and tzr > thr["max_tactile_zero_ratio"])
    tsr = metrics.get("tactile_sat_ratio")
    hit("tactile_saturated", tsr is not None and tsr > thr["max_tactile_sat_ratio"])
    return f, w


def qc_episode(ep_dir, thr=None, write=True):
    """跑完整 QC，返回 quality dict（write=True 时同时写 quality.json）。"""
    thr = dict(DEFAULT_THRESHOLDS, **(thr or {}))
    frames = list(iter_frames(ep_dir))
    metrics = compute_metrics(frames)
    flags, warnings = evaluate(metrics, thr)
    meta = load_meta(ep_dir) if os.path.isfile(meta_path(ep_dir)) else {}
    if not meta:
        warnings.append("no_meta")
    elif meta.get("obs_container") == "mcap":
        # MCAP episode 的 meta.num_frames 记的是"对齐帧数"(=action 条数)，
        # 而 metrics.n_frames 是 MCAP 里的 obs 帧总数（含没 join 上 action 的）。
        if meta.get("num_frames") not in (None, metrics.get("action_frames")):
            warnings.append("meta_frame_count_mismatch")
    elif meta.get("num_frames") not in (None, len(frames)):
        warnings.append("meta_frame_count_mismatch")

    # SDK TopicRecorder 自带的录制质量（drop_rate / sync_rate），录制时存进了 meta，
    # 这里原样带进 quality.json，方便和我们自己算的指标交叉验证。
    if meta.get("sdk_quality"):
        metrics["sdk_quality"] = meta["sdk_quality"]

    q = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": meta.get("episode_id", os.path.basename(os.path.normpath(ep_dir))),
        "pass": len(flags) == 0,
        "flags": flags,
        "warnings": warnings,
        "metrics": metrics,
        "thresholds": thr,
        "qc_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if write:
        tmp = quality_path(ep_dir) + ".tmp"
        with open(tmp, "w") as fp:
            json.dump(q, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        os.replace(tmp, quality_path(ep_dir))
    return q


def route(ep_dir, passed, clean_dir, rejected_dir, mode):
    """把 episode 归到 clean/ 或 rejected/。link 用相对软链，最省空间且可回溯。"""
    if mode == "none":
        return None
    dst_root = clean_dir if passed else rejected_dir
    os.makedirs(dst_root, exist_ok=True)
    dst = os.path.join(dst_root, os.path.basename(os.path.normpath(ep_dir)))
    if os.path.islink(dst):
        os.unlink(dst)
    elif os.path.exists(dst):
        if mode == "link":
            return dst
        shutil.rmtree(dst)
    if mode == "link":
        os.symlink(os.path.relpath(os.path.abspath(ep_dir), dst_root), dst)
    elif mode == "copy":
        shutil.copytree(ep_dir, dst)
    elif mode == "move":
        shutil.move(ep_dir, dst)
    return dst


def _parse_set(pairs):
    thr = {}
    for p in pairs:
        k, _, v = p.partition("=")
        k = k.strip()
        if k not in DEFAULT_THRESHOLDS:
            raise SystemExit("unknown threshold: %s（可用：%s）"
                             % (k, ", ".join(sorted(DEFAULT_THRESHOLDS))))
        thr[k] = float(v)
    return thr


def main():
    ap = argparse.ArgumentParser(description="episode 自动质检")
    ap.add_argument("path", nargs="?", default="data/episodes",
                    help="单个 episode 目录，或包含多个 episode 的父目录")
    ap.add_argument("--route", default="none", choices=["none", "link", "copy", "move"],
                    help="分流方式（默认 none 只写 quality.json 不动文件）")
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--rejected-dir", default="data/rejected")
    ap.add_argument("--thresholds", default="", help="阈值 JSON 文件，覆盖默认值")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="KEY=VAL", help="单条覆盖阈值，可重复")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON 汇总")
    ap.add_argument("--strict", action="store_true", help="存在 fail 时退出码 1")
    ap.add_argument("--dry-run", action="store_true", help="不写 quality.json、不分流")
    ap.add_argument("--print-thresholds", action="store_true")
    args = ap.parse_args()

    thr = {}
    if args.thresholds:
        with open(args.thresholds) as f:
            thr.update(json.load(f))
    thr.update(_parse_set(args.sets))
    if args.print_thresholds:
        print(json.dumps(dict(DEFAULT_THRESHOLDS, **thr), indent=2))
        return 0

    if not os.path.exists(args.path):
        raise SystemExit("路径不存在：%s" % args.path)
    eps = find_episodes(args.path)
    if not eps:
        raise SystemExit("没找到 episode（需要目录里有 frames.jsonl）：%s" % args.path)

    results = []
    for ep in eps:
        q = qc_episode(ep, thr, write=not args.dry_run)
        dst = None
        if not args.dry_run:
            dst = route(ep, q["pass"], args.clean_dir, args.rejected_dir, args.route)
        results.append((ep, q, dst))

    n_pass = sum(1 for _, q, _ in results if q["pass"])
    if args.json:
        print(json.dumps({
            "total": len(results), "pass": n_pass, "fail": len(results) - n_pass,
            "episodes": [dict(q, path=ep, routed_to=dst) for ep, q, dst in results],
        }, ensure_ascii=False, indent=2))
    else:
        print("%-34s %-6s %7s %7s %8s %8s  %s"
              % ("episode", "verdict", "frames", "Hz", "drop%", "range", "flags"))
        for ep, q, _ in results:
            m = q["metrics"]
            notes = ",".join(q["flags"]) or "-"
            if q["warnings"]:
                notes += "  (warn: %s)" % ",".join(q["warnings"])
            print("%-34s %-6s %7d %7s %8s %8s  %s" % (
                q["episode_id"][:34],
                "PASS" if q["pass"] else "FAIL",
                m.get("n_frames", 0),
                ("%.1f" % m["rate_hz_median"]) if m.get("rate_hz_median") else "-",
                ("%.2f" % (100 * m["dropout_ratio"])) if m.get("dropout_ratio") is not None else "-",
                ("%.3f" % m["signal_range_max_rad"]) if m.get("signal_range_max_rad") is not None else "-",
                notes))
        print("\n%d episodes: %d pass, %d fail" % (len(results), n_pass, len(results) - n_pass))
        if args.route != "none" and not args.dry_run:
            print("routed → %s / %s (%s)" % (args.clean_dir, args.rejected_dir, args.route))

    return 1 if (args.strict and n_pass != len(results)) else 0


if __name__ == "__main__":
    sys.exit(main())
