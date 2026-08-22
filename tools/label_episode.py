#!/usr/bin/env python3
"""给 episode 打标签：success / intervention / notes / task，直接改 meta.json。

纯标准库，不需要手套/SDK。

用法：
    python tools/label_episode.py data/episodes --list              # 看标注状态
    python tools/label_episode.py data/episodes/ep_xxx --success y
    python tools/label_episode.py data/episodes --review            # 逐条过一遍
    python tools/label_episode.py data/episodes --success n --yes   # 批量（需 --yes）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.episode_format import (  # noqa: E402
    find_episodes, load_meta, meta_path, quality_path, write_meta,
)


def _tri(v):
    return {"y": True, "yes": True, "true": True, "1": True,
            "n": False, "no": False, "false": False, "0": False,
            "clear": None, "none": None, "": None}[v.strip().lower()]


def _qc_brief(ep):
    p = quality_path(ep)
    if not os.path.isfile(p):
        return "无 quality.json"
    with open(p) as f:
        q = json.load(f)
    m = q.get("metrics", {})
    bits = ["PASS" if q.get("pass") else "FAIL"]
    if m.get("n_frames"):
        bits.append("%d 帧" % m["n_frames"])
    if m.get("rate_hz_median"):
        bits.append("%.0fHz" % m["rate_hz_median"])
    if m.get("signal_range_max_rad") is not None:
        bits.append("range=%.3f" % m["signal_range_max_rad"])
    if q.get("flags"):
        bits.append("flags=" + ",".join(q["flags"]))
    return "  ".join(bits)


def show(eps):
    print("%-34s %-9s %-13s %-8s %s"
          % ("episode", "success", "intervention", "task", "qc"))
    for ep in eps:
        m = load_meta(ep) if os.path.isfile(meta_path(ep)) else {}
        s = m.get("success")
        print("%-34s %-9s %-13s %-8s %s"
              % (os.path.basename(ep),
                 {True: "yes", False: "no", None: "-"}.get(s, str(s)),
                 "yes" if m.get("intervention") else "-",
                 (m.get("task") or "-")[:8], _qc_brief(ep)))


def apply_label(ep, success=..., intervention=..., notes=None, task=None):
    """... 表示"不修改这个字段"（None 是合法值，代表清空 success）。"""
    meta = load_meta(ep) if os.path.isfile(meta_path(ep)) else {}
    if success is not ...:
        meta["success"] = success
    if intervention is not ...:
        meta["intervention"] = bool(intervention)
    if notes is not None:
        meta["notes"] = notes
    if task is not None:
        meta["task"] = task
    write_meta(ep, meta)
    return meta


def review(eps):
    """逐条交互标注：显示 QC 摘要 + 预览视频路径，然后问 y/n。"""
    for i, ep in enumerate(eps, 1):
        m = load_meta(ep) if os.path.isfile(meta_path(ep)) else {}
        print("\n[%d/%d] %s" % (i, len(eps), os.path.basename(ep)))
        print("   qc      : %s" % _qc_brief(ep))
        print("   task=%s side=%s 当前 success=%s"
              % (m.get("task"), m.get("side"), m.get("success")))
        prev = os.path.join(ep, "preview.mp4")
        if os.path.isfile(prev):
            print("   preview : %s" % prev)
        try:
            a = input("   success? [y/n/回车=跳过/c=清空/q=退出] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n中断")
            return
        if a == "q":
            return
        if a == "":
            continue
        if a == "c":
            apply_label(ep, success=None)
        elif a in ("y", "n"):
            apply_label(ep, success=(a == "y"))
        else:
            print("   ? 跳过")


def main():
    ap = argparse.ArgumentParser(description="给 episode 打 success/intervention 标签")
    ap.add_argument("path", nargs="?", default="data/episodes")
    ap.add_argument("--list", action="store_true", help="只列出当前标注状态")
    ap.add_argument("--review", action="store_true", help="逐条交互标注")
    ap.add_argument("--success", default=None,
                    help="y/n/clear —— 设置或清空 success")
    ap.add_argument("--intervention", default=None, help="y/n")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--task", default=None, help="改 task 名")
    ap.add_argument("--yes", action="store_true", help="批量修改多条时的确认")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit("路径不存在：%s" % args.path)
    eps = find_episodes(args.path)
    if not eps:
        raise SystemExit("没找到 episode：%s" % args.path)

    if args.review:
        review(eps)
        show(eps)
        return 0
    if args.list or (args.success is None and args.intervention is None
                     and args.notes is None and args.task is None):
        show(eps)
        return 0

    if len(eps) > 1 and not args.yes:
        raise SystemExit("会改动 %d 条 episode，确认请加 --yes" % len(eps))

    kw = {}
    if args.success is not None:
        kw["success"] = _tri(args.success)
    if args.intervention is not None:
        kw["intervention"] = _tri(args.intervention)
    if args.notes is not None:
        kw["notes"] = args.notes
    if args.task is not None:
        kw["task"] = args.task
    for ep in eps:
        apply_label(ep, **kw)
    show(eps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
