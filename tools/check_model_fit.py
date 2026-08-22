#!/usr/bin/env python3
"""判定一批 episode 的动作到底属于哪套手模型 —— 用关节限位越界率当判据。

为什么需要：retarget 输出落在**目标手型自己的 URDF 限位**内。如果拿错代的 MJCF 去
裁剪或回放，越界率会立刻暴露出来。本仓库 vendor 的是一代手，用二代 profile 录的数据
在一代限位下实测 63.7% 的帧越界、最大 0.53 rad(30°) —— 那批数据的指令曾被错误裁剪。

这个脚本让上面这个结论可复现、也可以用来给新数据选模型：

    ./venv312/bin/python tools/check_model_fit.py data/episodes
    ./venv312/bin/python tools/check_model_fit.py data/clean --field hand_state
    ./venv312/bin/python tools/check_model_fit.py data/episodes \\
        --mjcf wuji_hand_description/mjcf/right.xml /tmp/h2/beta2.xml
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.episode_format import find_episodes, iter_frames, load_meta, meta_path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MJCF = [
    os.path.join(ROOT, "wuji_hand_description", "mjcf", "%s.xml"),
    os.path.join(ROOT, "wuji_hand_description2", "mjcf", "%s.xml"),
]


def joint_ranges(mjcf_path):
    """直接解 XML 取 (name, lo, hi)，不经 MuJoCo —— 不需要 mesh 文件也能比。"""
    out = []
    for j in ET.parse(mjcf_path).getroot().iter("joint"):
        if j.get("name") and j.get("range"):
            lo, hi = (float(x) for x in j.get("range").split())
            out.append((j.get("name"), lo, hi))
    return out


def violation(arr, lo, hi):
    """返回 (最大越界弧度, 越界帧占比, 逐关节最大越界)。"""
    ovr = np.maximum(lo - arr, 0) + np.maximum(arr - hi, 0)
    return float(ovr.max()), float((ovr.max(1) > 1e-6).mean()), ovr.max(0)


def main():
    ap = argparse.ArgumentParser(description="按关节限位越界率判定手模型是否匹配")
    ap.add_argument("path", nargs="?", default="data/episodes")
    ap.add_argument("--field", default="action", choices=["action", "hand_state"],
                    help="拿哪一路信号比：retarget 指令 或 真机实测")
    ap.add_argument("--mjcf", nargs="*", default=[], help="要比的 MJCF；留空按默认候选")
    ap.add_argument("--side", default="", help="left/right；留空按 episode meta 推断")
    args = ap.parse_args()

    eps = find_episodes(args.path)
    if not eps:
        raise SystemExit("没找到 episode：%s" % args.path)

    vals, sides = [], set()
    for ep in eps:
        if os.path.isfile(meta_path(ep)):
            sides.add(load_meta(ep).get("side"))
        for f in iter_frames(ep):
            v = f.get(args.field)
            if isinstance(v, list) and all(x is not None for x in v):
                vals.append(v)
    if not vals:
        raise SystemExit("这些 episode 里没有 %s 字段" % args.field)
    A = np.asarray(vals, float)
    side = args.side or (sides.pop() if len(sides) == 1 else "right")
    print("%d 条 episode / %d 帧 × %d 维（%s，side=%s）"
          % (len(eps), A.shape[0], A.shape[1], args.field, side))

    cands = args.mjcf or [p % side for p in DEFAULT_MJCF if os.path.isfile(p % side)]
    if not cands:
        raise SystemExit("没有可比的 MJCF（二代模型跑 tools/fetch_hand2_description.sh）")

    print()
    print("%-42s %10s %10s %s" % ("MJCF", "越界最大", "越界帧占比", "判定"))
    best = None
    for p in cands:
        r = joint_ranges(p)
        if len(r) != A.shape[1]:
            print("%-42s %10s %10s 关节数 %d ≠ %d，跳过"
                  % (os.path.relpath(p, ROOT), "-", "-", len(r), A.shape[1]))
            continue
        lo = np.array([x[1] for x in r])
        hi = np.array([x[2] for x in r])
        mx, ratio, per = violation(A, lo, hi)
        verdict = "匹配" if mx < 1e-6 else "不匹配"
        print("%-42s %9.4f %9.1f%%  %s"
              % (os.path.relpath(p, ROOT), mx, 100 * ratio, verdict))
        if mx > 1e-6:
            k = int(per.argmax())
            print("%-42s   最差关节 %s 越界 %.4f rad (%.1f°)"
                  % ("", r[k][0], per[k], np.degrees(per[k])))
        if best is None or mx < best[0]:
            best = (mx, p)

    print()
    if best and best[0] < 1e-6:
        print("→ 匹配：%s" % os.path.relpath(best[1], ROOT))
    else:
        print("→ 没有完全匹配的模型；越界最小的是 %s（%.4f rad）"
              % (os.path.relpath(best[1], ROOT), best[0]))
        print("  指令被错误限位裁剪会直接降低遥操质量，回放和 clip 也会失真。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
