#!/usr/bin/env python3
"""合成 episode（可注入缺陷），用来在**没有手套**的情况下验证 QC 规则。

只用标准库。既是 tests/test_qc_episode.py 的夹具，也可以手动造一批数据看 QC 输出：

    python tools/synth_episode.py --out data/episodes --defect none
    python tools/synth_episode.py --out data/episodes --defect all
    python tools/qc_episode.py data/episodes
"""
import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.episode_format import EpisodeWriter  # noqa: E402

DEFECTS = ["none", "dropout", "duplicate", "static", "jump", "short", "lowrate",
           "gap", "nonfinite", "lowconf", "tactile_dead", "obs_only"]

DIM = 20
N_SK = 21


def _action(t, phase, amp=0.45):
    """20 维平滑动作：中心 0.5 rad，正弦摆动，range≈2*amp。"""
    return [0.5 + amp * math.sin(2 * math.pi * 0.5 * t + phase * j) for j in range(DIM)]


def _skeleton(t, rng):
    return [[0.01 * i + 0.005 * math.sin(t + i) + rng.gauss(0, 2e-4),
             0.002 * math.cos(t + i) + rng.gauss(0, 2e-4),
             0.003 * math.sin(0.5 * t + i) + rng.gauss(0, 2e-4)]
            for i in range(N_SK)]


def make(out_dir, defect="none", side="left", task="synth", n=200, hz=100.0,
         seed=0, tactile=False, episode_id=None):
    """造一条 episode，返回目录路径。"""
    rng = random.Random(seed)
    if defect == "short":
        n = 10
    if defect == "lowrate":
        hz = 8.0
    dt = 1.0 / hz
    t_host0 = 1_770_000_000.0
    t_dev0 = 5_000_000

    env = {
        "glove_sn": "SYNTH000000000000",
        "hand_model": "wuji_hand",
        "hand_model_path": "<synthetic>",
        "user_id": "synthetic",
        "calibrated": False,
        "user_mode": "default_user",
        "sdk_version": "synthetic",
        "action_space": {"name": "retarget_qpos", "dim": DIM, "unit": "rad"},
        "recorder": "tools/synth_episode.py",
        "synthetic_defect": defect,
    }
    ep = EpisodeWriter(out_dir, side=side, task=task, env=env,
                       episode_id=episode_id or "ep_synth_%s_%s" % (defect, side))

    seq = 1000
    extra_gap = 0.0
    for i in range(n):
        # 缺陷注入：丢帧 = seq 跳号（帧不写但设备侧确实产了帧）
        if defect == "dropout" and i % 5 == 0 and i > 0:
            seq += 3
        else:
            seq += 1
        if defect == "gap" and i == n // 2:
            extra_gap += 1.0            # 中间卡 1 秒

        t = i * dt + extra_gap
        t_host = t_host0 + t
        t_dev_us = int(t_dev0 + t * 1e6)

        if defect == "static":
            act = _action(0.0, 0.3, amp=0.01)      # 几乎不动
        else:
            act = _action(t, 0.3)
        if defect == "jump" and i % 40 == 20:
            act = [a + 1.2 for a in act]
        if defect == "nonfinite" and i == n // 3:
            act[0] = float("nan")

        conf = [0.95] * N_SK
        if defect == "lowconf" and i % 2 == 0:
            conf = [0.05] * N_SK

        tac = None
        if tactile or defect == "tactile_dead":
            tac = ([0.0] * 96 if defect == "tactile_dead"
                   else [max(0.0, math.sin(t + k)) for k in range(96)])

        kw = dict(seq=seq, t_dev_us=t_dev_us, t_host=t_host,
                  skeleton=_skeleton(t, rng), confidence=conf,
                  joint_angles=act[:DIM], ja_seq=seq, tactile=tac)
        if defect != "obs_only":
            kw["action"] = act
            kw["action_raw_max_ovr"] = 0.0
        ep.write_frame(**kw)

        # 重复帧：同一个 seq 再写一遍（模拟录制端没按 seq 去重）
        if defect == "duplicate" and i % 4 == 0:
            kw["t_host"] = t_host + dt * 0.3
            kw["t_dev_us"] = t_dev_us
            ep.write_frame(**kw)

    ep.finalize(success=None if defect != "none" else True)
    return ep.path


def main():
    ap = argparse.ArgumentParser(description="合成 episode（离线验证 QC 用）")
    ap.add_argument("--out", default="data/episodes")
    ap.add_argument("--defect", default="none", choices=DEFECTS + ["all"])
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--task", default="synth")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--tactile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    defects = DEFECTS if args.defect == "all" else [args.defect]
    for dfc in defects:
        p = make(args.out, dfc, args.side, args.task, args.frames, args.hz,
                 args.seed, args.tactile)
        print("wrote", p)


if __name__ == "__main__":
    main()
