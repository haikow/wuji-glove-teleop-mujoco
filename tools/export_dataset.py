#!/usr/bin/env python3
"""clean episode → LeRobot 数据集（v3 布局，由官方库写）。

为什么用官方库而不是自己拼 parquet：LeRobot v3 的布局不只是 parquet 文件，还有
`meta/info.json`（schema/fps/path 模板）、`meta/stats.json`（归一化用）、
`meta/tasks.parquet`、`meta/episodes/`（episode 边界、chunk/file 索引、
dataset_from_index/to_index）。手写这套分片索引很容易出错，交给
`LeRobotDataset.create()/add_frame()/save_episode()`，格式由库负责。

**溯源分组守卫**：retarget 输出强依赖加载的手 URDF，默认用户走内置模型、具名用户走
per-user 标定模型，两者 action 分布不同。所以默认按
(task, side, hand_model_path, action_dim, obs_dim) 分组，发现多于一组就直接报错，
不允许悄悄混进一个数据集训出平均态。要混必须显式 --allow-mixed。

用法：
    ./venv312/bin/python tools/export_dataset.py --input data/clean \
        --repo-id local/wuji_pick_cube --out data/datasets/pick_cube

    # 只导某个任务、放宽 QC 门槛、把标了失败的也带上
    ... --task pick_cube --no-qc-filter --include-failures
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.episode_format import (  # noqa: E402
    find_episodes, iter_frames, load_meta, meta_path, quality_path,
)

# --obs 可以是逗号组合，例如 skeleton,hand。both = skeleton,joints（向后兼容）
OBS_COMPONENTS = ("skeleton", "joints", "hand")


def parse_obs_mode(mode):
    """"both" → ("skeleton","joints")；"skeleton,hand" → ("skeleton","hand")。"""
    if mode == "both":
        return ("skeleton", "joints")
    parts = tuple(x.strip() for x in mode.split(",") if x.strip())
    bad = [x for x in parts if x not in OBS_COMPONENTS]
    if bad or not parts:
        raise SystemExit("--obs 只能是 %s 的逗号组合（或 both），收到：%s"
                         % ("/".join(OBS_COMPONENTS), mode))
    return parts


def _obs_vector(frame, mode):
    """按 --obs 拼出 observation.state；任一部分缺失就返回 None（丢帧）。"""
    comps = parse_obs_mode(mode) if isinstance(mode, str) else mode
    parts = []
    if "skeleton" in comps:
        sk = frame.get("skeleton")
        if not sk:
            return None
        for p in sk:
            parts.extend(p)
    if "joints" in comps:
        ja = frame.get("joint_angles")
        if not ja:
            return None
        parts.extend(ja)
    if "hand" in comps:
        # 真机手本体感（joint_states 按时间戳并上来的 20 维实际位置）
        hs = frame.get("hand_state")
        if not hs or any(v is None for v in hs):
            return None
        parts.extend(hs)
    return parts


def _obs_names(mode, n_ja, joint_names=None):
    comps = parse_obs_mode(mode) if isinstance(mode, str) else mode
    names = []
    if "skeleton" in comps:
        names += ["sk%d_%s" % (i, a) for i in range(21) for a in ("x", "y", "z")]
    if "joints" in comps:
        names += ["ja%d" % i for i in range(n_ja)]
    if "hand" in comps:
        names += ["hand_%s" % (joint_names[i] if joint_names and i < len(joint_names)
                               else i) for i in range(20)]
    return names


def collect(input_dir, task=None, qc_filter=True, include_failures=False,
            obs_mode="both"):
    """扫描 episode，做 QC/标签过滤，返回 [(ep_dir, meta, frames)]。"""
    kept, skipped = [], []
    for ep in find_episodes(input_dir):
        if not os.path.isfile(meta_path(ep)):
            skipped.append((ep, "no_meta"))
            continue
        meta = load_meta(ep)
        if task and meta.get("task") != task:
            skipped.append((ep, "task_mismatch"))
            continue
        if qc_filter:
            qp = quality_path(ep)
            if not os.path.isfile(qp):
                skipped.append((ep, "no_quality_json"))
                continue
            with open(qp) as f:
                if not json.load(f).get("pass"):
                    skipped.append((ep, "qc_fail"))
                    continue
        if not include_failures and meta.get("success") is False:
            skipped.append((ep, "labeled_failure"))
            continue
        frames = [f for f in iter_frames(ep)
                  if isinstance(f.get("action"), list) and _obs_vector(f, obs_mode)]
        if not frames:
            skipped.append((ep, "no_aligned_frames"))
            continue
        kept.append((ep, meta, frames))
    return kept, skipped


def model_identity(meta):
    """这条 episode 的 retarget 实际用了哪套手模型 —— 分组的硬依据。

    优先 urdf_source_path（offline_pipeline 解析出的实际 URDF），兼容早期只记
    hand_model_path 的 episode。两者都没有时退回 (urdf_source, hand_model)。
    """
    return (meta.get("urdf_source_path") or meta.get("hand_model_path")
            or "%s/%s" % (meta.get("urdf_source"), meta.get("hand_model")))


def group_key(meta, action_dim, obs_dim):
    return (meta.get("task"), meta.get("side"), model_identity(meta),
            action_dim, obs_dim)


def verify(root, repo_id):
    """把导出的数据集用 LeRobotDataset 读回来 —— 证明它真的能被训练栈消费。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=root)
    print("加载成功：%s" % root)
    print("  episodes=%d  frames=%d  fps=%s  robot_type=%s"
          % (ds.meta.total_episodes, ds.meta.total_frames, ds.meta.fps,
             ds.meta.robot_type))
    print("  features: %s" % ", ".join(sorted(ds.meta.features)))
    s = ds[0]
    for k in ("observation.state", "action"):
        if k in s:
            print("  %-26s shape=%s dtype=%s" % (k, tuple(s[k].shape), s[k].dtype))
    stats = ds.meta.stats or {}
    if "action" in stats:
        a = stats["action"]
        print("  action stats: mean[0]=%.4f std[0]=%.4f min[0]=%.4f max[0]=%.4f"
              % (float(np.asarray(a["mean"]).ravel()[0]),
                 float(np.asarray(a["std"]).ravel()[0]),
                 float(np.asarray(a["min"]).ravel()[0]),
                 float(np.asarray(a["max"]).ravel()[0])))
    prov = os.path.join(root, "wuji_provenance.json")
    if os.path.isfile(prov):
        with open(prov) as f:
            p = json.load(f)
        hm = {e.get("urdf_source_path") or e.get("hand_model_path")
              for e in p["episodes"]}
        print("  溯源：%d 条 episode，手模型 %d 种 %s"
              % (len(p["episodes"]), len(hm), sorted(x or "?" for x in hm)))
    return 0


def main():
    ap = argparse.ArgumentParser(description="clean episode → LeRobot 数据集")
    ap.add_argument("--input", default="data/clean")
    ap.add_argument("--out", default="data/datasets/export",
                    help="数据集根目录（LeRobotDataset 的 root）")
    ap.add_argument("--repo-id", default="local/wuji_glove_teleop")
    ap.add_argument("--task", default="", help="只导这个 task")
    ap.add_argument("--obs", default="both",
                    help="observation.state 组成，逗号组合：skeleton(63) / joints / "
                         "hand(真机本体感 20)；both = skeleton,joints")
    ap.add_argument("--fps", type=int, default=0, help="0=按 quality.json 的中位帧率取整")
    ap.add_argument("--robot-type", default="wuji_hand")
    ap.add_argument("--no-qc-filter", action="store_true", help="不要求 quality.json pass")
    ap.add_argument("--include-failures", action="store_true",
                    help="把 success=false 的 episode 也导进去")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="允许把不同 task/side/手模型的 episode 混进同一个数据集")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要导出的内容")
    ap.add_argument("--verify", default="", metavar="DATASET_ROOT",
                    help="不导出，改为把已有数据集用 LeRobotDataset 读回来自检")
    args = ap.parse_args()

    if args.verify:
        return verify(args.verify, args.repo_id)

    kept, skipped = collect(args.input, args.task or None, not args.no_qc_filter,
                            args.include_failures, args.obs)
    if not kept:
        print("没有可导出的 episode。跳过明细：")
        for ep, why in skipped:
            print("  %-40s %s" % (os.path.basename(ep), why))
        return 1

    # ---- 溯源分组守卫 ----
    groups = {}
    for ep, meta, frames in kept:
        adim = len(frames[0]["action"])
        odim = len(_obs_vector(frames[0], args.obs))
        groups.setdefault(group_key(meta, adim, odim), []).append((ep, meta, frames))
    if len(groups) > 1 and not args.allow_mixed:
        print("发现 %d 个不同的 (task, side, hand_model_path, action_dim, obs_dim) 组合，"
              "拒绝混合导出：" % len(groups))
        for k, v in groups.items():
            print("  task=%s side=%s model=%s action_dim=%s obs_dim=%s  → %d 条"
                  % (k[0], k[1], k[2], k[3], k[4], len(v)))
        print("\nretarget 输出依赖加载的手 URDF，混在一起训会学出平均态。"
              "确认要混请加 --allow-mixed，或用 --task 分开导。")
        return 2

    eps = [x for v in groups.values() for x in v]
    adim = len(eps[0][2][0]["action"])
    odim = len(_obs_vector(eps[0][2][0], args.obs))
    n_ja = len(eps[0][2][0].get("joint_angles") or [])
    total = sum(len(f) for _, _, f in eps)

    # fps：优先 quality.json 里量出来的中位帧率
    fps = args.fps
    if not fps:
        rates = []
        for ep, _, _ in eps:
            qp = quality_path(ep)
            if os.path.isfile(qp):
                with open(qp) as f:
                    r = json.load(f).get("metrics", {}).get("rate_hz_median")
                if r:
                    rates.append(r)
        fps = int(round(sum(rates) / len(rates))) if rates else 30
    task_name = eps[0][1].get("task") or "teleop"
    joint_names = (eps[0][1].get("action_space") or {}).get("joint_names")

    print("将导出 %d 条 episode / %d 帧 → %s" % (len(eps), total, args.out))
    print("  repo_id=%s fps=%d obs=%s(dim=%d) action_dim=%d task=%s"
          % (args.repo_id, fps, args.obs, odim, adim, task_name))
    for ep, meta, frames in eps:
        print("  %-34s %5d 帧  success=%s" % (os.path.basename(ep), len(frames),
                                              meta.get("success")))
    if skipped:
        print("跳过 %d 条：" % len(skipped))
        for ep, why in skipped:
            print("    %-34s %s" % (os.path.basename(ep), why))
    if args.dry_run:
        return 0

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if os.path.exists(args.out):
        if not args.overwrite:
            print("\n输出目录已存在：%s（加 --overwrite 覆盖）" % args.out)
            return 3
        shutil.rmtree(args.out)

    features = {
        "observation.state": {"dtype": "float32", "shape": (odim,),
                              "names": _obs_names(args.obs, n_ja, joint_names)},
        "action": {"dtype": "float32", "shape": (adim,), "names": joint_names},
        # 真实设备时间戳：LeRobot 自己按 frame_index/fps 生成均匀 timestamp，
        # 而我们的帧列有 ~2% 空洞，把设备时钟原样存一列，事后能查真实间隔。
        "observation.timestamp_dev": {"dtype": "float32", "shape": (1,), "names": None},
    }

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=features,
                               root=args.out, robot_type=args.robot_type,
                               use_videos=False)
    for ep, meta, frames in eps:
        t0 = frames[0].get("t_dev_us") or 0
        for fr in frames:
            ds.add_frame({
                "observation.state": np.asarray(_obs_vector(fr, args.obs), np.float32),
                "action": np.asarray(fr["action"], np.float32),
                "observation.timestamp_dev": np.asarray(
                    [((fr.get("t_dev_us") or t0) - t0) / 1e6], np.float32),
                "task": meta.get("task") or task_name,
            })
        ds.save_episode()
        print("  saved %s (%d 帧)" % (os.path.basename(ep), len(frames)))

    # 溯源清单：哪些 episode、什么身份、什么 SDK 版本进了这个数据集
    prov = {
        "source_input": os.path.abspath(args.input),
        "obs_mode": args.obs, "fps": fps, "allow_mixed": args.allow_mixed,
        "episodes": [{
            "episode_id": m.get("episode_id"), "path": os.path.abspath(e),
            "frames": len(f), "success": m.get("success"),
            "urdf_source": m.get("urdf_source"),
            "urdf_source_path": m.get("urdf_source_path"),
            "hand_model": m.get("hand_model"),
            "calibrated": m.get("calibrated"), "user_id": m.get("user_id"),
            "sdk_version": m.get("sdk_version"), "glove_sn": m.get("glove_sn"),
        } for e, m, f in eps],
    }
    with open(os.path.join(args.out, "wuji_provenance.json"), "w") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)

    print("\n完成：%d 条 / %d 帧 → %s" % (len(eps), total, args.out))
    print("校验： ./venv312/bin/python tools/export_dataset.py --verify %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
